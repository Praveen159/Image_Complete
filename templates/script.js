$( document ).ready(function() {
    $("#image_gif").hide();
    $("#success_text").hide();

    
    // function preview_images() {
    //     var total_file = document.getElementById("images").files.length;
    //     for (var i = 0; i < total_file; i++) {
    //       $('#image_preview').append("<img class='img-responsive' src='" + URL.createObjectURL(event.target.files[i]) + "'>");
    //     }
    //   }
  
    //   function onSubmit() {
  
    //   }

      $("#images").change(function(){
        var total_file = document.getElementById("images").files.length;
        for (var i = 0; i < total_file; i++) {
          $('#image_preview').append("<img class='img-responsive' id='car' src='" + URL.createObjectURL(event.target.files[i]) + "'>");
        }
    });

    $("#uploadImage").click(function(){
        $("#image_gif").show();
        setTimeout(function(){
            $("#image_gif").hide();
            $("#success_text").show(); 
        }, 3000);
    });

});