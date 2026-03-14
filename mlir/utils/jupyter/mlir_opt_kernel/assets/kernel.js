define([ "codemirror/lib/codemirror", "base/js/namespace" ],
       function(CodeMirror, IPython) {
         "use strict";
         var onload = function() {
           // TODO: Add syntax highlighting.
           console.log("Loading kernel.js from MlirOptKernel");
         };
         return {onload : onload};
       });
