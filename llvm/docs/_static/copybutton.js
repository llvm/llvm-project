// Simple "copy to clipboard" button for code blocks.
// Adds a button to each <div class="highlight"> block.
(function () {
  "use strict";

  function addCopyButton(block) {
    var button = document.createElement("button");
    button.className = "copybutton";
    button.title = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.innerHTML =
      '<svg aria-hidden="true" height="16" viewBox="0 0 16 16" width="16">' +
      '<path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 ' +
      "0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 " +
      "0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z" +
      '"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 ' +
      "1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 " +
      ".138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z" +
      '"/></svg>';

    button.addEventListener("click", function () {
      var code = block.querySelector("pre").innerText;
      navigator.clipboard.writeText(code).then(
        function () {
          button.classList.add("copied");
          setTimeout(function () {
            button.classList.remove("copied");
          }, 2000);
        },
        function () {
          // Fallback for older browsers
          var ta = document.createElement("textarea");
          ta.value = code;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.focus();
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
          button.classList.add("copied");
          setTimeout(function () {
            button.classList.remove("copied");
          }, 2000);
        }
      );
    });

    block.style.position = "relative";
    block.appendChild(button);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("div.highlight").forEach(addCopyButton);
  });
})();
