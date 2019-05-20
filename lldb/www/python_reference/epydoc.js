function toggle_private() {
        // Search for any private/public links on this page.  Store
        // their old text in "cmd," so we will know what action to
        // take; and change their text to the opposite action.
        var cmd = "?";
        var elts = document.getElementsByTagName("a");
        for(var i=0; i<elts.length; i++) {
          if (elts[i].className == "privatelink") {
            cmd = elts[i].innerHTML;
            elts[i].innerHTML = ((cmd && cmd.substr(0,4)=="show")?
                                    "hide&nbsp;private":"show&nbsp;private");
          }
        }
        // Update all DIVs containing private objects.
        var elts = document.getElementsByTagName("div");
        for(var i=0; i<elts.length; i++) {
          if (elts[i].className == "private") {
            elts[i].style.display = ((cmd && cmd.substr(0,4)=="hide")?"none":"block");
          }
          else if (elts[i].className == "public") {
            elts[i].style.display = ((cmd && cmd.substr(0,4)=="hide")?"block":"none");
          }
        }
        // Update all table rows containing private objects.  Note, we
        // use "" instead of "block" becaue IE & firefox disagree on what
        // this should be (block vs table-row), and "" just gives the
        // default for both browsers.
        var elts = document.getElementsByTagName("tr");
        for(var i=0; i<elts.length; i++) {
          if (elts[i].className == "private") {
            elts[i].style.display = ((cmd && cmd.substr(0,4)=="hide")?"none":"");
          }
        }
        // Update all list items containing private objects.
        var elts = document.getElementsByTagName("li");
        for(var i=0; i<elts.length; i++) {
          if (elts[i].className == "private") {
            elts[i].style.display = ((cmd && cmd.substr(0,4)=="hide")?
                                        "none":"");
          }
        }
        // Update all list items containing private objects.
        var elts = document.getElementsByTagName("ul");
        for(var i=0; i<elts.length; i++) {
          if (elts[i].className == "private") {
            elts[i].style.display = ((cmd && cmd.substr(0,4)=="hide")?"none":"block");
          }
        }
        // Set a cookie to remember the current option.
        document.cookie = "EpydocPrivate="+cmd;
      }
function show_private() {
        var elts = document.getElementsByTagName("a");
        for(var i=0; i<elts.length; i++) {
          if (elts[i].className == "privatelink") {
            cmd = elts[i].innerHTML;
            if (cmd && cmd.substr(0,4)=="show")
                toggle_private();
          }
        }
      }
function getCookie(name) {
        var dc = document.cookie;
        var prefix = name + "=";
        var begin = dc.indexOf("; " + prefix);
        if (begin == -1) {
          begin = dc.indexOf(prefix);
          if (begin != 0) return null;
        } else
        { begin += 2; }
        var end = document.cookie.indexOf(";", begin);
        if (end == -1)
        { end = dc.length; }
        return unescape(dc.substring(begin + prefix.length, end));
      }
function setFrame(url1, url2) {
          parent.frames[1].location.href = url1;
          parent.frames[2].location.href = url2;
      }
function checkCookie() {
        var cmd=getCookie("EpydocPrivate");
        if (cmd && cmd.substr(0,4)!="show" && location.href.indexOf("#_") < 0)
            toggle_private();
      }
function toggleCallGraph(id) {
        var elt = document.getElementById(id);
        if (elt.style.display == "none")
            elt.style.display = "block";
        else
            elt.style.display = "none";
      }
function expand(id) {
  var elt = document.getElementById(id+"-expanded");
  if (elt) elt.style.display = "block";
  var elt = document.getElementById(id+"-expanded-linenums");
  if (elt) elt.style.display = "block";
  var elt = document.getElementById(id+"-collapsed");
  if (elt) { elt.innerHTML = ""; elt.style.display = "none"; }
  var elt = document.getElementById(id+"-collapsed-linenums");
  if (elt) { elt.innerHTML = ""; elt.style.display = "none"; }
  var elt = document.getElementById(id+"-toggle");
  if (elt) { elt.innerHTML = "-"; }
}

function collapse(id) {
  var elt = document.getElementById(id+"-expanded");
  if (elt) elt.style.display = "none";
  var elt = document.getElementById(id+"-expanded-linenums");
  if (elt) elt.style.display = "none";
  var elt = document.getElementById(id+"-collapsed-linenums");
  if (elt) { elt.innerHTML = "<br />"; elt.style.display="block"; }
  var elt = document.getElementById(id+"-toggle");
  if (elt) { elt.innerHTML = "+"; }
  var elt = document.getElementById(id+"-collapsed");
  if (elt) {
    elt.style.display = "block";
    
    var indent = elt.getAttribute("indent");
    var pad = elt.getAttribute("pad");
    var s = "<tt class='py-lineno'>";
    for (var i=0; i<pad.length; i++) { s += "&nbsp;" }
    s += "</tt>";
    s += "&nbsp;&nbsp;<tt class='py-line'>";
    for (var i=0; i<indent.length; i++) { s += "&nbsp;" }
    s += "<a href='#' onclick='expand(\"" + id;
    s += "\");return false'>...</a></tt><br />";
    elt.innerHTML = s;
  }
}

function toggle(id) {
  elt = document.getElementById(id+"-toggle");
  if (elt.innerHTML == "-")
      collapse(id); 
  else
      expand(id);
  return false;
}

function highlight(id) {
  var elt = document.getElementById(id+"-def");
  if (elt) elt.className = "py-highlight-hdr";
  var elt = document.getElementById(id+"-expanded");
  if (elt) elt.className = "py-highlight";
  var elt = document.getElementById(id+"-collapsed");
  if (elt) elt.className = "py-highlight";
}

function num_lines(s) {
  var n = 1;
  var pos = s.indexOf("\n");
  while ( pos > 0) {
    n += 1;
    pos = s.indexOf("\n", pos+1);
  }
  return n;
}

// Collapse all blocks that mave more than `min_lines` lines.
function collapse_all(min_lines) {
  var elts = document.getElementsByTagName("div");
  for (var i=0; i<elts.length; i++) {
    var elt = elts[i];
    var split = elt.id.indexOf("-");
    if (split > 0)
      if (elt.id.substring(split, elt.id.length) == "-expanded")
        if (num_lines(elt.innerHTML) > min_lines)
          collapse(elt.id.substring(0, split));
  }
}

function expandto(href) {
  var start = href.indexOf("#")+1;
  if (start != 0 && start != href.length) {
    if (href.substring(start, href.length) != "-") {
      collapse_all(4);
      pos = href.indexOf(".", start);
      while (pos != -1) {
        var id = href.substring(start, pos);
        expand(id);
        pos = href.indexOf(".", pos+1);
      }
      var id = href.substring(start, href.length);
      expand(id);
      highlight(id);
    }
  }
}

function kill_doclink(id) {
  var parent = document.getElementById(id);
  parent.removeChild(parent.childNodes.item(0));
}
function auto_kill_doclink(ev) {
  if (!ev) var ev = window.event;
  if (!this.contains(ev.toElement)) {
    var parent = document.getElementById(this.parentID);
    parent.removeChild(parent.childNodes.item(0));
  }
}

function doclink(id, name, targets_id) {
  var elt = document.getElementById(id);

  // If we already opened the box, then destroy it.
  // (This case should never occur, but leave it in just in case.)
  if (elt.childNodes.length > 1) {
    elt.removeChild(elt.childNodes.item(0));
  }
  else {
    // The outer box: relative + inline positioning.
    var box1 = document.createElement("div");
    box1.style.position = "relative";
    box1.style.display = "inline";
    box1.style.top = 0;
    box1.style.left = 0;
  
    // A shadow for fun
    var shadow = document.createElement("div");
    shadow.style.position = "absolute";
    shadow.style.left = "-1.3em";
    shadow.style.top = "-1.3em";
    shadow.style.background = "#404040";
    
    // The inner box: absolute positioning.
    var box2 = document.createElement("div");
    box2.style.position = "relative";
    box2.style.border = "1px solid #a0a0a0";
    box2.style.left = "-.2em";
    box2.style.top = "-.2em";
    box2.style.background = "white";
    box2.style.padding = ".3em .4em .3em .4em";
    box2.style.fontStyle = "normal";
    box2.onmouseout=auto_kill_doclink;
    box2.parentID = id;

    // Get the targets
    var targets_elt = document.getElementById(targets_id);
    var targets = targets_elt.getAttribute("targets");
    var links = "";
    target_list = targets.split(",");
    for (var i=0; i<target_list.length; i++) {
        var target = target_list[i].split("=");
        links += "<li><a href='" + target[1] + 
               "' style='text-decoration:none'>" +
               target[0] + "</a></li>";
    }
  
    // Put it all together.
    elt.insertBefore(box1, elt.childNodes.item(0));
    //box1.appendChild(box2);
    box1.appendChild(shadow);
    shadow.appendChild(box2);
    box2.innerHTML =
        "Which <b>"+name+"</b> do you want to see documentation for?" +
        "<ul style='margin-bottom: 0;'>" +
        links + 
        "<li><a href='#' style='text-decoration:none' " +
        "onclick='kill_doclink(\""+id+"\");return false;'>"+
        "<i>None of the above</i></a></li></ul>";
  }
  return false;
}

function get_anchor() {
          var href = location.href;
          var start = href.indexOf("#")+1;
          if ((start != 0) && (start != href.length))
              return href.substring(start, href.length);
      }
function redirect_url(dottedName) {
          // Scan through each element of the "pages" list, and check
          // if "name" matches with any of them.
          for (var i=0; i<pages.length; i++) {

              // Each page has the form "<pagename>-m" or "<pagename>-c";
              // extract the <pagename> portion & compare it to dottedName.
              var pagename = pages[i].substring(0, pages[i].length-2);
              if (pagename == dottedName.substring(0,pagename.length)) {

                  // We've found a page that matches `dottedName`;
                  // construct its URL, using leftover `dottedName`
                  // content to form an anchor.
                  var pagetype = pages[i].charAt(pages[i].length-1);
                  var url = pagename + ((pagetype=="m")?"-module.html":
                                                        "-class.html");
                  if (dottedName.length > pagename.length)
                      url += "#" + dottedName.substring(pagename.length+1,
                                                        dottedName.length);
                  return url;
              }
          }
      }
