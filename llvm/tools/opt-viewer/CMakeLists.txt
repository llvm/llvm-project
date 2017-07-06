set (files
  "opt-diff.py"
  "opt-stats.py"
  "opt-viewer.py"
  "optpmap.py"
  "optrecord.py"
  "style.css")

foreach (file ${files})
  install(PROGRAMS ${file}
    DESTINATION share/opt-viewer
    COMPONENT opt-viewer)
endforeach (file)
