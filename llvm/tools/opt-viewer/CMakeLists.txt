set (files
  "optpmap.py"
  "style.css")

set (generated_files
  "opt-diff.py"
  "opt-stats.py"
  "opt-viewer.py"
  "optrecord.py")

foreach (file ${generated_files})
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${file}.in
    ${CMAKE_CURRENT_BINARY_DIR}/${file})
endforeach (file)

foreach (file ${files})
  install(PROGRAMS ${file}
    DESTINATION share/opt-viewer
    COMPONENT opt-viewer)
endforeach (file)


foreach (file ${generated_files})
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${file}
    DESTINATION share/opt-viewer
    COMPONENT opt-viewer)
endforeach (file)
