set (files
  "opt-diff.py"
  "opt-stats.py"
  "opt-viewer.py"
  "optpmap.py"
  "optrecord.py"
  "style.css")

foreach (file ${files})
  install(PROGRAMS ${file}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/opt-viewer"
    COMPONENT opt-viewer)
endforeach (file)

add_custom_target(opt-viewer DEPENDS ${files})
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-opt-viewer"
                           DEPENDS opt-viewer
                           COMPONENT opt-viewer)
endif()
