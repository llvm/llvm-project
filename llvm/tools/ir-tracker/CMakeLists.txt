set(files
  "ir-tracker.py"
  "irtrackdb.py"
  "irtrackhtml.py")

foreach(file ${files})
  install(PROGRAMS ${file}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/ir-tracker"
    COMPONENT ir-tracker)
endforeach(file)

add_custom_target(ir-tracker DEPENDS ${files})
set_target_properties(ir-tracker PROPERTIES FOLDER "LLVM/Tools")
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-ir-tracker"
                           DEPENDS ir-tracker
                           COMPONENT ir-tracker)
endif()
