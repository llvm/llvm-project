set(files
  "call_graph.py"
  "extract_type_map.py"
  "graph_helper.py"
  "stack_estimator.py"
  "symbolizer.py"
  "consolidate_sorted_table.py")

foreach (file ${files})
  install(PROGRAMS ${file}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/stack-size-estimator"
    COMPONENT stack-size-estimator)
endforeach (file)

add_custom_target(stack-size-estimator DEPENDS ${files})
set_target_properties(stack-size-estimator PROPERTIES FOLDER "LLVM/Tools")
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-stack-size-estimator"
                           DEPENDS stack-size-estimator
                           COMPONENT stack-size-estimator)
endif()
