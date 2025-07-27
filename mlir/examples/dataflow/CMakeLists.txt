add_custom_target(DataFlowExample)
set_target_properties(DataFlowExample PROPERTIES FOLDER "MLIR/Examples")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(lib)
add_subdirectory(dataflow-opt)
