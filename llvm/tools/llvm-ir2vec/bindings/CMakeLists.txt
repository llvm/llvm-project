find_package(Python3 COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)

if(Python3_FOUND AND pybind11_FOUND)
  # Required for Python shared library
  set_target_properties(ir2vec_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
  
  pybind11_add_module(py_ir2vec MODULE ir2vec_bindings.cpp)
  target_link_libraries(py_ir2vec PRIVATE ir2vec_utils)
  
  message(STATUS "Python bindings for llvm-ir2vec will be built")
endif()