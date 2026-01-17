find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set_target_properties(LLVMEmbUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(py_ir2vec MODULE ir2vec_bindings.cpp)
target_link_libraries(py_ir2vec PRIVATE LLVMEmbUtils)

message(STATUS "Python bindings for llvm-ir2vec will be built")
