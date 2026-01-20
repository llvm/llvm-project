find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

set_target_properties(LLVMEmbUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(py_ir2vec MODULE ir2vec_bindings.cpp)
target_link_libraries(py_ir2vec PRIVATE LLVMEmbUtils)

message(STATUS "Python bindings for llvm-ir2vec will be built with nanobind")
