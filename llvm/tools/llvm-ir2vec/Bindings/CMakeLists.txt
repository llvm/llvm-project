set(BINDINGS_MINIMUM_PYTHON_VERSION 3.10)
find_package(Python ${BINDINGS_MINIMUM_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE STATUS
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT STATUS EQUAL "0")
    message(FATAL_ERROR "nanobind not found (install via 'pip install nanobind' or set nanobind_DIR)")
endif()

find_package(nanobind CONFIG REQUIRED)

set_target_properties(LLVMEmbUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(ir2vec MODULE PyIR2Vec.cpp)
target_link_libraries(ir2vec PRIVATE LLVMEmbUtils)
target_include_directories(ir2vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

message(STATUS "Python bindings for llvm-ir2vec will be built with nanobind")
