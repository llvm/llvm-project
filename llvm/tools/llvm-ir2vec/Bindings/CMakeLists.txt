if(NOT LLVM_ENABLE_PIC)
  message(FATAL_ERROR "Python bindings require LLVM_ENABLE_PIC=ON. "
                      "Please reconfigure LLVM with -DLLVM_ENABLE_PIC=ON")
endif()

set(BINDINGS_MINIMUM_PYTHON_VERSION 3.10)
find_package(Python ${BINDINGS_MINIMUM_PYTHON_VERSION} COMPONENTS Interpreter Development.Module REQUIRED)

if(nanobind_DIR)
  message(STATUS "Using explicit nanobind cmake directory: ${nanobind_DIR}")
else()
  message(STATUS "Checking for nanobind in python path...")
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
  )
  
  if(NOT STATUS EQUAL "0")
      message(FATAL_ERROR "nanobind not found (install via 'pip install nanobind' or set nanobind_DIR)")
  endif()
  
  message(STATUS "found nanobind at: ${PACKAGE_DIR}")
  set(nanobind_DIR "${PACKAGE_DIR}")
endif()

find_package(nanobind CONFIG REQUIRED)

message(STATUS "Python bindings for ir2vec will be built with nanobind")
nanobind_add_module(ir2vec MODULE PyIR2Vec.cpp)

set(LLVM_IR2VEC_BINDINGS_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/ir2vec" CACHE INTERNAL "")
set_target_properties(ir2vec PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_IR2VEC_BINDINGS_DIR}"
)

# Python bindings require exception handling and RTTI to convert C++ exceptions
# to Python exceptions. LLVM core is compiled with -fno-exceptions -fno-rtti,
# but Python bindings are compiled separately with these features enabled.
# This follows the pattern established by MLIR Python bindings
# (see mlir/cmake/modules/AddMLIRPython.cmake).
target_compile_options(ir2vec PRIVATE -fexceptions -frtti)
target_link_libraries(ir2vec PRIVATE LLVMIREmbUtils)
target_include_directories(ir2vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Stage Python package into build directory so that downstream wheel assembly
# scripts can operate solely from the build tree, without requiring access to
# the source tree.
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/python/"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/")

# Copy the seed embedding vocabulary into the build-time Python package
set(IR2VEC_VOCAB_75D
    "${LLVM_MAIN_SRC_DIR}/lib/Analysis/models/seedEmbeddingVocab75D.json")

if(NOT EXISTS "${IR2VEC_VOCAB_75D}")
  message(FATAL_ERROR
    "IR2Vec vocabulary file not found: ${IR2VEC_VOCAB_75D}")
endif()

file(COPY "${IR2VEC_VOCAB_75D}"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ir2vec/")
message(STATUS "IR2Vec vocabulary copied to ${CMAKE_CURRENT_BINARY_DIR}/python/ir2vec/")
