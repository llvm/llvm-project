add_subdirectory(CAPI)
add_subdirectory(EDSC)
add_subdirectory(mlir-cpu-runner)
add_subdirectory(SDBM)
add_subdirectory(lib)

llvm_canonicalize_cmake_booleans(
  MLIR_BINDINGS_PYTHON_ENABLED
  LLVM_BUILD_EXAMPLES
  MLIR_CUDA_CONVERSIONS_ENABLED
  MLIR_CUDA_RUNNER_ENABLED
  MLIR_ROCM_CONVERSIONS_ENABLED
  MLIR_ROCM_RUNNER_ENABLED
  MLIR_VULKAN_RUNNER_ENABLED
  )

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for linalg integration tests.
set(MLIR_DIALECT_LINALG_INTEGRATION_TEST_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(MLIR_RUNNER_UTILS_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for the mlir cuda / rocm / spirv / vulkan runner tests.
set(MLIR_CUDA_WRAPPER_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(MLIR_ROCM_WRAPPER_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(MLIR_SPIRV_WRAPPER_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(MLIR_VULKAN_WRAPPER_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(MLIR_TEST_DEPENDS
  FileCheck count not
  MLIRUnitTests
  mlir-capi-ir-test
  mlir-capi-pass-test
  mlir-cpu-runner
  mlir-edsc-builder-api-test
  mlir-linalg-ods-gen
  mlir-opt
  mlir-reduce
  mlir-sdbm-api-test
  mlir-tblgen
  mlir-translate
  mlir_test_cblas
  mlir_test_cblas_interface
  mlir_runner_utils
  mlir_c_runner_utils
  mlir_async_runtime
  )

if(LLVM_BUILD_EXAMPLES)
  list(APPEND MLIR_TEST_DEPENDS
    toyc-ch1
    toyc-ch2
    toyc-ch3
    toyc-ch4
    toyc-ch5
    toyc-ch6
    toyc-ch7
    )
endif()

if(MLIR_CUDA_RUNNER_ENABLED)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-cuda-runner
  )
endif()

if(MLIR_ROCM_RUNNER_ENABLED)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-rocm-runner
  )
endif()

if(MLIR_SPIRV_CPU_RUNNER_ENABLED)
  add_subdirectory(mlir-spirv-cpu-runner)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-spirv-cpu-runner
    mlir_test_spirv_cpu_runner_c_wrappers
  )
endif()

if(MLIR_VULKAN_RUNNER_ENABLED)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-vulkan-runner
  )
endif()

if(MLIR_BINDINGS_PYTHON_ENABLED)
  list(APPEND MLIR_TEST_DEPENDS
    MLIRBindingsPythonExtension
    MLIRTransformsBindingsPythonExtension
  )
endif()

add_lit_testsuite(check-mlir "Running the MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
  )
set_target_properties(check-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
)
