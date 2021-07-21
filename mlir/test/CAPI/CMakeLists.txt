function(_add_capi_test_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})
  set(LLVM_LINK_COMPONENTS
    )
  add_llvm_executable(${name}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})
  target_link_libraries(${name}
    ${ARG_LINK_LIBS})
endfunction(_add_capi_test_executable)

_add_capi_test_executable(mlir-capi-execution-engine-test
  execution_engine.c
LINK_LIBS PRIVATE
  MLIRCAPIConversion
  MLIRCEXECUTIONENGINE
  MLIRCAPIRegistration
)

_add_capi_test_executable(mlir-capi-ir-test
  ir.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIStandard
    MLIRCAPIRegistration
)

_add_capi_test_executable(mlir-capi-llvm-test
  llvm.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPILLVM
    MLIRCAPIRegistration
)

_add_capi_test_executable(mlir-capi-pass-test
  pass.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIRegistration
    MLIRCAPITransforms
)

_add_capi_test_executable(mlir-capi-sparse-tensor-test
  sparse_tensor.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIRegistration
    MLIRCAPISparseTensor
)
