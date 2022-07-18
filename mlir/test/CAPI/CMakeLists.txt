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
  if(MLIR_BUILD_MLIR_C_DYLIB)
    target_link_libraries(${name} PRIVATE
      MLIR-C)
  else()
    target_link_libraries(${name} PRIVATE
      ${ARG_LINK_LIBS})
  endif()
endfunction(_add_capi_test_executable)

# Only enable the ExecutionEngine if the native target is configured in.
if(TARGET ${LLVM_NATIVE_ARCH})
  _add_capi_test_executable(mlir-capi-execution-engine-test
    execution_engine.c
  LINK_LIBS PRIVATE
    MLIRCAPIConversion
    MLIRCAPIExecutionEngine
    MLIRCAPIRegisterEverything
)
endif()

_add_capi_test_executable(mlir-capi-ir-test
  ir.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIFunc
    MLIRCAPIRegisterEverything
)

_add_capi_test_executable(mlir-capi-llvm-test
  llvm.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPILLVM
    MLIRCAPIRegisterEverything
)

_add_capi_test_executable(mlir-capi-pass-test
  pass.c
  LINK_LIBS PRIVATE
    MLIRCAPIFunc
    MLIRCAPIIR
    MLIRCAPIRegisterEverything
    MLIRCAPITransforms
)

_add_capi_test_executable(mlir-capi-sparse-tensor-test
  sparse_tensor.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIRegisterEverything
    MLIRCAPISparseTensor
)

_add_capi_test_executable(mlir-capi-quant-test
  quant.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIRegisterEverything
    MLIRCAPIQuant
)

_add_capi_test_executable(mlir-capi-pdl-test
  pdl.c
  LINK_LIBS PRIVATE
    MLIRCAPIIR
    MLIRCAPIRegisterEverything
    MLIRCAPIPDL
)
