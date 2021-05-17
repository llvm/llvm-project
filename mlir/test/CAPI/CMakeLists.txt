function(_add_capi_test_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    ""
    ${ARGN})
  set(LLVM_LINK_COMPONENTS
    )
  add_llvm_executable(${name}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})
  target_link_libraries(${name}
    PRIVATE
    MLIRPublicAPI)
endfunction(_add_capi_test_executable)

_add_capi_test_executable(mlir-capi-execution-engine-test
  execution_engine.c
DEPENDS
  MLIRConversionPassIncGen
)

_add_capi_test_executable(mlir-capi-ir-test
  ir.c
)

_add_capi_test_executable(mlir-capi-pass-test
  pass.c
)

_add_capi_test_executable(mlir-capi-sparse-tensor-test
  sparse_tensor.c
)
