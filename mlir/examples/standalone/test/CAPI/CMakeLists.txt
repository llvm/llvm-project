# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own standalone dialect.
add_mlir_aggregate(StandaloneCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIRegistration
  StandaloneCAPI
)

add_llvm_executable(standalone-capi-test
  standalone-capi-test.c
)
llvm_update_compile_flags(standalone-capi-test)
target_link_libraries(standalone-capi-test
  PRIVATE StandaloneCAPITestLib)
