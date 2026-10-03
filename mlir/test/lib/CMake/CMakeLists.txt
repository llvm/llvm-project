function(_mlir_assert_no_pch_reuse target)
  get_target_property(pch_reuse ${target} PRECOMPILE_HEADERS_REUSE_FROM)
  if(pch_reuse)
    message(FATAL_ERROR
      "${target} unexpectedly reuses the PCH from ${pch_reuse}")
  endif()
endfunction()

# Verify that a dependency added after target creation cannot override an
# explicit PCH-reuse opt-out.
add_mlir_library(MLIRTestPCHExplicitOptOut
  Inputs/PCHOptOut.cpp

  PARTIAL_SOURCES_INTENDED
  EXCLUDE_FROM_LIBMLIR
  DISABLE_INSTALL
  DISABLE_PCH_REUSE
  )
set_target_properties(MLIRTestPCHExplicitOptOut PROPERTIES
  EXCLUDE_FROM_ALL TRUE)
mlir_target_link_libraries(MLIRTestPCHExplicitOptOut PRIVATE MLIRIR)
_mlir_assert_no_pch_reuse(MLIRTestPCHExplicitOptOut)

# Mixed C and C++ targets cannot reuse a C++ PCH. Exercise the object-library
# path because the delayed-link helper applies PCH properties to that target.
add_mlir_library(MLIRTestPCHMixedLanguageOptOut
  Inputs/PCHOptOut.c
  Inputs/PCHOptOut.cpp

  PARTIAL_SOURCES_INTENDED
  OBJECT
  EXCLUDE_FROM_LIBMLIR
  DISABLE_INSTALL
  )
set_target_properties(MLIRTestPCHMixedLanguageOptOut PROPERTIES
  EXCLUDE_FROM_ALL TRUE)
mlir_target_link_libraries(MLIRTestPCHMixedLanguageOptOut PRIVATE MLIRIR)
_mlir_assert_no_pch_reuse(obj.MLIRTestPCHMixedLanguageOptOut)
