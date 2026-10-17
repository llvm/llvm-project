set(LLVM_OPTIONAL_SOURCES ByteCode.cpp)

set(pdl_header_libraries)
if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  set(pdl_header_libraries MLIRPDLInterpDialect)
endif()

add_mlir_library(MLIRRewrite
  FrozenRewritePatternSet.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

  DEPENDS
  mlir-generic-headers
  MLIRConversionPassIncGen

  # FrozenRewritePatternSet.cpp includes PDLInterp and PDL when enabled.
  # PDLInterp's public interface also provides the PDL generated headers.
  HEADER_LIBS
  ${pdl_header_libraries}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )

if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  add_mlir_library(MLIRRewritePDL
    ByteCode.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

    DEPENDS
    mlir-generic-headers

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRSideEffectInterfaces
  )

  target_link_libraries(MLIRRewrite PUBLIC
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRRewritePDL)
endif()
