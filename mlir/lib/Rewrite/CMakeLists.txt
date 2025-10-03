set(LLVM_OPTIONAL_SOURCES ByteCode.cpp)

add_mlir_library(MLIRRewrite
  FrozenRewritePatternSet.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

  DEPENDS
  mlir-generic-headers
  MLIRConversionPassIncGen

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

