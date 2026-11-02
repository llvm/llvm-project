set(LLVM_OPTIONAL_SOURCES ByteCode.cpp)

set(pdl_header_deps)
if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  list(APPEND pdl_header_deps MLIRPDLOpsIncGen MLIRPDLInterpOpsIncGen)
endif()

add_mlir_library(MLIRRewrite
  FrozenRewritePatternSet.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

  DEPENDS
  mlir-generic-headers
  MLIRConversionPassIncGen
  ${pdl_header_deps}

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
    MLIRAnalysisBase
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

