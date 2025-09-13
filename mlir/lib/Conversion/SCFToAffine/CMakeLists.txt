
#set(LLVM_TARGET_DEFINITIONS SCFToAffinePasses.td)

#mlir_tablegen(SCFToAffinePasses.h.inc --gen-pass-decls -name SCFToAffine)
#mlir_tablegen(SCFToAffinePasses.cpp.inc --gen-pass-defs -name SCFToAffine)

#add_public_tablegen_target(MLIRSCFToAffinePassIncGen)

add_mlir_conversion_library(MLIRSCFToAffine
  RaiseToAffine.cpp
  AffineCFG.cpp
  Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToAffine

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAffineDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)
