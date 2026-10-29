set(LLVM_OPTIONAL_SOURCES
  OpenMPToLLVM.cpp
  OpenMPToLLVMPass.cpp
  )

# Reusable OpenMP patterns, conversion legality, and dialect interface. Clients
# with their own lowering pipeline do not need the standalone pass's conversions
# for other dialects.
add_mlir_conversion_library(MLIROpenMPToLLVMConversion
  OpenMPToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenMPToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  MLIRConvertToLLVMInterfaceIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIROpenMPDialect
  )

# Keep the existing target as the complete conversion library, including the
# standalone pass and all APIs provided by MLIROpenMPToLLVMConversion.
add_mlir_conversion_library(MLIROpenMPToLLVM
  OpenMPToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenMPToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIROpenMPDialect
  MLIROpenMPToLLVMConversion
  MLIRPass
  MLIRTransformUtils
  )
