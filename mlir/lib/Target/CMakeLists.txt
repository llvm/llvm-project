add_subdirectory(SPIRV)
add_subdirectory(LLVMIR)

add_mlir_translation_library(MLIRTargetLLVMIRModuleTranslation
  LLVMIR/DebugTranslation.cpp
  LLVMIR/ModuleTranslation.cpp
  LLVMIR/TypeTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  FrontendOpenMP
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRLLVMIR
  MLIROpenMP
  MLIRLLVMIRTransforms
  MLIRTranslation
  )

add_mlir_translation_library(MLIRTargetLLVMIR
  LLVMIR/ConvertFromLLVMIR.cpp
  LLVMIR/ConvertToLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  MLIRLLVMArmNeonToLLVMIRTranslation
  MLIRLLVMArmSVEToLLVMIRTranslation
  MLIRLLVMAVX512ToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRTargetLLVMIRModuleTranslation
  )
