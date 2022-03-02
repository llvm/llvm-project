add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertFromLLVMIR.cpp
  ConvertToLLVMIR.cpp
  DebugTranslation.cpp
  ModuleTranslation.cpp
  TypeToLLVM.cpp
  TypeFromLLVM.cpp
  )


add_mlir_translation_library(MLIRTargetLLVMIRExport
  DebugTranslation.cpp
  ModuleTranslation.cpp
  TypeToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  FrontendOpenMP
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRDLTI
  MLIRLLVMIR
  MLIRLLVMIRTransforms
  MLIRTranslation
  )

add_mlir_translation_library(MLIRToLLVMIRTranslationRegistration
  ConvertToLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRArmNeonToLLVMIRTranslation
  MLIRArmSVEToLLVMIRTranslation
  MLIRAMXToLLVMIRTranslation
  MLIRX86VectorToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  )

add_mlir_translation_library(MLIRTargetLLVMIRImport
  ConvertFromLLVMIR.cpp
  TypeFromLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  MLIRDLTI
  MLIRLLVMIR
  MLIRTranslation
  )
