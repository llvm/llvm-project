add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertFromLLVMIR.cpp
  ConvertToLLVMIR.cpp
  DebugTranslation.cpp
  ModuleTranslation.cpp
  TypeTranslation.cpp
  )


add_mlir_translation_library(MLIRTargetLLVMIRExport
  DebugTranslation.cpp
  ModuleTranslation.cpp
  TypeTranslation.cpp

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

add_mlir_translation_library(MLIRToLLVMIRTranslationRegistration
  ConvertToLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRArmNeonToLLVMIRTranslation
  MLIRAMXToLLVMIRTranslation
  MLIRAVX512ToLLVMIRTranslation
  MLIRLLVMArmSVEToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  )

add_mlir_translation_library(MLIRTargetLLVMIRImport
  ConvertFromLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  MLIRLLVMIR
  MLIRTranslation
  )
