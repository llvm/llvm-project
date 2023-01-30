add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertFromLLVMIR.cpp
  ConvertToLLVMIR.cpp
  DebugTranslation.cpp
  DebugImporter.cpp
  ModuleTranslation.cpp
  ModuleImport.cpp
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
  TargetParser

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRTranslateLib
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
  DebugImporter.cpp
  ModuleImport.cpp
  TypeFromLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRTranslateLib
  )

add_mlir_translation_library(MLIRFromLLVMIRTranslationRegistration
  ConvertFromLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMIRToLLVMTranslation
  )
