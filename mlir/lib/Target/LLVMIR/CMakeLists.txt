add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertFromLLVMIR.cpp
  ConvertToLLVMIR.cpp
  DataLayoutImporter.cpp
  DebugTranslation.cpp
  DebugImporter.cpp
  LoopAnnotationImporter.cpp
  LoopAnnotationTranslation.cpp
  ModuleTranslation.cpp
  ModuleImport.cpp
  TypeToLLVM.cpp
  TypeFromLLVM.cpp
  )


add_mlir_translation_library(MLIRTargetLLVMIRExport
  DebugTranslation.cpp
  LoopAnnotationTranslation.cpp
  ModuleTranslation.cpp
  TypeToLLVM.cpp
  Dialect/OpenMPCommon.cpp

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
  MLIRTransformUtils
  )

add_mlir_translation_library(MLIRToLLVMIRTranslationRegistration
  ConvertToLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRArmNeonToLLVMIRTranslation
  MLIRArmSMEToLLVMIRTranslation
  MLIRArmSVEToLLVMIRTranslation
  MLIRAMXToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRX86VectorToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  )

add_mlir_translation_library(MLIRTargetLLVMIRImport
  DataLayoutImporter.cpp
  DebugImporter.cpp
  LoopAnnotationImporter.cpp
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
