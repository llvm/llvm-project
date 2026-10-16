add_subdirectory(Dialect)
add_subdirectory(Transforms)

set(LLVM_OPTIONAL_SOURCES
  ConvertFromLLVMIR.cpp
  ConvertToLLVMIR.cpp
  DataLayoutImporter.cpp
  DebugTranslation.cpp
  DebugImporter.cpp
  LoopAnnotationImporter.cpp
  LoopAnnotationTranslation.cpp
  LLVMImportInterface.cpp
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
  MLIRLLVMTranslationDialectInterfaceIncGen

  LINK_COMPONENTS
  Analysis
  Core
  FrontendOpenMP
  TransformUtils
  TargetParser

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRLLVMIRExportLegalization
  MLIRRewriteUtils
  )

add_mlir_translation_library(MLIRToLLVMIRTranslationRegistration
  ConvertToLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRTranslateLib
  MLIRArmNeonToLLVMIRTranslation
  MLIRArmSMEToLLVMIRTranslation
  MLIRArmSVEToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRPtrToLLVMIRTranslation
  MLIRSPIRVToLLVMIRTranslation
  MLIRVCIXToLLVMIRTranslation
  MLIRXeVMToLLVMIRTranslation
  )

# Data-layout conversion is also used by frontends that do not import LLVM IR.
add_mlir_library(MLIRLLVMIRDataLayoutImporter
  OBJECT
  DataLayoutImporter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRLLVMDialect
  )

add_mlir_translation_library(MLIRTargetLLVMIRImport
  DebugImporter.cpp
  LoopAnnotationImporter.cpp
  ModuleImport.cpp
  LLVMImportInterface.cpp
  TypeFromLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  MLIRLLVMImportDialectInterfaceIncGen

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRLLVMIRDataLayoutImporter
  MLIRTranslateLib
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRTargetLLVMIRImport PRIVATE $<TARGET_OBJECTS:obj.MLIRLLVMIRDataLayoutImporter>)
endif()

add_mlir_translation_library(MLIRFromLLVMIRTranslationRegistration
  ConvertFromLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMIRToLLVMTranslation
  MLIRLLVMIRToNVVMTranslation
  )
