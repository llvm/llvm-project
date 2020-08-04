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
  MLIRLLVMIRTransforms
  MLIRTranslation
  )

add_mlir_translation_library(MLIRTargetAVX512
  LLVMIR/LLVMAVX512Intr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  MLIRLLVMAVX512ConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMAVX512
  MLIRLLVMIR
  MLIRTargetLLVMIRModuleTranslation
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
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRTargetNVVMIR
  LLVMIR/ConvertToNVVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRTargetROCDLIR
  LLVMIR/ConvertToROCDLIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRTargetLLVMIRModuleTranslation
  )
