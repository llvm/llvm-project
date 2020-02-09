add_mlir_library(MLIRTargetLLVMIRModuleTranslation
  LLVMIR/DebugTranslation.cpp
  LLVMIR/ModuleTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  MLIRLLVMIR
  LLVMCore
  LLVMIRReader
  LLVMSupport
  LLVMTransformUtils
  MLIRIR
  MLIRSupport
  MLIRTranslation
  )

add_mlir_library(MLIRTargetLLVMIR
  LLVMIR/ConvertFromLLVMIR.cpp
  LLVMIR/ConvertToLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  LINK_LIBS
  MLIRIR
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRModuleTranslation
  MLIRTranslation
  LLVMCore
  LLVMIRReader
  LLVMSupport
  )

add_mlir_library(MLIRTargetNVVMIR
  LLVMIR/ConvertToNVVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIRTranslation
  LLVMCore
  LLVMSupport
  )

add_mlir_library(MLIRTargetROCDLIR
  LLVMIR/ConvertToROCDLIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRTargetLLVMIRModuleTranslation
  MLIRTranslation
  LLVMCore
  LLVMSupport
  )
