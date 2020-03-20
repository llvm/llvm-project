add_mlir_library(MLIRTargetLLVMIRModuleTranslation
  LLVMIR/DebugTranslation.cpp
  LLVMIR/ModuleTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(MLIRTargetLLVMIRModuleTranslation
  PUBLIC
  MLIRLLVMIR
  MLIRLLVMIRTransforms
  LLVMCore
  LLVMIRReader
  LLVMSupport
  LLVMTransformUtils
  MLIRTranslation
  )

add_mlir_library(MLIRTargetAVX512
  LLVMIR/LLVMAVX512Intr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  DEPENDS
  MLIRLLVMAVX512ConversionsIncGen
  )
target_link_libraries(MLIRTargetAVX512
  PUBLIC
  MLIRIR
  MLIRLLVMAVX512
  MLIRLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_library(MLIRTargetLLVMIR
  LLVMIR/ConvertFromLLVMIR.cpp
  LLVMIR/ConvertToLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  )
target_link_libraries(MLIRTargetLLVMIR
  PUBLIC
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_library(MLIRTargetNVVMIR
  LLVMIR/ConvertToNVVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(MLIRTargetNVVMIR
  PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_library(MLIRTargetROCDLIR
  LLVMIR/ConvertToROCDLIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(MLIRTargetROCDLIR
  PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRTargetLLVMIRModuleTranslation
  )
