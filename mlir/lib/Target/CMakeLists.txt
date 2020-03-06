add_llvm_library(MLIRTargetLLVMIRModuleTranslation
  LLVMIR/DebugTranslation.cpp
  LLVMIR/ModuleTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(MLIRTargetLLVMIRModuleTranslation
  PUBLIC
  MLIRLLVMIR LLVMCore LLVMIRReader LLVMSupport LLVMTransformUtils
  MLIRTranslation)
add_llvm_library(MLIRTargetLLVMIR
  LLVMIR/ConvertFromLLVMIR.cpp
  LLVMIR/ConvertToLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  )
target_link_libraries(MLIRTargetLLVMIR
  PUBLIC
  MLIRTargetLLVMIRModuleTranslation)
add_llvm_library(MLIRTargetNVVMIR
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
add_llvm_library(MLIRTargetROCDLIR
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
