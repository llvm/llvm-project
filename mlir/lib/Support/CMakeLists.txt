set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  JitRunner.cpp
  MlirOptMain.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp
  TranslateClParser.cpp
)

add_llvm_library(MLIRSupport
  FileUtilities.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS
  LLVMSupport
  ${LLVM_PTHREAD_LIB}
  )

add_llvm_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  LLVMSupport
  )

add_llvm_library(MLIRTranslateClParser
  TranslateClParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS
  LLVMSupport
  MLIRAnalysis
  MLIRIR
  MLIRTranslation
  MLIRParser
  )

add_llvm_library(MLIRJitRunner
  JitRunner.cpp

  LINK_LIBS
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMIR
  MLIRParser
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRStandardToLLVM
  MLIRSupport
  LLVMCore
  LLVMOrcJIT
  LLVMSupport
  LLVMX86CodeGen
  LLVMX86Desc
  LLVMX86Info
  )
