set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  JitRunner.cpp
  MlirOptMain.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp
)

add_mlir_library(MLIRSupport
  FileUtilities.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB})

add_mlir_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRParser
  MLIRSupport
  )

# Exclude from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.
add_mlir_library(MLIRJitRunner
  JitRunner.cpp

  DEPENDS
  intrinsics_gen

  EXCLUDE_FROM_LIBMLIR

  LINK_COMPONENTS
  Core
  OrcJIT
  JITLink
  Support

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRStandardToLLVM
  MLIRSupport
)
