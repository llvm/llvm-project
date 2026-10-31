set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  GPUInterfaces.cpp
  LLVMInterfaces.cpp
  MemRefInterfaces.cpp
  )

add_mlir_extension_library(MLIROpenACCMemRefInterfaces
  MemRefInterfaces.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIROpenACCDialect
  MLIRMemRefDialect
  )

add_mlir_extension_library(MLIROpenACCLLVMInterfaces
  LLVMInterfaces.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIROpenACCDialect
  MLIRLLVMDialect
  )

add_mlir_extension_library(MLIROpenACCGPUInterfaces
  GPUInterfaces.cpp

  LINK_LIBS PUBLIC
  MLIROpenACCDialect
  MLIRGPUDialect
  )

add_mlir_extension_library(MLIROpenACCAllExtensions
  AllExtensions.cpp

  LINK_LIBS PUBLIC
  MLIROpenACCGPUInterfaces
  MLIROpenACCLLVMInterfaces
  MLIROpenACCMemRefInterfaces
  )
