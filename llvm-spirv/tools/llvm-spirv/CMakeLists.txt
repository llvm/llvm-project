set(LLVM_LINK_COMPONENTS
  SPIRVLib
  Analysis
  BitReader
  BitWriter
  Core
  Support
  TransformUtils
)

add_llvm_tool(llvm-spirv
  llvm-spirv.cpp
  # llvm_setup_rpath messes with the rpath making llvm-spirv not executable from the build directory
  NO_INSTALL_RPATH
)

if (BUILD_EXTERNAL)
  target_link_libraries(llvm-spirv PRIVATE LLVMSPIRVLib)
endif()

target_include_directories(llvm-spirv
  PRIVATE
    ${LLVM_INCLUDE_DIR}
    ${LLVM_SPIRV_INCLUDE_DIRS}
)
