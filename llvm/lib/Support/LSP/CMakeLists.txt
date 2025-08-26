add_llvm_library(LLVMSupportLSP STATIC
  Protocol.cpp
  Transport.cpp
  Logging.cpp
)

target_include_directories(LLVMSupportLSP
  PUBLIC
    $<BUILD_INTERFACE:${LLVM_MAIN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LLVM_LIBRARY_DIR}/Support/LSP>
    $<INSTALL_INTERFACE:include>   # for installation
)

target_link_libraries(LLVMSupportLSP
  PUBLIC
    LLVMSupport
)
