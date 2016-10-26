set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  Object)

set(LLVM_XRAY_TOOLS
  xray-extract.cc
  xray-registry.cc)

add_llvm_tool(llvm-xray llvm-xray.cc ${LLVM_XRAY_TOOLS})
