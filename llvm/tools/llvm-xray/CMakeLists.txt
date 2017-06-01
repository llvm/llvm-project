set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  Object
  Support
  Symbolize
  XRay)

set(LLVM_XRAY_TOOLS
  func-id-helper.cc
  xray-account.cc
  xray-converter.cc
  xray-extract.cc
  xray-extract.cc
  xray-registry.cc)

add_llvm_tool(llvm-xray llvm-xray.cc ${LLVM_XRAY_TOOLS})
