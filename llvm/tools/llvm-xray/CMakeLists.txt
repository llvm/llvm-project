set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  Object
  Support
  Symbolize
  XRay
  )

set(LLVM_XRAY_TOOLS
  func-id-helper.cpp
  xray-account.cpp
  xray-color-helper.cpp
  xray-converter.cpp
  xray-extract.cpp
  xray-graph.cpp
  xray-graph-diff.cpp
  xray-stacks.cpp
  xray-registry.cpp
  )

add_llvm_tool(llvm-xray
  llvm-xray.cpp
  ${LLVM_XRAY_TOOLS}
  )
