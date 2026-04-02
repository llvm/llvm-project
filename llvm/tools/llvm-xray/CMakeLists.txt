set(LLVM_LINK_COMPONENTS
  Object
  Support
  Symbolize
  XRay
  )

# Multiple xray-*.cpp files define cl::opt variables with the same names
# (Unused, Demangle, NoDemangle) causing redefinition errors in unity builds.
set(CMAKE_UNITY_BUILD OFF)

add_llvm_tool(llvm-xray
  func-id-helper.cpp
  llvm-xray.cpp
  xray-account.cpp
  xray-color-helper.cpp
  xray-converter.cpp
  xray-extract.cpp
  xray-fdr-dump.cpp
  xray-graph-diff.cpp
  xray-graph.cpp
  xray-registry.cpp
  xray-stacks.cpp
  )
