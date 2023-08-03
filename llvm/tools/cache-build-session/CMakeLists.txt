set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(cache-build-session
  CacheBuildSession.cpp
  CMakeFileAPI.cpp
)
