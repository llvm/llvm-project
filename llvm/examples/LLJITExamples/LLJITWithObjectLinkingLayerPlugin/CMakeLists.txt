set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  JITLink
  OrcJIT
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithObjectLinkingLayerPlugin
  LLJITWithObjectLinkingLayerPlugin.cpp
  )
