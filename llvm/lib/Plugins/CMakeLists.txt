set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS PluginsOptions.td)
tablegen(LLVM PluginsOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(PluginsOptionsTableGen)

add_llvm_component_library(LLVMPlugins
  PassPlugin.cpp

  DEPENDS
  PluginsOptionsTableGen

  LINK_COMPONENTS
  Option
  Support
)
