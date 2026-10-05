set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(SplitFileOptsTableGen)

add_llvm_utility(split-file
  split-file.cpp
  DEPENDS
  SplitFileOptsTableGen
  )
