set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)

set(LLVM_TARGET_DEFINITIONS SubCommandOpts.td)
tablegen(LLVM SubCommandOpts.inc -gen-opt-parser-defs)

add_public_tablegen_target(OptsTestTableGen)

add_llvm_unittest(OptionTests
  OptionParsingTest.cpp
  OptionMarshallingTest.cpp
  OptionSubCommandsTest.cpp
  )
