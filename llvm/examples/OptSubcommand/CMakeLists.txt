# Set the .td file to be processed for this target.
set(LLVM_TARGET_DEFINITIONS Opts.td)

tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(HelloSubTableGen)

set(LLVM_LINK_COMPONENTS  
  Support
  Option  
  )

add_llvm_example(OptSubcommand
  llvm-hello-sub.cpp  
  )

target_include_directories(OptSubcommand
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  )
