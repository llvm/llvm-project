if( LLVM_USE_SANITIZE_COVERAGE )

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
  set(LLVM_LINK_COMPONENTS
      AllTargetsAsmPrinters
      AllTargetsDescs
      AllTargetsDisassemblers
      AllTargetsInfos
      MC
      MCDisassembler
      MCParser
      Support
      )
  add_llvm_tool(llvm-mc-disassemble-fuzzer
                llvm-mc-disassemble-fuzzer.cpp)
endif()
