if( LLVM_USE_SANITIZE_COVERAGE )

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
  set(LLVM_LINK_COMPONENTS
      AsmParser
      BitWriter
      Core
      Support
      )
  add_llvm_tool(llvm-as-fuzzer
                llvm-as-fuzzer.cpp)
endif()
