set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Object
  Support
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")

add_llvm_executable(llvm-dwarfdump-fuzzer
  EXCLUDE_FROM_ALL
  llvm-dwarfdump-fuzzer.cpp
  )
