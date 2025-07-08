cmake_minimum_required(VERSION 3.18)

set(LLVM_TOOL_LLVM_ADVISOR_BUILD_DEFAULT ON)
set(LLVM_REQUIRE_EXE_NAMES llvm-advisor)

add_subdirectory(src)

# Set the executable name
set_target_properties(llvm-advisor PROPERTIES
  OUTPUT_NAME llvm-advisor)

# Install the binary
install(TARGETS llvm-advisor
  RUNTIME DESTINATION bin
  COMPONENT llvm-advisor)
