add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_EXEGESIS_X86_UNITTEST_SOURCES
  BenchmarkResultTest.cpp
  RegisterAliasingTest.cpp
  SchedClassResolutionTest.cpp
  SnippetFileTest.cpp
  SnippetGeneratorTest.cpp
  SnippetRepetitorTest.cpp
  TargetTest.cpp
)

# Only compile the subprocess memory unittests on x86_64 linux. They only
# sense to run on POSIX systems due to the implementations use of the POSIX
# shared memory APIs and there are currently test failures on certain platforms
# (s390x/PPC) that need further investigation.
# TODO(boomanaiden154): Investigate and fix test failures on PPC
if(LLVM_HOST_TRIPLE MATCHES "^x86_64-.*-linux")
  list(APPEND LLVM_EXEGESIS_X86_UNITTEST_SOURCES SubprocessMemoryTest.cpp)
endif()

add_llvm_exegesis_unittest_sources(
  ${LLVM_EXEGESIS_X86_UNITTEST_SOURCES}
  )

add_llvm_exegesis_unittest_link_components(
  Core
  Codegen
  MC
  MCParser
  Object
  Support
  Symbolize
  X86
  )

add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisX86
  )
