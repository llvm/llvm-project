set(LLVM_LINK_COMPONENTS
  asmparser
  core
  ipa
  )

set(VMCoreSources
  ConstantsTest.cpp
  DominatorTreeTest.cpp
  IRBuilderTest.cpp
  InstructionsTest.cpp
  MDBuilderTest.cpp
  MetadataTest.cpp
  PassManagerTest.cpp
  TypeBuilderTest.cpp
  TypesTest.cpp
  ValueMapTest.cpp
  VerifierTest.cpp
  )

# MSVC9 and 8 cannot compile ValueMapTest.cpp due to their bug.
# See issue#331418 in Visual Studio.
if(MSVC AND MSVC_VERSION LESS 1600)
  list(REMOVE_ITEM VMCoreSources ValueMapTest.cpp)
endif()

# HACK: Declare a couple of source files as optionally compiled to satisfy the
# missing-file-checker in LLVM's weird CMake build.
set(LLVM_OPTIONAL_SOURCES
  ValueMapTest.cpp
  )

add_llvm_unittest(VMCoreTests
  ${VMCoreSources}
  )
