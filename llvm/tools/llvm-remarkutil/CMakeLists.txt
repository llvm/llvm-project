set(LLVM_LINK_COMPONENTS
  Demangle
  Remarks
  Support
  )

# Multiple files define the same cl::opt variables and include a shared
# header with non-inline class definitions, causing redefinition errors.
set(CMAKE_UNITY_BUILD OFF)

add_llvm_tool(llvm-remarkutil
  RemarkConvert.cpp
  RemarkCount.cpp
  RemarkCounter.cpp
  RemarkFilter.cpp
  RemarkInstructionMix.cpp
  RemarkSizeDiff.cpp
  RemarkSummary.cpp
  RemarkUtil.cpp
  RemarkUtilHelpers.cpp
  RemarkUtilRegistry.cpp
  )
