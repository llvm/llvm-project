set(LLVM_LINK_COMPONENTS
  Demangle
  Remarks
  Support
  )

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
