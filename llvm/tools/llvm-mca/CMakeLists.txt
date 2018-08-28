include_directories(include)

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Support
  )

add_llvm_tool(llvm-mca
  llvm-mca.cpp
  CodeRegion.cpp
  PipelinePrinter.cpp
  Views/DispatchStatistics.cpp
  Views/InstructionInfoView.cpp
  Views/RegisterFileStatistics.cpp
  Views/ResourcePressureView.cpp
  Views/RetireControlUnitStatistics.cpp
  Views/SchedulerStatistics.cpp
  Views/SummaryView.cpp
  Views/TimelineView.cpp
  Views/View.cpp
  )

set(LLVM_MCA_SOURCE_DIR ${CURRENT_SOURCE_DIR})
add_subdirectory(lib)
target_link_libraries(llvm-mca PRIVATE LLVMMCA)
