include_directories(include)

add_subdirectory(lib)

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MCA
  MC
  MCParser
  Support
  )

add_llvm_tool(llvm-mca
  llvm-mca.cpp
  CodeRegion.cpp
  CodeRegionGenerator.cpp
  PipelinePrinter.cpp
  Views/BottleneckAnalysis.cpp
  Views/DispatchStatistics.cpp
  Views/InstructionInfoView.cpp
  Views/InstructionView.cpp
  Views/RegisterFileStatistics.cpp
  Views/ResourcePressureView.cpp
  Views/RetireControlUnitStatistics.cpp
  Views/SchedulerStatistics.cpp
  Views/SummaryView.cpp
  Views/TimelineView.cpp
  Views/View.cpp
  )

set(LLVM_MCA_SOURCE_DIR ${CURRENT_SOURCE_DIR})

target_link_libraries(llvm-mca PRIVATE
  ${LLVM_MCA_CUSTOMBEHAVIOUR_TARGETS}
  )

add_definitions(${LLVM_MCA_MACROS_TO_DEFINE})
