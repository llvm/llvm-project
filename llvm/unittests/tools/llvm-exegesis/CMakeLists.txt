set(exegesis_includes
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  )

set(exegesis_sources
  BenchmarkRunnerTest.cpp
  ClusteringTest.cpp
  PerfHelperTest.cpp
  RegisterValueTest.cpp
  )

set(exegesis_link_libraries LLVMExegesis)

function(add_llvm_exegesis_unittest_includes)
  set(exegesis_includes ${exegesis_includes} ${ARGV} PARENT_SCOPE)
endfunction()

function(add_llvm_exegesis_unittest_sources)
  set(sources ${ARGV})
  list(TRANSFORM sources PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
  set(exegesis_sources ${exegesis_sources} ${sources} PARENT_SCOPE)
endfunction()

function(add_llvm_exegesis_unittest_link_components comps)
  set(LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS} ${ARGV} PARENT_SCOPE)
endfunction()

function(add_llvm_exegesis_unittest_link_libraries libs)
  set(exegesis_link_libraries ${exegesis_link_libraries} ${ARGV} PARENT_SCOPE)
endfunction()


if(LLVM_TARGETS_TO_BUILD MATCHES "X86")
  include(X86/CMakeLists.txt)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "ARM")
  include(ARM/CMakeLists.txt)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "AArch64")
  include(AArch64/CMakeLists.txt)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "PowerPC")
  include(PowerPC/CMakeLists.txt)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "Mips")
  include(Mips/CMakeLists.txt)
endif()

include_directories(${exegesis_includes})

list(REMOVE_DUPLICATES LLVM_LINK_COMPONENTS)

add_llvm_target_unittest(LLVMExegesisTests
  ${exegesis_sources}
  )
target_link_libraries(LLVMExegesisTests PRIVATE ${exegesis_link_libraries})

set_property(TARGET LLVMExegesisTests PROPERTY FOLDER "Tests/UnitTests/ToolTests")
