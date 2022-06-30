set(LLVM_LINK_COMPONENTS
  MC
  MCA
  Object
  Support
  )

set(mca_root ${LLVM_MAIN_SRC_DIR}/tools/llvm-mca)

set(mca_includes
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${mca_root}
  )

# Right now we only need SummaryView.
set(mca_views_sources
  SummaryView.cpp
  )
list(TRANSFORM mca_views_sources PREPEND "${mca_root}/Views/")

set(mca_sources
  MCATestBase.cpp
  ${mca_views_sources}
  )

function(add_llvm_mca_unittest_includes)
  set(mca_includes ${mca_includes} ${ARGV} PARENT_SCOPE)
endfunction()

function(add_llvm_mca_unittest_sources)
  set(sources ${ARGV})
  list(TRANSFORM sources PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
  set(mca_sources ${mca_sources} ${sources} PARENT_SCOPE)
endfunction()

function(add_llvm_mca_unittest_link_components comps)
  set(LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS} ${ARGV} PARENT_SCOPE)
endfunction()

if(LLVM_TARGETS_TO_BUILD MATCHES "X86")
  include(X86/CMakeLists.txt)
endif()

list(REMOVE_DUPLICATES LLVM_LINK_COMPONENTS)

include_directories(${mca_includes})

add_llvm_target_unittest(LLVMMCATests
  ${mca_sources}
  )

set_property(TARGET LLVMMCATests PROPERTY FOLDER "Tests/UnitTests/ToolTests")
