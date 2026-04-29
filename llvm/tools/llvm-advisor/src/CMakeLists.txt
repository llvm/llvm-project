set(LLVM_LINK_COMPONENTS
  Support
)

file(GLOB_RECURSE LLVM_ADVISOR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Storage/StorageManager.h)
  list(FILTER LLVM_ADVISOR_SOURCES EXCLUDE REGEX ".*/Core/JobManager\\.cpp$")
endif()

add_llvm_tool(llvm-advisor
  ${LLVM_ADVISOR_SOURCES}
)

target_include_directories(llvm-advisor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
