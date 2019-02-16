set(LLVM_LINK_COMPONENTS
  OptRemarks
  )

set(SOURCES
  liboptremarks.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/OptRemarks.exports)

add_llvm_library(OptRemarks SHARED ${SOURCES})

install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/OptRemarks.h
  DESTINATION include/llvm-c
  COMPONENT OptRemarks)

if (APPLE)
  set(OPTREMARKS_VERSION ${LLVM_VERSION_MAJOR})
  set_property(TARGET OptRemarks APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version 1 -current_version ${OPTREMARKS_VERSION}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
endif()
