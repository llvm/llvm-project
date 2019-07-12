set(LLVM_LINK_COMPONENTS
  Remarks
  )

set(SOURCES
  libremarks.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Remarks.exports)

add_llvm_library(Remarks SHARED INSTALL_WITH_TOOLCHAIN ${SOURCES})

install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/Remarks.h
  DESTINATION include/llvm-c
  COMPONENT Remarks)

if (APPLE)
  set(REMARKS_VERSION ${LLVM_VERSION_MAJOR})
  set_property(TARGET Remarks APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version 1 -current_version ${REMARKS_VERSION}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
endif()
