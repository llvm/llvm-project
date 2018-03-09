if (NOT LIBIPT_INCLUDE_PATH)
  message (FATAL_ERROR "libipt include path not provided")
endif()

if (NOT EXISTS "${LIBIPT_INCLUDE_PATH}")
  message (FATAL_ERROR "invalid libipt include path provided")
endif()
include_directories(${LIBIPT_INCLUDE_PATH})

if (NOT LIBIPT_LIBRARY_PATH)
  find_library(LIBIPT_LIBRARY ipt)
else()
  if (NOT EXISTS "${LIBIPT_LIBRARY_PATH}")
    message (FATAL_ERROR "invalid libipt library path provided")
  endif()
  find_library(LIBIPT_LIBRARY ipt PATHS ${LIBIPT_LIBRARY_PATH})
endif()

if (NOT LIBIPT_LIBRARY)
  message (FATAL_ERROR "libipt library not found")
endif()

add_lldb_library(lldbIntelPT
  PTDecoder.cpp
  Decoder.cpp
  cli-wrapper-pt.cpp

  LINK_LIBS
    ${LIBIPT_LIBRARY}
    liblldb
  )
