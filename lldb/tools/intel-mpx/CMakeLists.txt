if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  return ()
endif ()

include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

add_library(lldb-intel-mpxtable SHARED
  IntelMPXTablePlugin.cpp
  )

target_link_libraries(lldb-intel-mpxtable
                      PUBLIC liblldb LLVMSupport)

install(TARGETS lldb-intel-mpxtable
  LIBRARY DESTINATION bin)
