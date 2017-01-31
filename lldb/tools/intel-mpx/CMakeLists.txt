if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  return ()
endif ()

include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

add_library(lldb-intel-mpxtable SHARED
  IntelMPXTablePlugin.cpp
  )

target_link_libraries(lldb-intel-mpxtable PUBLIC liblldb)

if (LLDB_LINKER_SUPPORTS_GROUPS)
  target_link_libraries(lldb-intel-mpxtable PUBLIC
                        -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
else()
  target_link_libraries(lldb-intel-mpxtable PUBLIC ${LLDB_USED_LIBS})
endif()
llvm_config(lldb-intel-mpxtable ${LLVM_LINK_COMPONENTS})


install(TARGETS lldb-intel-mpxtable
  LIBRARY DESTINATION bin)
