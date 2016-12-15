include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

add_lldb_tool(lldb-argdumper INCLUDE_IN_FRAMEWORK
  argdumper.cpp
  )

if (LLDB_LINKER_SUPPORTS_GROUPS)
  target_link_libraries(lldb-argdumper -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
else()
  target_link_libraries(lldb-argdumper ${LLDB_USED_LIBS})
endif()
llvm_config(lldb-argdumper ${LLVM_LINK_COMPONENTS})

