add_lldb_executable(lit-cpuid
  lit-cpuid.cpp
  )

target_link_libraries(lit-cpuid PRIVATE LLVMSupport)
set_target_properties(lit-cpuid PROPERTIES FOLDER "lldb utils")
