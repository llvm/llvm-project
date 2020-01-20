add_custom_target(lldb-repro)
add_dependencies(lldb-repro lldb-test-deps)
set_target_properties(lldb-repro PROPERTIES FOLDER "lldb utils")
configure_file(lldb-repro.py ${LLVM_RUNTIME_OUTPUT_INTDIR}/lldb-repro COPYONLY)
