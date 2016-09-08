find_package(PythonInterp REQUIRED)
configure_file(streamexecutor-config.in streamexecutor-config)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/streamexecutor-config DESTINATION bin)
