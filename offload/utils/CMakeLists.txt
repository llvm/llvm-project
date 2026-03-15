set(OPENMP_UTILS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")

macro(add_openmp_util path)
  install(PROGRAMS
     ${path}
     DESTINATION "${OPENMP_UTILS_INSTALL_DIR}")
endmacro()

add_subdirectory(gpurun)
