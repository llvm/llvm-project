# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

# Set to an arbitrary directory to silence GNUInstallDirs warnings
# regarding being unable to determine libdir.
set(CMAKE_INSTALL_LIBDIR "lib")
include(GNUInstallDirs)

function(install_symlink name target outdir)
  set(DESTDIR $ENV{DESTDIR})
  if(NOT IS_ABSOLUTE "${outdir}")
    set(outdir "${CMAKE_INSTALL_PREFIX}/${outdir}")
  endif()
  set(outdir "${DESTDIR}${outdir}")

  message(STATUS "Creating ${name}")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${target}" "${name}"
    WORKING_DIRECTORY "${outdir}" ERROR_VARIABLE has_err)
  if(CMAKE_HOST_WIN32 AND has_err)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E copy "${target}" "${name}"
      WORKING_DIRECTORY "${outdir}")
  endif()

endfunction()
