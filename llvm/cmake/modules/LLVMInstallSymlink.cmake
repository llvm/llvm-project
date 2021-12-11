# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

function(install_symlink name target outdir)
  set(DESTDIR $ENV{DESTDIR})
  set(bindir "${DESTDIR}${CMAKE_INSTALL_PREFIX}/${outdir}/")

  message(STATUS "Creating ${name}")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${target}" "${name}"
    WORKING_DIRECTORY "${bindir}" ERROR_VARIABLE has_err)
  if(CMAKE_HOST_WIN32 AND has_err)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E copy "${target}" "${name}"
      WORKING_DIRECTORY "${bindir}")
  endif()

endfunction()
