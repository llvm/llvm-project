# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

function(install_symlink name target outdir)
  set(DESTDIR $ENV{DESTDIR})
  if(CMAKE_HOST_UNIX)
    set(LINK_OR_COPY create_symlink)
  else()
    set(LINK_OR_COPY copy)
  endif()

  set(bindir "${DESTDIR}${CMAKE_INSTALL_PREFIX}/${outdir}/")

  message(STATUS "Creating ${name}")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E ${LINK_OR_COPY} "${target}" "${name}"
    WORKING_DIRECTORY "${bindir}")

endfunction()
