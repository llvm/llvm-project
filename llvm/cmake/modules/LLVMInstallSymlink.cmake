# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.
#
# `outdir` must be an absolute path. This module gets a very reduced
# `CMAKE_MODULE_PATH` so it is easier to make the caller the responsible
# for this.

include(GNUInstallDirs)

function(install_symlink name target outdir)
  set(DESTDIR $ENV{DESTDIR})
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
