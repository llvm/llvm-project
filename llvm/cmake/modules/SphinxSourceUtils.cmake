# Common helpers for Sphinx source scan and sync scripts.

# CMake-script native implementation of cmake -E write_if_changed. This keeps
# frequently run scan/sync scripts from paying for an extra subprocess just to
# avoid rewriting unchanged files.
function(write_if_changed output_file contents)
  if (EXISTS "${output_file}")
    file(READ "${output_file}" old_contents)
  else()
    set(old_contents)
  endif()

  if (NOT contents STREQUAL old_contents)
    get_filename_component(output_dir "${output_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_dir}")
    file(WRITE "${output_file}" "${contents}")
  endif()
endfunction()
