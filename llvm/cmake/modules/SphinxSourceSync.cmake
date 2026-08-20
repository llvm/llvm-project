# Synchronize a documentation source tree into an intermediate Sphinx source
# tree.
#
# Usage:
#
#   cmake -DSOURCE_DIR=/path/to/project/docs \
#         -DDESTINATION_DIR=/path/to/build/project/docs \
#         -DFILE_LIST=/path/to/build/file-list.txt \
#         -DPRESERVE_FILE=/path/to/build/preserve.txt \
#         -DMANIFEST_FILE=/path/to/build/manifest.txt \
#         -DDEPFILE=/path/to/build/sync.d \
#         -DSTAMP_FILE=/path/to/build/sync.stamp \
#         -P /path/to/SphinxSourceSync.cmake
#
# The sync copies all source files into DESTINATION_DIR, avoids rewriting files
# whose contents are unchanged, and removes stale .rst and .md files that no
# longer exist under SOURCE_DIR unless they are listed in PRESERVE_FILE. This
# is intended for Sphinx builds that merge checked-in docs with generated docs
# in a build-tree source directory.

include("${CMAKE_CURRENT_LIST_DIR}/SphinxSourceUtils.cmake")

if (NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR must be set")
endif()

if (NOT DEFINED DESTINATION_DIR)
  message(FATAL_ERROR "DESTINATION_DIR must be set")
endif()

if (NOT DEFINED FILE_LIST)
  message(FATAL_ERROR "FILE_LIST must be set")
endif()

if (NOT DEFINED PRESERVE_FILE)
  message(FATAL_ERROR "PRESERVE_FILE must be set")
endif()

if (NOT DEFINED MANIFEST_FILE)
  message(FATAL_ERROR "MANIFEST_FILE must be set")
endif()

if (NOT DEFINED DEPFILE)
  message(FATAL_ERROR "DEPFILE must be set")
endif()

if (NOT DEFINED STAMP_FILE)
  message(FATAL_ERROR "STAMP_FILE must be set")
endif()

file(MAKE_DIRECTORY "${DESTINATION_DIR}")
file(STRINGS "${FILE_LIST}" source_files)
file(STRINGS "${PRESERVE_FILE}" preserve_docs)

set(depfile_dependencies)
set(source_docs)
foreach(relative_path IN LISTS source_files)
  set(source_path "${SOURCE_DIR}/${relative_path}")
  set(destination_path "${DESTINATION_DIR}/${relative_path}")
  get_filename_component(destination_parent "${destination_path}" DIRECTORY)
  file(MAKE_DIRECTORY "${destination_parent}")
  # configure_file(COPYONLY) preserves mtimes for unchanged files and avoids
  # the subprocess overhead of cmake -E copy_if_different.
  configure_file("${source_path}" "${destination_path}" COPYONLY)
  list(APPEND depfile_dependencies "${source_path}")

  string(TOLOWER "${relative_path}" lower_relative_path)
  if (lower_relative_path MATCHES "\\.(rst|md)$")
    list(APPEND source_docs "${relative_path}")
  endif()
endforeach()

file(GLOB_RECURSE destination_docs
  LIST_DIRECTORIES false
  RELATIVE "${DESTINATION_DIR}"
  "${DESTINATION_DIR}/*.rst"
  "${DESTINATION_DIR}/*.md")
foreach(relative_path IN LISTS destination_docs)
  list(FIND source_docs "${relative_path}" source_index)
  list(FIND preserve_docs "${relative_path}" preserve_index)
  # Generated docs are absent from SOURCE_DIR, but callers list them in
  # PRESERVE_FILE so stale-source cleanup does not remove build outputs.
  if (source_index EQUAL -1 AND preserve_index EQUAL -1)
    file(REMOVE "${DESTINATION_DIR}/${relative_path}")
  endif()
endforeach()

set(manifest_contents)
foreach(relative_path IN LISTS source_docs)
  string(APPEND manifest_contents "${relative_path}\n")
endforeach()
write_if_changed("${MANIFEST_FILE}" "${manifest_contents}")

escape_depfile_path("${STAMP_FILE}" escaped_stamp)
set(depfile_contents "${escaped_stamp}:")
foreach(source_path IN LISTS depfile_dependencies)
  escape_depfile_path("${source_path}" escaped_source_path)
  string(APPEND depfile_contents " ${escaped_source_path}")
endforeach()
string(APPEND depfile_contents "\n")
file(WRITE "${DEPFILE}" "${depfile_contents}")

get_filename_component(stamp_dir "${STAMP_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${stamp_dir}")
file(TOUCH "${STAMP_FILE}")
