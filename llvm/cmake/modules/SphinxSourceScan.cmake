# Enumerate a documentation source tree for SphinxSourceSync.cmake.
#
# Usage:
#
#   cmake -DSOURCE_DIR=/path/to/project/docs \
#         -DDESTINATION_DIR=/path/to/build/project/docs \
#         -DFILE_LIST=/path/to/build/file-list.txt \
#         -DDESTINATION_DOC_LIST=/path/to/build/destination-docs.txt \
#         -DMISSING_FILE_LIST=/path/to/build/missing-files.txt \
#         -DIGNORE_MISSING_FILE=/path/to/build/ignore-missing.txt \
#         -P /path/to/SphinxSourceScan.cmake
#
# The output file contains source-relative paths, one per line. It is written
# only when the directory listing changes, allowing downstream custom commands
# to skip work when the always-run scan observes no additions, removals, or
# renames. DESTINATION_DOC_LIST similarly records destination-relative .rst and
# .md files so destination-only stale docs can trigger cleanup.
# MISSING_FILE_LIST records source files that do not currently exist in the
# destination tree, excluding paths listed in IGNORE_MISSING_FILE.

include("${CMAKE_CURRENT_LIST_DIR}/SphinxSourceUtils.cmake")

if (NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR must be set")
endif()

if (NOT DEFINED FILE_LIST)
  message(FATAL_ERROR "FILE_LIST must be set")
endif()

if (NOT DEFINED DESTINATION_DIR)
  message(FATAL_ERROR "DESTINATION_DIR must be set")
endif()

if (NOT DEFINED DESTINATION_DOC_LIST)
  message(FATAL_ERROR "DESTINATION_DOC_LIST must be set")
endif()

if (NOT DEFINED MISSING_FILE_LIST)
  message(FATAL_ERROR "MISSING_FILE_LIST must be set")
endif()

if (NOT DEFINED IGNORE_MISSING_FILE)
  message(FATAL_ERROR "IGNORE_MISSING_FILE must be set")
endif()

file(GLOB_RECURSE source_files
  LIST_DIRECTORIES false
  RELATIVE "${SOURCE_DIR}"
  "${SOURCE_DIR}/*")
list(SORT source_files)

set(contents)
foreach(relative_path IN LISTS source_files)
  string(APPEND contents "${relative_path}\n")
endforeach()
write_if_changed("${FILE_LIST}" "${contents}")

file(STRINGS "${IGNORE_MISSING_FILE}" ignore_missing_files)
set(contents)
foreach(relative_path IN LISTS source_files)
  list(FIND ignore_missing_files "${relative_path}" ignore_index)
  if (ignore_index EQUAL -1 AND
      NOT EXISTS "${DESTINATION_DIR}/${relative_path}")
    string(APPEND contents "${relative_path}\n")
  endif()
endforeach()
write_if_changed("${MISSING_FILE_LIST}" "${contents}")

if (EXISTS "${DESTINATION_DIR}")
  file(GLOB_RECURSE destination_docs
    LIST_DIRECTORIES false
    RELATIVE "${DESTINATION_DIR}"
    "${DESTINATION_DIR}/*.rst"
    "${DESTINATION_DIR}/*.md")
  list(SORT destination_docs)
else()
  set(destination_docs)
endif()

set(contents)
foreach(relative_path IN LISTS destination_docs)
  string(APPEND contents "${relative_path}\n")
endforeach()
write_if_changed("${DESTINATION_DOC_LIST}" "${contents}")
