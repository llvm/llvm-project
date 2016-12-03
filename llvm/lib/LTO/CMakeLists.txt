# Figure out if we can track VC revisions.
function(find_first_existing_file out_var)
  foreach(file ${ARGN})
    if(EXISTS "${file}")
      set(${out_var} "${file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

macro(find_first_existing_vc_file out_var path)
  find_first_existing_file(${out_var}
    "${path}/.git/logs/HEAD" # Git
    "${path}/.svn/wc.db"     # SVN 1.7
    "${path}/.svn/entries"   # SVN 1.6
    )
endmacro()

find_first_existing_vc_file(llvm_vc "${LLVM_MAIN_SRC_DIR}")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/LLVMLTORevision.h")

set(get_svn_script "${LLVM_MAIN_SRC_DIR}/cmake/modules/GenerateVersionFromCVS.cmake")

if(DEFINED llvm_vc)
  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${version_inc}"
    DEPENDS "${llvm_vc}" "${get_svn_script}"
    COMMAND
    ${CMAKE_COMMAND} "-DSOURCE_DIR=${LLVM_MAIN_SRC_DIR}"
                     "-DNAME=LLVM_REVISION"
                     "-DHEADER_FILE=${version_inc}"
                     -P "${get_svn_script}")

  # Mark the generated header as being generated.
  set_source_files_properties("${version_inc}"
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")
else()
  # Not producing a VC revision include.
  set(version_inc)
endif()


add_llvm_library(LLVMLTO
  Caching.cpp
  LTO.cpp
  LTOBackend.cpp
  LTOModule.cpp
  LTOCodeGenerator.cpp
  UpdateCompilerUsed.cpp
  ThinLTOCodeGenerator.cpp
  ${version_inc}

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LTO
)

add_dependencies(LLVMLTO intrinsics_gen)
