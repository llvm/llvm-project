# LLVM-style projects generally have the same directory structure.  This file
# provides some bolierplate cmake support for projects that supports this
# directory structure.  Note that generally speaking, projects should prefer
# to use their own rules for these rather than relying on the core llvm build
# targets.

# Generally name should be lower case.
function(add_llvm_project_options name)
  string(TOUPPER "${name}" uppername)

  # Define options to control the inclusion and default build behavior for
  # components which may not strictly be necessary (tools, examples, and tests).
  #
  # This is primarily to support building smaller or faster project files.
  option(${uppername}_INCLUDE_TOOLS
    "Generate build targets for the ${uppername} tools."
    ${LLVM_INCLUDE_TOOLS})
  option(${uppername}_BUILD_TOOLS
    "Build the ${uppername} tools. If OFF, just generate build targets."
    ${LLVM_BUILD_TOOLS})

  option(${uppername}_INCLUDE_UTILS
    "Generate build targets for the ${uppername} utils."
    ${LLVM_INCLUDE_UTILS})
  option(${uppername}_BUILD_UTILS
    "Build ${uppername} utility binaries. If OFF, just generate build targets."
    ${LLVM_BUILD_UTILS})
  option(${uppername}_INSTALL_UTILS
    "Include utility binaries in the 'install' target."
    ${LLVM_INSTALL_UTILS})

  # i.e. Don't install headers, for instance.
  option(${uppername}_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target."
    ${LLVM_INSTALL_TOOLCHAIN_ONLY})

  option(${uppername}_BUILD_EXAMPLES
    "Build the ${uppername} example programs. If OFF, just generate build targets."
    ${LLVM_BUILD_EXAMPLES})
  option(${uppername}_INCLUDE_EXAMPLES
    "Generate build targets for the ${uppername} examples"
    ${LLVM_INCLUDE_EXAMPLES})
  if(${uppername}_BUILD_EXAMPLES)
    add_definitions(-DBUILD_EXAMPLES)
  endif(${uppername}_BUILD_EXAMPLES)

  option(${uppername}_BUILD_TESTS
    "Build ${uppername} unit tests. If OFF, just generate build targets."
    ${LLVM_BUILD_TESTS})
  option(${uppername}_INCLUDE_TESTS
    "Generate build targets for the ${uppername} unit tests."
    ${LLVM_INCLUDE_TESTS})
  if (${uppername}_INCLUDE_TESTS)
    add_definitions(-D${uppername}_INCLUDE_TESTS)
  endif()

  option(${uppername}_INCLUDE_INTEGRATION_TESTS
    "Generate build targets for the ${uppername} integration tests."
    ${LLVM_INCLUDE_INTEGRATION_TESTS})
  if (${uppername}_INCLUDE_INTEGRATION_TESTS)
    add_definitions(-D${uppername}_INCLUDE_INTEGRATION_TESTS)
  endif()

  option(${uppername}_INCLUDE_DOCS
    "Generate build targets for the ${uppername} docs."
    ${LLVM_INCLUDE_DOCS})

endfunction(add_llvm_project_options)
