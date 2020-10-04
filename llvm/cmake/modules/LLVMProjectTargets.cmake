# For project foo, this function generates:
# add_foo_tool(name)     (An executable installed by default)
# add_foo_utility(name)  (An executable *not* installed by default)
# add_foo_example(name)  (An executable which is built, but never installed)
# add_foo_example_library(name)  (A library to go along with an example)

# It also assumes the following configuration environment variables
# (see LLVMProjectOptions.cmake)
# FOO_TOOLS_INSTALL_DIR
# FOO_BUILD_TOOLS
# FOO_BUILD_UTILS
# FOO_INSTALL_UTILS
# FOO_BUILD_EXAMPLES
# FOO_HAS_EXPORTS
# FOO_INSTALL_TOOLCHAIN_ONLY

function(add_llvm_project_targets projectname)
  string(TOUPPER "${name}" upperprojectname)

  macro(add_${projectname}_tool name)
    if( NOT ${upperprojectname}_BUILD_TOOLS )
      set(EXCLUDE_FROM_ALL ON)
    endif()
    add_llvm_executable(${name} ${ARGN})

    if ( ${name} IN_LIST LLVM_TOOLCHAIN_TOOLS OR NOT ${upperprojectname}_INSTALL_TOOLCHAIN_ONLY)
      if( ${upperprojectname}_BUILD_TOOLS )
        set(export_to_${projectname}exports)
        if(${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
            NOT LLVM_DISTRIBUTION_COMPONENTS)
          set(export_to_${projectname}exports EXPORT ${upperprojectname}Exports)
          set_property(GLOBAL PROPERTY ${upperprojectname}_HAS_EXPORTS True)
        endif()

        install(TARGETS ${name}
          ${export_to_${projectname}exports}
          RUNTIME DESTINATION ${${upperprojectname}_TOOLS_INSTALL_DIR}
          COMPONENT ${name})

        if (NOT LLVM_ENABLE_IDE)
          add_llvm_install_targets(install-${name}
            DEPENDS ${name}
            COMPONENT ${name})
        endif()
      endif()
    endif()
    if( ${upperprojectname}_BUILD_TOOLS )
      set_property(GLOBAL APPEND PROPERTY ${upperprojectname}_EXPORTS ${name})
    endif()
    set_target_properties(${name} PROPERTIES FOLDER "Tools")
  endmacro(add_${projectname}_tool name)

  macro(add_${projectname}_example name)
    if( NOT ${upperprojectname}_BUILD_EXAMPLES )
      set(EXCLUDE_FROM_ALL ON)
    endif()
    add_llvm_executable(${name} ${ARGN})
    if( ${upperprojectname}_BUILD_EXAMPLES )
      install(TARGETS ${name} RUNTIME DESTINATION examples)
    endif()
    set_target_properties(${name} PROPERTIES FOLDER "Examples")
  endmacro(add_${projectname}_example name)

  macro(add_${projectname}_example_library name)
    if( NOT ${upperprojectname}_BUILD_EXAMPLES )
      set(EXCLUDE_FROM_ALL ON)
      add_llvm_library(${name} BUILDTREE_ONLY ${ARGN})
    else()
      add_llvm_library(${name} ${ARGN})
    endif()

    set_target_properties(${name} PROPERTIES FOLDER "Examples")
  endmacro(add_${projectname}_example_library name)

  # This is a macro that is used to create targets for executables that are needed
  # for development, but that are not intended to be installed by default.
  macro(add_${projectname}_utility name)
    if ( NOT ${upperprojectname}_BUILD_UTILS )
      set(EXCLUDE_FROM_ALL ON)
    endif()

    add_llvm_executable(${name} DISABLE_LLVM_LINK_LLVM_DYLIB ${ARGN})
    set_target_properties(${name} PROPERTIES FOLDER "Utils")
    if (NOT ${upperprojectname}_INSTALL_TOOLCHAIN_ONLY)
      if (${upperprojectname}_INSTALL_UTILS AND ${upperprojectname}_BUILD_UTILS)
        set(export_to_${projectname}exports)
        if (${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
            NOT LLVM_DISTRIBUTION_COMPONENTS)
          set(export_to_${projectname}exports EXPORT ${upperprojectname}Exports)
          set_property(GLOBAL PROPERTY ${upperprojectname}_HAS_EXPORTS True)
        endif()

        install(TARGETS ${name}
          ${export_to_${projectname}exports}
          RUNTIME DESTINATION ${LLVM_UTILS_INSTALL_DIR}
          COMPONENT ${name})

        if (NOT LLVM_ENABLE_IDE)
          add_llvm_install_targets(install-${name}
            DEPENDS ${name}
            COMPONENT ${name})
        endif()
        set_property(GLOBAL APPEND PROPERTY ${upperprojectname}_EXPORTS ${name})
      elseif(${upperprojectname}_BUILD_UTILS)
        set_property(GLOBAL APPEND PROPERTY ${upperprojectname}_EXPORTS_BUILDTREE_ONLY ${name})
      endif()
    endif()
  endmacro(add_${projectname}_utility name)
endfunction(add_llvm_project_targets)
