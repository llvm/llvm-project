set(LLVM_LINK_COMPONENTS support)

# We need to generate the BuildVariables.inc file containing values which are
# only defined when under certain build modes. Unfortunately, that precludes
# doing this inside CMake so we have to shell out to sed. For now, that means we
# can't expect to build llvm-config on Window.s
set(BUILDVARIABLES_SRCPATH ${CMAKE_CURRENT_SOURCE_DIR}/BuildVariables.inc.in)
set(BUILDVARIABLES_OBJPATH ${CMAKE_CURRENT_BINARY_DIR}/BuildVariables.inc)
set(SEDSCRIPT_OBJPATH ${CMAKE_CURRENT_BINARY_DIR}/BuildVariables.configure.sed)

# Compute the substitution values for various items.
get_system_libs(LLVM_SYSTEM_LIBS_LIST)
foreach(l ${LLVM_SYSTEM_LIBS_LIST})
  set(SYSTEM_LIBS ${SYSTEM_LIBS} "-l${l}")
endforeach()
set(C_FLGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(CXX_FLGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(CPP_FLGS "${CMAKE_CPP_FLAGS} ${CMAKE_CPP_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")

add_custom_command(OUTPUT ${BUILDVARIABLES_OBJPATH}
  COMMAND echo s!@LLVM_SRC_ROOT@!${LLVM_MAIN_SRC_DIR}! > ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_OBJ_ROOT@!${LLVM_BINARY_DIR}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_CPPFLAGS@!${CPP_FLGS}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_CFLAGS@!${C_FLGS}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_CXXFLAGS@!${CXX_FLGS}! >> ${SEDSCRIPT_OBJPATH}
  # TODO: Use general flags for linking! not just for shared libs:
  COMMAND echo s!@LLVM_LDFLAGS@!${CMAKE_SHARED_LINKER_FLAGS}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_BUILDMODE@!${CMAKE_BUILD_TYPE}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_SYSTEM_LIBS@!${SYSTEM_LIBS}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND echo s!@LLVM_TARGETS_BUILT@!${LLVM_TARGETS_TO_BUILD}! >> ${SEDSCRIPT_OBJPATH}
  COMMAND sed -f ${SEDSCRIPT_OBJPATH} < ${BUILDVARIABLES_SRCPATH} > ${BUILDVARIABLES_OBJPATH}
  VERBATIM
  COMMENT "Building BuildVariables.inc include."
  )

# Add the llvm-config tool.
add_llvm_tool(llvm-config
  llvm-config.cpp
  )

# Add the dependency on the generation step.
add_file_dependencies(${CMAKE_CURRENT_SOURCE_DIR}/llvm-config.cpp ${BUILDVARIABLES_OBJPATH})
