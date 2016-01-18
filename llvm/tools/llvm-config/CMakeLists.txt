set(LLVM_LINK_COMPONENTS support)

set(BUILDVARIABLES_SRCPATH ${CMAKE_CURRENT_SOURCE_DIR}/BuildVariables.inc.in)
set(BUILDVARIABLES_OBJPATH ${CMAKE_CURRENT_BINARY_DIR}/BuildVariables.inc)

# Add the llvm-config tool.
add_llvm_tool(llvm-config
  llvm-config.cpp
  )

# Compute the substitution values for various items.
get_property(LLVM_SYSTEM_LIBS_LIST TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS)
foreach(l ${LLVM_SYSTEM_LIBS_LIST})
  set(SYSTEM_LIBS ${SYSTEM_LIBS} "-l${l}")
endforeach()
string(REPLACE ";" " " SYSTEM_LIBS "${SYSTEM_LIBS}")

# Fetch target specific compile options, e.g. RTTI option
get_property(COMPILE_FLAGS TARGET llvm-config PROPERTY COMPILE_FLAGS)

# Use configure_file to create BuildVariables.inc.
set(LLVM_SRC_ROOT ${LLVM_MAIN_SRC_DIR})
set(LLVM_OBJ_ROOT ${LLVM_BINARY_DIR})
set(LLVM_CPPFLAGS "${CMAKE_CPP_FLAGS} ${CMAKE_CPP_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(LLVM_CFLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(LLVM_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${COMPILE_FLAGS} ${LLVM_DEFINITIONS}")
set(LLVM_BUILD_SYSTEM cmake)
set(LLVM_HAS_RTTI ${LLVM_CONFIG_HAS_RTTI})

# Use the C++ link flags, since they should be a superset of C link flags.
set(LLVM_LDFLAGS "${CMAKE_CXX_LINK_FLAGS}")
set(LLVM_BUILDMODE ${CMAKE_BUILD_TYPE})
set(LLVM_SYSTEM_LIBS ${SYSTEM_LIBS})
string(REPLACE ";" " " LLVM_TARGETS_BUILT "${LLVM_TARGETS_TO_BUILD}")
configure_file(${BUILDVARIABLES_SRCPATH} ${BUILDVARIABLES_OBJPATH} @ONLY)

# Set build-time environment(s).
add_definitions(-DCMAKE_CFG_INTDIR="${CMAKE_CFG_INTDIR}")

# Add the dependency on the generation step.
add_file_dependencies(${CMAKE_CURRENT_SOURCE_DIR}/llvm-config.cpp ${BUILDVARIABLES_OBJPATH})

if(CMAKE_CROSSCOMPILING)
  set(${project}_LLVM_CONFIG_EXE "${LLVM_NATIVE_BUILD}/bin/llvm-config")
  set(${project}_LLVM_CONFIG_EXE ${${project}_LLVM_CONFIG_EXE} PARENT_SCOPE)

  add_custom_command(OUTPUT "${${project}_LLVM_CONFIG_EXE}"
    COMMAND ${CMAKE_COMMAND} --build . --target llvm-config --config $<CONFIGURATION>
    DEPENDS ${LLVM_NATIVE_BUILD}/CMakeCache.txt
    WORKING_DIRECTORY ${LLVM_NATIVE_BUILD}
    COMMENT "Building native llvm-config...")
  add_custom_target(${project}NativeLLVMConfig DEPENDS ${${project}_LLVM_CONFIG_EXE})
  add_dependencies(${project}NativeLLVMConfig CONFIGURE_LLVM_NATIVE)

  add_dependencies(llvm-config ${project}NativeLLVMConfig)
endif(CMAKE_CROSSCOMPILING)
