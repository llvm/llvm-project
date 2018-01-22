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
  if(MSVC)
    set(SYSTEM_LIBS ${SYSTEM_LIBS} "${l}.lib")
  else()
    if (l MATCHES "^-")
      # If it's an option, pass it without changes.
      set(SYSTEM_LIBS ${SYSTEM_LIBS} "${l}")
    else()
      # Otherwise assume it's a library name we need to link with.
      set(SYSTEM_LIBS ${SYSTEM_LIBS} "-l${l}")
    endif()
  endif()
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
set(LLVM_DYLIB_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}")
set(LLVM_HAS_GLOBAL_ISEL "ON")

# Use the C++ link flags, since they should be a superset of C link flags.
set(LLVM_LDFLAGS "${CMAKE_CXX_LINK_FLAGS}")
set(LLVM_BUILDMODE ${CMAKE_BUILD_TYPE})
set(LLVM_SYSTEM_LIBS ${SYSTEM_LIBS})
string(REPLACE ";" " " LLVM_TARGETS_BUILT "${LLVM_TARGETS_TO_BUILD}")
llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_LLVM_DYLIB
  LLVM_LINK_LLVM_DYLIB
  LLVM_HAS_RTTI
  LLVM_HAS_GLOBAL_ISEL
  BUILD_SHARED_LIBS)
configure_file(${BUILDVARIABLES_SRCPATH} ${BUILDVARIABLES_OBJPATH} @ONLY)

# Set build-time environment(s).
add_definitions(-DCMAKE_CFG_INTDIR="${CMAKE_CFG_INTDIR}")

if(LLVM_ENABLE_MODULES)
  target_compile_options(llvm-config PUBLIC
    "-fmodules-ignore-macro=CMAKE_CFG_INTDIR"
    )
endif()

# Add the dependency on the generation step.
add_file_dependencies(${CMAKE_CURRENT_SOURCE_DIR}/llvm-config.cpp ${BUILDVARIABLES_OBJPATH})

if(CMAKE_CROSSCOMPILING AND NOT LLVM_CONFIG_PATH)
  set(LLVM_CONFIG_PATH "${LLVM_NATIVE_BUILD}/bin/llvm-config" CACHE STRING "")

  add_custom_command(OUTPUT "${LLVM_CONFIG_PATH}"
    COMMAND ${CMAKE_COMMAND} --build . --target llvm-config --config $<CONFIGURATION>
    DEPENDS ${LLVM_NATIVE_BUILD}/CMakeCache.txt
    WORKING_DIRECTORY ${LLVM_NATIVE_BUILD}
    COMMENT "Building native llvm-config..."
    USES_TERMINAL)
  add_custom_target(NativeLLVMConfig DEPENDS ${LLVM_CONFIG_PATH})
  add_dependencies(NativeLLVMConfig CONFIGURE_LLVM_NATIVE)

  add_dependencies(llvm-config NativeLLVMConfig)
endif()
