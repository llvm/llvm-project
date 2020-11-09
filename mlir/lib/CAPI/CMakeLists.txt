add_subdirectory(IR)
add_subdirectory(Registration)
add_subdirectory(Standard)
add_subdirectory(Transforms)


################################################################################
# libMLIRPublicAPI shared library/DLL.
################################################################################

get_property(public_api_libs GLOBAL PROPERTY MLIR_PUBLIC_C_API_LIBS)

foreach(lib ${public_api_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # Accumulate transitive deps of each exported lib into _DEPS.
  list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach()

add_mlir_library(MLIRPublicAPI
  SHARED
  ${_OBJECTS}
  EXCLUDE_FROM_LIBMLIR
  LINK_LIBS
  # Dependency on the implementation shared library.
  $<$<BOOL:${LLVM_BUILD_LLVM_DYLIB}>:MLIR>
  ${_DEPS}
)

target_link_options(
  MLIRPublicAPI
  PRIVATE
    # On Linux, disable re-export of any static linked libraries that
    # came through.
    $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
)
