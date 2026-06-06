# Building libmlir-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  return()
endif()

# Building libmlir-cpp.so may not work on MSVC
if (MSVC)
  return()
endif()

get_property(mlir_libs GLOBAL PROPERTY MLIR_STATIC_LIBS)
get_property(mlir_llvm_link_components GLOBAL PROPERTY MLIR_LLVM_LINK_COMPONENTS)
list(REMOVE_DUPLICATES mlir_libs)
list(REMOVE_DUPLICATES mlir_llvm_link_components)

foreach (lib ${mlir_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # libClang needs this, but it causes problems for MLIR (probably
  # because we use public library dependencies within MLIR.)
  # list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach ()

if(MLIR_LINK_MLIR_DYLIB)
  set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
endif()

if(LLVM_BUILD_LLVM_DYLIB)
  add_mlir_library(
    MLIR
    SHARED
    EXCLUDE_FROM_LIBMLIR
    ${INSTALL_WITH_TOOLCHAIN}
    mlir-shlib.cpp
    ${_OBJECTS}
    LINK_LIBS
    ${_DEPS}

    LINK_COMPONENTS
    ${mlir_llvm_link_components}
  )
  target_link_libraries(MLIR PRIVATE ${LLVM_PTHREAD_LIB})
endif()

#message("Libraries included in libMLIR.so: ${mlir_libs}")
#message("LLVM Components included in libMLIR.so: ${mlir_llvm_link_components}")

mlir_check_all_link_libraries(MLIR)
