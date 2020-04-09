# Building libmlir-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  return()
endif()

# Building libmlir-cpp.so may not work on MSVC
if (MSVC)
  return()
endif()

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
list(REMOVE_DUPLICATES mlir_libs)

foreach (lib ${mlir_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  list(APPEND _DEPS $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach ()

if(MLIR_LINK_MLIR_DYLIB)
  set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
endif()

# libMLIR.so depends on LLVM components.  To avoid multiple
# copies of those LLVM components, libMLIR.so depends on libLLVM.so.
# This probably won't work if some LLVM components are not included
# in libLLVM.so.
if(LLVM_BUILD_LLVM_DYLIB)
  add_llvm_library(MLIR
    SHARED
    ${INSTALL_WITH_TOOLCHAIN}

    mlir-shlib.cpp
    )
  target_link_libraries(MLIR PRIVATE LLVM ${LLVM_PTHREAD_LIB})
endif()
