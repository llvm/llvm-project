macro(add_se_library name)
  add_llvm_library(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "streamexecutor libraries")
endmacro(add_se_library)

add_subdirectory(Utils)

add_se_library(
    streamexecutor
    Device.cpp
    DeviceMemory.cpp
    Kernel.cpp
    KernelSpec.cpp
    PackedKernelArgumentArray.cpp
    Platform.cpp
    PlatformDevice.cpp
    PlatformManager.cpp
    Stream.cpp

    LINK_LIBS
      utils
    )

install(TARGETS streamexecutor DESTINATION lib)
