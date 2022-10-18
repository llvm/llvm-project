##===----------------------------------------------------------------------===##
#
# Build a plugin for a NEC Aurora machine if available. (Can also run on host)
#
##===----------------------------------------------------------------------===##


if(${LIBOMPTARGET_DEP_VEO_FOUND})
  libomptarget_say("Building SX-Aurora VE offloading plugin.")
  set(additional_libs "")
  set(additional_libs ${LIBOMPTARGET_DEP_VEO_LIBRARIES}
                      ${LIBOMPTARGET_DEP_VEOSINFO_LIBRARIES}
                      ${additional_libs})

  set(tmachine_name "ve")
  set(tmachine_libname "ve")
  set(tmachine_triple "ve-unknown-linux-unknown")
  set(elf_machine_id 251)

  # Define macro to be used as prefix of the runtime messages for this target.
  add_definitions("-DTARGET_NAME=${tmachine_name}")

  # Define macro with the ELF ID for this target.
  add_definitions("-DTARGET_ELF_ID=${elf_machine_id}")

  add_llvm_library("omptarget.rtl.${tmachine_libname}" 
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rtl.cpp

    ADDITIONAL_HEADER_DIRS
    ${LIBOMPTARGET_INCLUDE_DIR}
    ${LIBOMPTARGET_DEP_VEO_INCLUDE_DIR}

    LINK_LIBS
    PRIVATE
    elf_common
    ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
    ${additional_libs}
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports -Wl,-z,defs"

    NO_INSTALL_RPATH
  )

  # Install plugin under the lib destination folder.
  install(TARGETS  "omptarget.rtl.${tmachine_libname}" LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
  set_target_properties("omptarget.rtl.${tmachine_libname}" PROPERTIES 
    INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")

  target_include_directories("omptarget.rtl.${tmachine_libname}" PRIVATE
    ${LIBOMPTARGET_INCLUDE_DIR}
    ${LIBOMPTARGET_DEP_VEO_INCLUDE_DIR})

  target_link_libraries(
    "omptarget.rtl.${tmachine_libname}"
    elf_common
    ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
    ${additional_libs}
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports -Wl,-z,defs")

  # Report to the parent scope that we are building a plugin.
  set(LIBOMPTARGET_SYSTEM_TARGETS
    "${LIBOMPTARGET_SYSTEM_TARGETS} ${tmachine_triple}" PARENT_SCOPE)
else()
    libomptarget_say("Not building nec-aurora plugin: libveo or libveosinfo not found.")
endif()
