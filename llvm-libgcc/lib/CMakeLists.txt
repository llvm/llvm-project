include(CheckLibraryExists)
include(GNUInstallDirs)

string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

add_custom_target(gcc_s_ver ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gcc_s.ver")
set(LLVM_LIBGCC_GCC_S_VER "${CMAKE_CURRENT_BINARY_DIR}/gcc_s.ver")

add_custom_target(gcc_s.ver ALL
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gcc_s.ver"
  COMMAND
    "${CMAKE_C_COMPILER}"
    "-E"
    "-xc" "${CMAKE_CURRENT_SOURCE_DIR}/gcc_s.ver"
    "-o" "${CMAKE_CURRENT_BINARY_DIR}/gcc_s.ver"
)
set_target_properties(gcc_s.ver PROPERTIES
  OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/gcc_s.ver")

add_library(libgcc_s SHARED blank.c)
add_dependencies(libgcc_s gcc_s_ver)
set_target_properties(libgcc_s
  PROPERTIES
    LINKER_LANGUAGE C
    OUTPUT_NAME "unwind"
    VERSION "1.0"
    SOVERSION "1"
    POSITION_INDEPENDENT_CODE ON)
string(REGEX MATCH "[^-]+" LLVM_LIBGCC_TARGET_ARCH ${CMAKE_C_COMPILER_TARGET})
target_link_libraries(libgcc_s PRIVATE
  $<TARGET_OBJECTS:unwind_static>
  $<TARGET_OBJECTS:clang_rt.builtins-${LLVM_LIBGCC_TARGET_ARCH}>
)
target_link_options(libgcc_s PRIVATE
  -nostdlib
  -Wl,--version-script,$<TARGET_PROPERTY:gcc_s.ver,OUTPUT_PATH>)

check_library_exists(m sin "" LLVM_LIBGCC_HAS_LIBM)
target_link_libraries(libgcc_s PRIVATE
  $<$<BOOL:LLVM_LIBGCC_HAS_LIBM>:m>
  c
)

get_filename_component(LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT "${CMAKE_INSTALL_PREFIX}/${LIBUNWIND_INSTALL_LIBRARY_DIR}" ABSOLUTE)
#string(REPLACE "${CMAKE_INSTALL_FULL_LIBDIR}/" "" LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT "${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}")

install(TARGETS libgcc_s
        LIBRARY DESTINATION "${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}" COMPONENT unwind
        ARCHIVE DESTINATION "${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}" COMPONENT unwind
        RUNTIME DESTINATION bin COMPONENT unwind)

get_compiler_rt_install_dir(${LLVM_LIBGCC_TARGET_ARCH} install_dir_builtins)
string(REGEX REPLACE "^lib/" "" install_dir_builtins "${install_dir_builtins}")
string(FIND "${install_dir_builtins}" "clang" install_path_contains_triple)
if(install_path_contains_triple EQUAL -1)
  set(builtins_suffix "-${LLVM_LIBGCC_TARGET_ARCH}")
else()
  string(PREPEND install_dir_builtins "../")
endif()
install(CODE "execute_process(
                COMMAND \"\${CMAKE_COMMAND}\" -E
                create_symlink ${install_dir_builtins}/libclang_rt.builtins${builtins_suffix}.a libgcc.a
                WORKING_DIRECTORY \"\$ENV{DESTDIR}${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}\")"
        COMPONENT unwind)

install(CODE "execute_process(
                COMMAND \"\${CMAKE_COMMAND}\" -E
                create_symlink libunwind.a libgcc_eh.a
                WORKING_DIRECTORY \"\$ENV{DESTDIR}${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}\")"
        COMPONENT unwind)
install(CODE "execute_process(
               COMMAND \"\${CMAKE_COMMAND}\" -E
               create_symlink libunwind.so libgcc_s.so.1.0
               WORKING_DIRECTORY \"\$ENV{DESTDIR}${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}\")"
        COMPONENT unwind)
install(CODE "execute_process(
                COMMAND \"\${CMAKE_COMMAND}\" -E
                create_symlink libgcc_s.so.1.0 libgcc_s.so.1
                WORKING_DIRECTORY \"\$ENV{DESTDIR}${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}\")"
        COMPONENT unwind)
install(CODE "execute_process(
                COMMAND \"\${CMAKE_COMMAND}\" -E
                create_symlink libgcc_s.so.1 libgcc_s.so
                WORKING_DIRECTORY \"\$ENV{DESTDIR}${LLVM_LIBGCC_LIBUNWIND_STATIC_ROOT}\")"
        COMPONENT unwind)
