set(LLVM_TARGET_DEFINITIONS Intrinsics.td)

tablegen(LLVM Intrinsics.gen -gen-intrinsic)

add_custom_target(intrinsics_gen ALL
  DEPENDS ${llvm_builded_incs_dir}/Intrinsics.gen)
set_target_properties(intrinsics_gen PROPERTIES FOLDER "Tablegenning")

if( MSVC_IDE OR XCODE )
  # Creates a dummy target containing all headers for the benefit of
  # Visual Studio users.
  file(GLOB_RECURSE headers *.h)
  add_td_sources(headers)
  add_library(llvm_headers_do_not_build EXCLUDE_FROM_ALL
    # We need at least one source file:
    ${LLVM_MAIN_SRC_DIR}/lib/Transforms/Hello/Hello.cpp
    ${headers})
  set_target_properties(llvm_headers_do_not_build PROPERTIES FOLDER "Misc")
endif()
