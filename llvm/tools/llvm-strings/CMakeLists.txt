set(LLVM_LINK_COMPONENTS
  Core
  Object
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(StringsOptsTableGen)

add_llvm_tool(llvm-strings
  llvm-strings.cpp
  DEPENDS
  StringsOptsTableGen
  )

if(APPLE)
  # We need the Xcode SDK's iconv here, even if other parts of LLVM are
  # configured to pick up a different version, and even if include directories
  # put another <iconv.h> first. Bypass find_package() and look only in the
  # SDK's directory.
  if(CMAKE_OSX_SYSROOT)
    set(Xcode_Iconv_SYSROOT ${CMAKE_OSX_SYSROOT})
  else()
    execute_process(COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE Xcode_Iconv_SYSROOT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  find_file(Xcode_Iconv_HEADER iconv.h PATHS ${Xcode_Iconv_SYSROOT}/usr/include NO_DEFAULT_PATH REQUIRED)
  target_compile_definitions(llvm-strings PRIVATE "ICONVH=\"${Xcode_Iconv_HEADER}\"")
  find_library(Xcode_Iconv_LIBRARY iconv PATHS ${Xcode_Iconv_SYSROOT}/usr/lib NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(llvm-strings PRIVATE "${Xcode_Iconv_LIBRARY}")
endif()

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(strings llvm-strings)
endif()
