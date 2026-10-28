# Returns the host triple.
# Invokes config.guess, except for platforms it cannot identify.

function( get_host_triple var )
  if( MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    if( CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "ARM64.*" )
      set( value "aarch64-pc-windows-msvc" )
    elseif( CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "ARM.*" )
      set( value "armv7-pc-windows-msvc" )
    elseif( CMAKE_C_COMPILER_ARCHITECTURE_ID STREQUAL "x64" )
      set( value "x86_64-pc-windows-msvc" )
    elseif( CMAKE_C_COMPILER_ARCHITECTURE_ID STREQUAL "X86" )
      set( value "i686-pc-windows-msvc" )
    elseif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      set( value "x86_64-pc-windows-msvc" )
    else()
      set( value "i686-pc-windows-msvc" )
    endif()
  elseif( MINGW AND NOT MSYS )
    # CMake doesn't provide COMPILER_ARCHITECTURE_ID for Clang/GCC,
    # but it does for MSVC.
    if( CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ARM.*" )
      if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set( value "aarch64-w64-windows-gnu" )
      else()
        set( value "armv7-w64-windows-gnu" )
      endif()
    else()
      if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set( value "x86_64-w64-windows-gnu" )
      else()
        set( value "i686-w64-windows-gnu" )
      endif()
    endif()
  elseif( CMAKE_SYSTEM_NAME MATCHES "OS390" )
    set( value "s390x-ibm-zos" )
  elseif( CMAKE_SYSTEM_NAME STREQUAL "AIX" )
    # We defer to dynamic detection of the host AIX version.
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      set( value "powerpc64-ibm-aix" )
    else()
      set( value "powerpc-ibm-aix" )
    endif()
  elseif( CMAKE_HOST_SYSTEM_NAME MATCHES "HarmonyOS|OpenHarmony|OHOS" )
    # config.guess cannot identify HarmonyOS: uname -s reports "HarmonyOS",
    # which config.guess does not recognize, and its probing is rejected by
    # the system sandbox. Build the triple from CMake's host detection using
    # the form clang targets on OHOS (e.g. aarch64-unknown-linux-ohos). CMake
    # itself may report the host processor as "unknown" because uname -p does,
    # so fall back to asking uname -m directly.
    set( TT_MACHINE "${CMAKE_HOST_SYSTEM_PROCESSOR}" )
    if( NOT TT_MACHINE OR TT_MACHINE STREQUAL "unknown" )
      find_program( TT_UNAME uname PATHS /bin /usr/bin /usr/local/bin
        NO_CMAKE_FIND_ROOT_PATH )
      set( TT_RV 1 )
      if( TT_UNAME )
        execute_process( COMMAND ${TT_UNAME} -m
          RESULT_VARIABLE TT_RV
          OUTPUT_VARIABLE TT_MACHINE
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET )
      endif()
      if( NOT TT_RV EQUAL 0 )
        set( TT_MACHINE "" )
      endif()
    endif()
    if( TT_MACHINE MATCHES "aarch64|arm64" )
      set( value "aarch64-unknown-linux-ohos" )
    elseif( TT_MACHINE MATCHES "^arm" )
      set( value "arm-unknown-linux-ohos" )
    elseif( TT_MACHINE MATCHES "^x86_64" )
      set( value "x86_64-unknown-linux-ohos" )
    elseif( TT_MACHINE MATCHES "^riscv64" )
      set( value "riscv64-unknown-linux-ohos" )
    elseif( TT_MACHINE MATCHES "^loongarch64" )
      set( value "loongarch64-unknown-linux-ohos" )
    else()
      message( FATAL_ERROR
        "Failed to determine host triple for ${CMAKE_HOST_SYSTEM_NAME}: "
        "host processor \"${TT_MACHINE}\" is not supported" )
    endif()
  else()
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows AND NOT MSYS)
      message(WARNING "unable to determine host target triple")
    else()
      set(config_guess ${LLVM_MAIN_SRC_DIR}/cmake/config.guess)
      execute_process(COMMAND sh ${config_guess}
        RESULT_VARIABLE TT_RV
        OUTPUT_VARIABLE TT_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if( NOT TT_RV EQUAL 0 )
        message(FATAL_ERROR "Failed to execute ${config_guess}")
      endif( NOT TT_RV EQUAL 0 )
      set( value ${TT_OUT} )
    endif()
  endif()
  set( ${var} ${value} PARENT_SCOPE )
endfunction( get_host_triple var )
