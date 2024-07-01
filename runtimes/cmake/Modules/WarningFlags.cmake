include(HandleFlags)

# Warning flags ===============================================================
function(cxx_add_warning_flags target enable_werror enable_pedantic)
  target_compile_definitions(${target} PUBLIC -D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
  if (MSVC)
    # -W4 is the cl.exe/clang-cl equivalent of -Wall. (In cl.exe and clang-cl,
    # -Wall is equivalent to -Weverything in GCC style compiler drivers.)
    target_add_compile_flags_if_supported(${target} PRIVATE -W4)
  else()
    target_add_compile_flags_if_supported(${target} PRIVATE -Wall)
  endif()
  # TODO: Should -Wconversion be enabled?
  target_add_compile_flags_if_supported(${target} PRIVATE
      -Wextra
      -Wnewline-eof
      -Wshadow
      -Wwrite-strings
      -Wno-unused-parameter
      -Wno-long-long
      -Werror=return-type
      -Wextra-semi
      -Wundef
      -Wunused-template
      -Wformat-nonliteral
      )

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    target_add_compile_flags_if_supported(${target} PRIVATE
      -Wno-user-defined-literals
      -Wno-covered-switch-default
      -Wno-suggest-override
    )
    if (LIBCXX_TARGETING_CLANG_CL)
      target_add_compile_flags_if_supported(${target} PRIVATE
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-c++11-compat
        -Wno-undef
        -Wno-reserved-id-macro
        -Wno-gnu-include-next
        -Wno-gcc-compat # For ignoring "'diagnose_if' is a clang extension" warnings
        -Wno-zero-as-null-pointer-constant # FIXME: Remove this and fix all occurrences.
        -Wno-deprecated-dynamic-exception-spec # For auto_ptr
        -Wno-sign-conversion
        -Wno-old-style-cast
        -Wno-deprecated # FIXME: Remove this and fix all occurrences.
        -Wno-shift-sign-overflow # FIXME: Why do we need this with clang-cl but not clang?
        -Wno-double-promotion # FIXME: remove me
      )
    endif()

  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

    target_add_compile_flags_if_supported(${target} PRIVATE
      -Wstrict-aliasing=2
      -Wstrict-overflow=4
      -Wno-attributes
      -Wno-literal-suffix
      -Wno-c++14-compat
      -Wno-noexcept-type
      -Wno-suggest-override
      -Wno-alloc-size-larger-than
      -Wno-deprecated-declarations
      -Wno-dangling-reference
      -Wno-strict-overflow
      -Wno-maybe-uninitialized
      -Wno-strict-aliasing
      )

  endif()
  if (${enable_werror})
    target_add_compile_flags_if_supported(${target} PRIVATE -Werror)
    target_add_compile_flags_if_supported(${target} PRIVATE -WX)
  else()
    # TODO(EricWF) Remove this. We shouldn't be suppressing errors when -Werror is
    # added elsewhere.
    target_add_compile_flags_if_supported(${target} PRIVATE -Wno-error)
  endif()
  if (${enable_pedantic})
    target_add_compile_flags_if_supported(${target} PRIVATE -pedantic)
  endif()
endfunction()
