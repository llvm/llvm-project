# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc configuration options.
The canonical list of user options is in 'libc/config/config.json'.
These options are then processed by CMake and turned into preprocessor
definitions. We don't have this logic in Bazel yet but the list of definitions
is discoverable with the following command:

> git grep -hoE '\bLIBC_COPT_\\w*'  -- '*.h' '*.cpp' | sort -u
"""

# This list of definitions is used to customize LLVM libc.
LIBC_CONFIGURE_OPTIONS = [
    # Documentation in libc/docs/dev/printf_behavior.rst
    # "LIBC_COPT_FLOAT_TO_STR_NO_SPECIALIZE_LD",
    # "LIBC_COPT_FLOAT_TO_STR_NO_TABLE",
    # "LIBC_COPT_FLOAT_TO_STR_USE_DYADIC_FLOAT",
    # "LIBC_COPT_FLOAT_TO_STR_USE_DYADIC_FLOAT_LD",
    # "LIBC_COPT_FLOAT_TO_STR_USE_INT_CALC",
    # "LIBC_COPT_FLOAT_TO_STR_USE_MEGA_LONG_DOUBLE_TABLE",

    # Documentation in libc/src/string/memory_utils/...
    # "LIBC_COPT_MEMCPY_USE_EMBEDDED_TINY",
    # "LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE",
    # "LIBC_COPT_MEMCPY_X86_USE_SOFTWARE_PREFETCHING",
    # "LIBC_COPT_MEMSET_X86_USE_SOFTWARE_PREFETCHING",

    # Documentation in libc/docs/dev/printf_behavior.rst
    # "LIBC_COPT_PRINTF_CONV_ATLAS",
    # "LIBC_COPT_PRINTF_DISABLE_FIXED_POINT",
    # "LIBC_COPT_PRINTF_DISABLE_FLOAT",
    # "LIBC_COPT_PRINTF_DISABLE_INDEX_MODE",
    "LIBC_COPT_PRINTF_DISABLE_WRITE_INT",
    # "LIBC_COPT_PRINTF_HEX_LONG_DOUBLE",
    # "LIBC_COPT_PRINTF_INDEX_ARR_LEN",
    # "LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS",
    # "LIBC_COPT_SCANF_DISABLE_FLOAT",
    # "LIBC_COPT_SCANF_DISABLE_INDEX_MODE",
    "LIBC_COPT_STDIO_USE_SYSTEM_FILE",
    # "LIBC_COPT_STRING_UNSAFE_WIDE_READ",
    # "LIBC_COPT_STRTOFLOAT_DISABLE_CLINGER_FAST_PATH",
    # "LIBC_COPT_STRTOFLOAT_DISABLE_EISEL_LEMIRE",
    # "LIBC_COPT_STRTOFLOAT_DISABLE_SIMPLE_DECIMAL_CONVERSION",

    # Documentation in libc/src/__support/libc_assert.h
    # "LIBC_COPT_USE_C_ASSERT",
]
