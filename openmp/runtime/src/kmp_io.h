/*
 * kmp_io.h -- RTL IO header file.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_IO_H
#define KMP_IO_H

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------ */

enum kmp_io { kmp_out = 0, kmp_err };

extern kmp_bootstrap_lock_t __kmp_stdio_lock; /* Control stdio functions */
extern kmp_bootstrap_lock_t
    __kmp_console_lock; /* Control console initialization */

extern void __kmp_vprintf(enum kmp_io stream, char const *format, va_list ap);
extern void __kmp_printf(char const *format, ...);
extern void __kmp_printf_no_lock(char const *format, ...);
extern void __kmp_fprintf(enum kmp_io stream, char const *format, ...);
extern void __kmp_close_console(void);

#ifdef __cplusplus
}
#endif

#endif /* KMP_IO_H */
