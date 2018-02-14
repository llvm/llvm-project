/*
 * kmp_io.cpp -- RTL IO
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __ABSOFT_WIN
#include <sys/types.h>
#endif

#include "kmp.h" // KMP_GTID_DNE, __kmp_debug_buf, etc
#include "kmp_io.h"
#include "kmp_lock.h"
#include "kmp_os.h"
#include "kmp_str.h"

#if KMP_OS_WINDOWS
#pragma warning(push)
#pragma warning(disable : 271 310)
#include <windows.h>
#pragma warning(pop)
#endif

/* ------------------------------------------------------------------------ */

kmp_bootstrap_lock_t __kmp_stdio_lock = KMP_BOOTSTRAP_LOCK_INITIALIZER(
    __kmp_stdio_lock); /* Control stdio functions */
kmp_bootstrap_lock_t __kmp_console_lock = KMP_BOOTSTRAP_LOCK_INITIALIZER(
    __kmp_console_lock); /* Control console initialization */

#if KMP_OS_WINDOWS

#ifdef KMP_DEBUG
/* __kmp_stdout is used only for dev build */
static HANDLE __kmp_stdout = NULL;
#endif
static HANDLE __kmp_stderr = NULL;
static int __kmp_console_exists = FALSE;
static kmp_str_buf_t __kmp_console_buf;

static int is_console(void) {
  char buffer[128];
  DWORD rc = 0;
  DWORD err = 0;
  // Try to get console title.
  SetLastError(0);
  // GetConsoleTitle does not reset last error in case of success or short
  // buffer, so we need to clear it explicitly.
  rc = GetConsoleTitle(buffer, sizeof(buffer));
  if (rc == 0) {
    // rc == 0 means getting console title failed. Let us find out why.
    err = GetLastError();
    // err == 0 means buffer too short (we suppose console exists).
    // In Window applications we usually have err == 6 (invalid handle).
  }
  return rc > 0 || err == 0;
}

void __kmp_close_console(void) {
  /* wait until user presses return before closing window */
  /* TODO only close if a window was opened */
  if (__kmp_console_exists) {
#ifdef KMP_DEBUG
    /* standard out is used only in dev build */
    __kmp_stdout = NULL;
#endif
    __kmp_stderr = NULL;
    __kmp_str_buf_free(&__kmp_console_buf);
    __kmp_console_exists = FALSE;
  }
}

/* For windows, call this before stdout, stderr, or stdin are used.
   It opens a console window and starts processing */
static void __kmp_redirect_output(void) {
  __kmp_acquire_bootstrap_lock(&__kmp_console_lock);

  if (!__kmp_console_exists) {
#ifdef KMP_DEBUG
    /* standard out is used only in dev build */
    HANDLE ho;
#endif
    HANDLE he;

    __kmp_str_buf_init(&__kmp_console_buf);

    AllocConsole();
// We do not check the result of AllocConsole because
//  1. the call is harmless
//  2. it is not clear how to communicate failue
//  3. we will detect failure later when we get handle(s)

#ifdef KMP_DEBUG
    ho = GetStdHandle(STD_OUTPUT_HANDLE);
    if (ho == INVALID_HANDLE_VALUE || ho == NULL) {

      DWORD err = GetLastError();
      // TODO: output error somehow (maybe message box)
      __kmp_stdout = NULL;

    } else {

      __kmp_stdout = ho; // temporary code, need new global for ho
    }
#endif
    he = GetStdHandle(STD_ERROR_HANDLE);
    if (he == INVALID_HANDLE_VALUE || he == NULL) {

      DWORD err = GetLastError();
      // TODO: output error somehow (maybe message box)
      __kmp_stderr = NULL;

    } else {

      __kmp_stderr = he; // temporary code, need new global
    }
    __kmp_console_exists = TRUE;
  }
  __kmp_release_bootstrap_lock(&__kmp_console_lock);
}

#else
#define __kmp_stderr (stderr)
#endif /* KMP_OS_WINDOWS */

void __kmp_vprintf(enum kmp_io __kmp_io, char const *format, va_list ap) {
#if KMP_OS_WINDOWS
  if (!__kmp_console_exists) {
    __kmp_redirect_output();
  }
  if (!__kmp_stderr && __kmp_io == kmp_err) {
    return;
  }
#ifdef KMP_DEBUG
  if (!__kmp_stdout && __kmp_io == kmp_out) {
    return;
  }
#endif
#endif /* KMP_OS_WINDOWS */

  if (__kmp_debug_buf && __kmp_debug_buffer != NULL) {

    int dc = (__kmp_debug_buf_atomic ? KMP_TEST_THEN_INC32(&__kmp_debug_count)
                                     : __kmp_debug_count++) %
             __kmp_debug_buf_lines;
    char *db = &__kmp_debug_buffer[dc * __kmp_debug_buf_chars];
    int chars = 0;

#ifdef KMP_DEBUG_PIDS
    chars = KMP_SNPRINTF(db, __kmp_debug_buf_chars, "pid=%d: ",
                         (kmp_int32)getpid());
#endif
    chars += KMP_VSNPRINTF(db, __kmp_debug_buf_chars, format, ap);

    if (chars + 1 > __kmp_debug_buf_chars) {
      if (chars + 1 > __kmp_debug_buf_warn_chars) {
#if KMP_OS_WINDOWS
        DWORD count;
        __kmp_str_buf_print(&__kmp_console_buf, "OMP warning: Debugging buffer "
                                                "overflow; increase "
                                                "KMP_DEBUG_BUF_CHARS to %d\n",
                            chars + 1);
        WriteFile(__kmp_stderr, __kmp_console_buf.str, __kmp_console_buf.used,
                  &count, NULL);
        __kmp_str_buf_clear(&__kmp_console_buf);
#else
        fprintf(__kmp_stderr, "OMP warning: Debugging buffer overflow; "
                              "increase KMP_DEBUG_BUF_CHARS to %d\n",
                chars + 1);
        fflush(__kmp_stderr);
#endif
        __kmp_debug_buf_warn_chars = chars + 1;
      }
      /* terminate string if overflow occurred */
      db[__kmp_debug_buf_chars - 2] = '\n';
      db[__kmp_debug_buf_chars - 1] = '\0';
    }
  } else {
#if KMP_OS_WINDOWS
    DWORD count;
#ifdef KMP_DEBUG_PIDS
    __kmp_str_buf_print(&__kmp_console_buf, "pid=%d: ", (kmp_int32)getpid());
#endif
    __kmp_str_buf_vprint(&__kmp_console_buf, format, ap);
    WriteFile(__kmp_stderr, __kmp_console_buf.str, __kmp_console_buf.used,
              &count, NULL);
    __kmp_str_buf_clear(&__kmp_console_buf);
#else
#ifdef KMP_DEBUG_PIDS
    fprintf(__kmp_stderr, "pid=%d: ", (kmp_int32)getpid());
#endif
    vfprintf(__kmp_stderr, format, ap);
    fflush(__kmp_stderr);
#endif
  }
}

void __kmp_printf(char const *format, ...) {
  va_list ap;
  va_start(ap, format);

  __kmp_acquire_bootstrap_lock(&__kmp_stdio_lock);
  __kmp_vprintf(kmp_err, format, ap);
  __kmp_release_bootstrap_lock(&__kmp_stdio_lock);

  va_end(ap);
}

void __kmp_printf_no_lock(char const *format, ...) {
  va_list ap;
  va_start(ap, format);

  __kmp_vprintf(kmp_err, format, ap);

  va_end(ap);
}
