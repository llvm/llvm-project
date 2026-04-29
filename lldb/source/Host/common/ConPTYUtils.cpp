//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ConPTYUtils.h"
#include <cstring>

using namespace lldb_private;

void lldb_private::StripConPTYSequences(void *data, size_t &len,
                                        bool strip_init) {
  auto *buf = static_cast<char *>(data);
  char *out = buf;
  const char *in = buf;
  const char *end = buf + len;

  while (in < end) {
    if (*in != '\x1b') {
      *out++ = *in++;
      continue;
    }

    size_t remaining = end - in;

    // \x1b[6n - cursor-position query (PSEUDOCONSOLE_INHERIT_CURSOR init)
    if (remaining >= 4 && memcmp(in, "\x1b[6n", 4) == 0) {
      in += 4;
      continue;
    }

    if (strip_init) {
      // \x1b[m - SGR reset (ConPTY init)
      if (remaining >= 3 && memcmp(in, "\x1b[m", 3) == 0) {
        in += 3;
        continue;
      }

      // \x1b[?25h - show cursor (ConPTY init)
      if (remaining >= 6 && memcmp(in, "\x1b[?25h", 6) == 0) {
        in += 6;
        continue;
      }
    }

    // \x1b[?9001h / \x1b[?9001l - Win32 Input Mode enable/disable
    if (remaining >= 8 && memcmp(in, "\x1b[?9001", 7) == 0 &&
        (in[7] == 'h' || in[7] == 'l')) {
      in += 8;
      continue;
    }

    // \x1b[?1004h / \x1b[?1004l - focus-event reporting enable/disable
    if (remaining >= 8 && memcmp(in, "\x1b[?1004", 7) == 0 &&
        (in[7] == 'h' || in[7] == 'l')) {
      in += 8;
      continue;
    }

    // \x1b]0;...\x07 - ConPTY window-title OSC sequence
    if (remaining >= 4 && in[1] == ']' && in[2] == '0' && in[3] == ';') {
      const char *bel =
          static_cast<const char *>(memchr(in + 4, '\x07', end - in - 4));
      if (bel)
        in = bel + 1;
      else
        in = end;
      continue;
    }

    *out++ = *in++;
  }

  len = static_cast<size_t>(out - buf);
}
