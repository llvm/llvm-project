//===-- X86.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIX86_h_
#define liblldb_ABIX86_h_

class ABIX86 {
public:
  static void Initialize();
  static void Terminate();
};
#endif
