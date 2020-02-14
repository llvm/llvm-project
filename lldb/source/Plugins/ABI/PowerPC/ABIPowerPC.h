//===-- PowerPC.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIPowerPC_h_
#define liblldb_ABIPowerPC_h_

class ABIPowerPC {
public:
  static void Initialize();
  static void Terminate();
};
#endif
