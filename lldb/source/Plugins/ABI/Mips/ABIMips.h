//===-- Mips.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIMips_h_
#define liblldb_ABIMips_h_

class ABIMips {
public:
  static void Initialize();
  static void Terminate();
};
#endif
