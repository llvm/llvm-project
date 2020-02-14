//===-- ARM.h -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIARM_h_
#define liblldb_ABIARM_h_

class ABIARM {
public:
  static void Initialize();
  static void Terminate();
};
#endif
