//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/wasm/RegisterContextWasm.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::wasm;

TEST(WasmVirtualRegisterInfoTest, RegisterNumbers) {
  const WasmVirtualRegisterInfo info(eWasmTagLocal, 3);
  EXPECT_EQ(info.kinds[eRegisterKindLLDB], GetWasmRegister(eWasmTagLocal, 3));
  for (RegisterKind kind : {eRegisterKindEHFrame, eRegisterKindDWARF,
                            eRegisterKindGeneric, eRegisterKindProcessPlugin})
    EXPECT_EQ(info.kinds[kind], LLDB_INVALID_REGNUM) << "kind " << kind;
}
