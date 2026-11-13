//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/wasm/RegisterContextWasm.h"
#include "lldb/Register/Arch/lldb-wasm-register-enums.h"
#include "gtest/gtest.h"

namespace reg_ids = lldb_private::registers::wasm::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::wasm;

TEST(WasmRegisterIdentityTest, EncodesKindAndIndex) {
  uint32_t id = reg_ids::GetWasmRegister(reg_ids::eWasmTagLocal, 3);
  EXPECT_EQ(reg_ids::GetWasmVirtualRegisterTag(id), reg_ids::eWasmTagLocal);
  EXPECT_EQ(reg_ids::GetWasmVirtualRegisterIndex(id), 3u);
  EXPECT_NE(id, reg_ids::GetWasmRegister(reg_ids::eWasmTagGlobal, 3));
}
