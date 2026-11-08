//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_powerpc.h"
#include "lldb/Register/Arch/lldb-powerpc-ehframe-register-enums.h"
#include "RegisterInfo_powerpc_internal.h"
#include "lldb/Register/Arch/lldb-powerpc-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-powerpc-register-enums.h"
#include "lldb/lldb-private-types.h"
#include <cassert>
#include <cstddef>
using namespace lldb_private;

static llvm::ArrayRef<RegisterDescriptor> Values32() {
  constexpr uint32_t word_size = 4;
#include "Register/Arch/RegisterDescriptions_powerpc.inc"
  return values;
}
static llvm::ArrayRef<RegisterDescriptor> Values64() {
  constexpr uint32_t word_size = 8;
#include "Register/Arch/RegisterDescriptions_powerpc.inc"
  return values;
}
std::shared_ptr<const RegisterDescription>
RegisterInfo_powerpc::CreateDescription(uint32_t value_bytes) {
  assert(value_bytes == 4 || value_bytes == 8);
  static const auto description32 = std::make_shared<RegisterDescription>(
      MakePowerPCDescription(Values32(), true));
  static const auto description64 = std::make_shared<RegisterDescription>(
      MakePowerPCDescription(Values64(), true));
  return value_bytes == 4 ? description32 : description64;
}
