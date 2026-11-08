//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_ppc64le.h"
#include "lldb/Register/Arch/lldb-ppc64le-ehframe-register-enums.h"
#include "RegisterInfo_powerpc_internal.h"
#include <cassert>
#include <cstddef>
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "llvm/ADT/STLExtras.h"
#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"
#include "lldb/lldb-private-types.h"
#include "llvm/Support/Compiler.h"
#include <iterator>

using namespace lldb_private;

static llvm::ArrayRef<RegisterDescriptor> Values8() {
  constexpr uint32_t word_size = 8;
#include "Register/Arch/RegisterDescriptions_ppc64le.inc"
  return values;
}

std::shared_ptr<const RegisterDescription>
RegisterInfo_ppc64le::CreateDescription(const ArchSpec &) {
  static const auto description8 = std::make_shared<RegisterDescription>(
      MakePowerPCDescription(Values8(), true));
  return description8;
}
