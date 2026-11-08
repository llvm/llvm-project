//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_s390x.h"
#include "lldb/Register/Arch/lldb-s390x-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-s390x-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-s390x-register-enums.h"
#include <iterator>


namespace reg_ids = lldb_private::registers::s390x::id;
using namespace lldb_private;

std::shared_ptr<const RegisterDescription>
RegisterInfo_s390x::CreateDescription() {
#include "Register/Arch/RegisterDescriptions_s390x.inc"
  static const auto description = [] {
    RegisterGroup gpr{"General Purpose Registers", "gpr", {}};
    RegisterGroup fpr{"Floating Point Registers", "fpr", {}};
    for (const auto &info : values) {
      auto id = info.id;
      (id <= reg_ids::k_last_gpr ? gpr : fpr).registers.push_back(id);
    }
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::vector<RegisterDescriptor>(
                                        std::begin(values), std::end(values)),
                                    {std::move(gpr), std::move(fpr)})));
  }();
  return description;
}
