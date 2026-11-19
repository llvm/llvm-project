//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERDESCRIPTIONPC_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERDESCRIPTIONPC_H

#include "lldb/Register/RegisterDescription.h"

namespace lldb_private {
inline std::shared_ptr<const RegisterDescription>
CreatePCRegisterDescription(uint32_t address_byte_size) {
  RegisterDescriptor pc;
  pc.id = 0;
  pc.name = "pc";
  pc.alt_name = "pc";
  pc.byte_size = address_byte_size;
  pc.format = lldb::eFormatPointer;
  pc.numbers[lldb::eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          {std::move(pc)}, {{"General Purpose Registers", "GPR", {0}}})));
}
} // namespace lldb_private

#endif
