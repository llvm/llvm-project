//===-- DumpRegisterInfo.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/DumpRegisterInfo.h"
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Register/RegisterType.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"

#include <cinttypes>
#include <functional>

using namespace lldb;
using namespace lldb_private;

using SetInfo = std::pair<const char *, uint32_t>;

void lldb_private::DumpRegisterInfo(
    Stream &strm, const RegisterDescription &description,
    const RegisterDescriptor &info, uint32_t terminal_width) {
  std::vector<const char *> invalidates;
  for (RegisterID id : info.invalidate_registers)
    if (const auto *reg = description.Find(id))
      invalidates.push_back(reg->name.c_str());

  // Keep indices so the names can also be used with "register read -s".
  std::vector<SetInfo> in_sets;
  for (size_t index = 0; index < description.GetGroups().size(); ++index) {
    const auto &group = description.GetGroups()[index];
    if (llvm::is_contained(group.registers, info.id))
      in_sets.push_back({group.name.c_str(), index});
  }
  std::vector<const char *> read_from;
  for (RegisterID id : info.value_registers)
    if (const auto *reg = description.Find(id))
      read_from.push_back(reg->name.c_str());

  DoDumpRegisterInfo(
      strm, info.name.c_str(),
      info.alt_name.empty() ? nullptr : info.alt_name.c_str(), info.byte_size,
      invalidates, read_from, in_sets, info.type, terminal_width);
}

template <typename ElementType>
static void DumpList(Stream &strm, const char *title,
                     const std::vector<ElementType> &list,
                     std::function<void(Stream &, ElementType)> emitter) {
  if (list.empty())
    return;

  strm.EOL();
  strm << title;
  bool first = true;
  for (ElementType elem : list) {
    if (!first)
      strm << ", ";
    first = false;
    emitter(strm, elem);
  }
}

void lldb_private::DoDumpRegisterInfo(
    Stream &strm, const char *name, const char *alt_name, uint32_t byte_size,
    const std::vector<const char *> &invalidates,
    const std::vector<const char *> &read_from,
    const std::vector<SetInfo> &in_sets, const RegisterType *register_type,
    uint32_t terminal_width) {
  strm << "       Name: " << name;
  if (alt_name)
    strm << " (" << alt_name << ")";
  strm.EOL();

  // Size in bits may seem obvious for the usual 32 or 64 bit registers.
  // When we get to vector registers, then scalable vector registers, it is very
  // useful to know without the user doing extra work.
  strm.Printf("       Size: %d bytes (%d bits)", byte_size, byte_size * 8);

  std::function<void(Stream &, const char *)> emit_str =
      [](Stream &strm, const char *s) { strm << s; };
  DumpList(strm, "Invalidates: ", invalidates, emit_str);
  DumpList(strm, "  Read from: ", read_from, emit_str);

  std::function<void(Stream &, SetInfo)> emit_set = [](Stream &strm,
                                                       SetInfo info) {
    strm.Printf("%s (index %d)", info.first, info.second);
  };
  DumpList(strm, "    In sets: ", in_sets, emit_set);

  if (auto flags_type =
          llvm::dyn_cast_if_present<RegisterTypeFlags>(register_type)) {
    strm.Printf("\n\n%s", flags_type->AsTable(terminal_width).c_str());

    std::string enumerators = flags_type->DumpEnums(terminal_width);
    if (enumerators.size())
      strm << "\n\n" << enumerators;
  } else if (auto *union_type =
                 llvm::dyn_cast_if_present<RegisterTypeUnion>(register_type)) {
    strm << "\n\n  Union members:";
    for (const RegisterTypeUnion::Field &field : union_type->GetFields()) {
      strm.Printf("\n    %s (%s", field.GetName().c_str(),
                  field.GetType()->GetID().c_str());
      if (std::optional<uint64_t> byte_size = field.GetType()->GetByteSize())
        strm.Printf(", %" PRIu64 " bytes", *byte_size);
      strm.PutChar(')');
    }
  } else if (auto *vector_type =
                 llvm::dyn_cast_if_present<RegisterTypeVector>(register_type)) {
    strm.Printf("\n\n  Vector elements: %u", vector_type->GetCount());
  }
}
