//===-- RegisterContextUnifiedCore.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextUnifiedCore.h"
#include "lldb/Target/Process.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextUnifiedCore::RegisterContextUnifiedCore(
    Thread &thread, uint32_t concrete_frame_idx, ThreadRegisterContextSP core,
    StructuredData::ObjectSP metadata)
    : ThreadRegisterContext(thread, concrete_frame_idx) {
  std::vector<RegisterDescriptor> registers;
  std::vector<RegisterGroup> groups;
  RegisterDescription::OwnedTypes types;
  if (auto description = core->GetDescription())
    llvm::append_range(types, description->GetOwnedTypes());

  // Keep the core's group order and its available, independent values. A core
  // snapshot owns values; it has no artificial combined native storage image.
  auto append = [&](RegisterDescriptor reg, RegisterValue value,
                    RegisterGroup &group) {
    reg.id = registers.size();
    reg.value_registers.clear();
    reg.invalidate_registers.clear();
    reg.zero_extend_parent = LLDB_INVALID_REGNUM;
    group.registers.push_back(reg.id);
    m_values.emplace(reg.id, std::move(value));
    registers.push_back(std::move(reg));
  };
  for (const auto &source : core->GetDescription()->GetGroups()) {
    RegisterGroup group{source.name, source.short_name, {}};
    for (RegisterID id : source.registers) {
      const auto *reg = core->GetDescription()->Find(id);
      if (!reg || !reg->value_registers.empty())
        continue;
      RegisterValue value;
      if (!llvm::errorToBool(
              core->ReadRegister((reg ? reg->id : LLDB_INVALID_REGNUM))
                  .moveInto(value)))
        append(*reg, std::move(value), group);
    }
    groups.push_back(std::move(group));
  }

  auto *metadata_dict = metadata ? metadata->GetAsDictionary() : nullptr;
  auto *object = metadata_dict
                     ? metadata_dict->GetValueForKey("register_info").get()
                     : nullptr;
  auto *definitions = object ? object->GetAsDictionary() : nullptr;
  auto additional =
      definitions ? CreateRegisterImageLayout(
                        *definitions,
                        thread.GetProcess()->GetTarget().GetArchitecture())
                  : nullptr;
  StructuredData::Array *input = nullptr;
  if (additional && definitions->GetValueForKeyAsArray("registers", input)) {
    auto description = additional->GetDescription();
    llvm::append_range(types, description->GetOwnedTypes());
    for (const auto &source : description->GetGroups()) {
      auto group = llvm::find_if(groups, [&](const auto &candidate) {
        return candidate.name == source.name;
      });
      if (group == groups.end()) {
        groups.push_back({source.name, source.short_name, {}});
        group = std::prev(groups.end());
      }
      for (RegisterID id : source.registers) {
        const auto &reg = *description->Find(id);
        const size_t index = *description->GetRegisterIndex(id);
        auto item = input->GetItemAtIndex(index);
        auto *entry = item ? item->GetAsDictionary() : nullptr;
        uint64_t bits = 0;
        if (!entry || !entry->GetValueForKeyAsInteger("value", bits))
          bits = 0;
        RegisterValue value;
        value.SetUInt(bits, reg.byte_size);
        append(reg, std::move(value), *group);
      }
    }
  }
  std::vector<RegisterDescriptor> presentation;
  for (const auto &group : groups)
    for (RegisterID id : group.registers)
      presentation.push_back(registers[id]);
  registers = std::move(presentation);
  SetRegisterDescription(std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(registers), std::move(groups), std::move(types)))));
}

llvm::Expected<RegisterValue>
RegisterContextUnifiedCore::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg = description ? description->Find(id) : nullptr;
  if (!reg)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  auto found = reg ? m_values.find(reg->id) : m_values.end();
  if (found == m_values.end())
    return llvm::createStringError("register value is unavailable");
  value = found->second;
  return value;
}

llvm::Error
RegisterContextUnifiedCore::WriteRegister(RegisterID id,
                                          const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}
