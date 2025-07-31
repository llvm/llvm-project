//===-- RegisterContextUnifiedCore.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextUnifiedCore.h"
#include "lldb/Target/DynamicRegisterInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/StructuredData.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextUnifiedCore::RegisterContextUnifiedCore(
    Thread &thread, uint32_t concrete_frame_idx,
    RegisterContextSP core_thread_regctx_sp,
    StructuredData::ObjectSP metadata_thread_registers)
    : RegisterContext(thread, concrete_frame_idx) {

  ProcessSP process_sp(thread.GetProcess());
  Target &target = process_sp->GetTarget();
  StructuredData::Dictionary *metadata_registers_dict = nullptr;

  // If we have thread metadata, check if the keys for register
  // definitions are present; if not, clear the ObjectSP.
  if (metadata_thread_registers &&
      metadata_thread_registers->GetAsDictionary() &&
      metadata_thread_registers->GetAsDictionary()->HasKey("register_info")) {
    metadata_registers_dict = metadata_thread_registers->GetAsDictionary()
                                  ->GetValueForKey("register_info")
                                  ->GetAsDictionary();
    if (metadata_registers_dict)
      if (!metadata_registers_dict->HasKey("sets") ||
          !metadata_registers_dict->HasKey("registers"))
        metadata_registers_dict = nullptr;
  }

  // When creating a register set list from the two sources,
  // the LC_THREAD aka core_thread_regctx_sp register sets
  // will be used at the same indexes.
  // Any additional sets named by the thread metadata registers
  // will be added after them.  If the thread metadata
  // specify a set with the same name as LC_THREAD, the already-used
  // index from the core register context will be used in
  // the RegisterInfo.
  std::map<size_t, size_t> metadata_regset_to_combined_regset;

  // Calculate the total size of the register store buffer we need
  // for all registers.  The corefile register definitions may include
  // RegisterInfo descriptions of registers that aren't actually
  // available.  For simplicity, calculate the size of all registers
  // as if they are available, so we can maintain the same offsets into
  // the buffer.
  uint32_t core_buffer_end = 0;
  for (size_t idx = 0; idx < core_thread_regctx_sp->GetRegisterCount(); idx++) {
    const RegisterInfo *reginfo =
        core_thread_regctx_sp->GetRegisterInfoAtIndex(idx);
    core_buffer_end =
        std::max(reginfo->byte_offset + reginfo->byte_size, core_buffer_end);
  }

  // Add metadata register sizes to the total buffer size.
  uint32_t combined_buffer_end = core_buffer_end;
  if (metadata_registers_dict) {
    StructuredData::Array *registers = nullptr;
    if (metadata_registers_dict->GetValueForKeyAsArray("registers", registers))
      registers->ForEach(
          [&combined_buffer_end](StructuredData::Object *ent) -> bool {
            uint32_t bitsize;
            if (!ent->GetAsDictionary()->GetValueForKeyAsInteger("bitsize",
                                                                 bitsize))
              return false;
            combined_buffer_end += (bitsize / 8);
            return true;
          });
  }
  m_register_data.resize(combined_buffer_end, 0);

  // Copy the core register values into our combined data buffer,
  // skip registers that are contained within another (e.g. w0 vs. x0)
  // and registers that return as "unavailable".
  for (size_t idx = 0; idx < core_thread_regctx_sp->GetRegisterCount(); idx++) {
    const RegisterInfo *reginfo =
        core_thread_regctx_sp->GetRegisterInfoAtIndex(idx);
    RegisterValue val;
    if (!reginfo->value_regs &&
        core_thread_regctx_sp->ReadRegister(reginfo, val))
      memcpy(m_register_data.data() + reginfo->byte_offset, val.GetBytes(),
             val.GetByteSize());
  }

  // Set 'offset' fields for each register definition into our combined
  // register data buffer. DynamicRegisterInfo needs this field set to
  // parse the JSON.
  // Also copy the values of the registers into our register data buffer.
  if (metadata_registers_dict) {
    size_t offset = core_buffer_end;
    ByteOrder byte_order = core_thread_regctx_sp->GetByteOrder();
    StructuredData::Array *registers;
    if (metadata_registers_dict->GetValueForKeyAsArray("registers", registers))
      registers->ForEach([this, &offset,
                          byte_order](StructuredData::Object *ent) -> bool {
        uint64_t bitsize;
        uint64_t value;
        if (!ent->GetAsDictionary()->GetValueForKeyAsInteger("bitsize",
                                                             bitsize))
          return false;
        if (!ent->GetAsDictionary()->GetValueForKeyAsInteger("value", value)) {
          // We had a bitsize but no value, so move the offset forward I guess.
          offset += (bitsize / 8);
          return false;
        }
        ent->GetAsDictionary()->AddIntegerItem("offset", offset);
        Status error;
        const int bytesize = bitsize / 8;
        switch (bytesize) {
        case 2: {
          Scalar value_scalar((uint16_t)value);
          value_scalar.GetAsMemoryData(m_register_data.data() + offset,
                                       bytesize, byte_order, error);
          offset += bytesize;
        } break;
        case 4: {
          Scalar value_scalar((uint32_t)value);
          value_scalar.GetAsMemoryData(m_register_data.data() + offset,
                                       bytesize, byte_order, error);
          offset += bytesize;
        } break;
        case 8: {
          Scalar value_scalar((uint64_t)value);
          value_scalar.GetAsMemoryData(m_register_data.data() + offset,
                                       bytesize, byte_order, error);
          offset += bytesize;
        } break;
        }
        return true;
      });
  }

  // Create a DynamicRegisterInfo from the metadata JSON.
  std::unique_ptr<DynamicRegisterInfo> additional_reginfo_up;
  if (metadata_registers_dict)
    additional_reginfo_up = DynamicRegisterInfo::Create(
        *metadata_registers_dict, target.GetArchitecture());

  // Put the RegisterSet names in the constant string pool,
  // to sidestep lifetime issues of char*'s.
  auto copy_regset_name = [](RegisterSet &dst, const RegisterSet &src) {
    dst.name = ConstString(src.name).AsCString();
    if (src.short_name)
      dst.short_name = ConstString(src.short_name).AsCString();
    else
      dst.short_name = nullptr;
  };

  // Copy the core thread register sets into our combined register set list.
  // RegisterSet indexes will be identical for the LC_THREAD RegisterContext.
  for (size_t idx = 0; idx < core_thread_regctx_sp->GetRegisterSetCount();
       idx++) {
    RegisterSet new_set;
    const RegisterSet *old_set = core_thread_regctx_sp->GetRegisterSet(idx);
    copy_regset_name(new_set, *old_set);
    m_register_sets.push_back(new_set);
  }

  // Add any additional metadata RegisterSets to our combined RegisterSet array.
  if (additional_reginfo_up) {
    for (size_t idx = 0; idx < additional_reginfo_up->GetNumRegisterSets();
         idx++) {
      // See if this metadata RegisterSet name matches one already present
      // from the LC_THREAD RegisterContext.
      bool found_match = false;
      const RegisterSet *old_set = additional_reginfo_up->GetRegisterSet(idx);
      for (size_t jdx = 0; jdx < m_register_sets.size(); jdx++) {
        if (strcmp(m_register_sets[jdx].name, old_set->name) == 0) {
          metadata_regset_to_combined_regset[idx] = jdx;
          found_match = true;
          break;
        }
      }
      // This metadata RegisterSet is a new one.
      // Add it to the combined RegisterSet array.
      if (!found_match) {
        RegisterSet new_set;
        copy_regset_name(new_set, *old_set);
        metadata_regset_to_combined_regset[idx] = m_register_sets.size();
        m_register_sets.push_back(new_set);
      }
    }
  }

  // Set up our combined RegisterInfo array, one RegisterSet at a time.
  for (size_t combined_regset_idx = 0;
       combined_regset_idx < m_register_sets.size(); combined_regset_idx++) {
    uint32_t registers_this_regset = 0;

    // Copy all LC_THREAD RegisterInfos that have a value into our
    // combined RegisterInfo array.  (the LC_THREAD RegisterContext
    // may describe registers that were not provided in this thread)
    //
    // LC_THREAD register set indexes are identical to the combined
    // register set indexes.  The combined register set array may have
    // additional entries.
    if (combined_regset_idx < core_thread_regctx_sp->GetRegisterSetCount()) {
      const RegisterSet *regset =
          core_thread_regctx_sp->GetRegisterSet(combined_regset_idx);
      // Copy all the registers that have values in.
      for (size_t j = 0; j < regset->num_registers; j++) {
        uint32_t reg_idx = regset->registers[j];
        const RegisterInfo *reginfo =
            core_thread_regctx_sp->GetRegisterInfoAtIndex(reg_idx);
        RegisterValue val;
        if (!reginfo->value_regs &&
            core_thread_regctx_sp->ReadRegister(reginfo, val)) {
          m_regset_regnum_collection[combined_regset_idx].push_back(
              m_register_infos.size());
          m_register_infos.push_back(*reginfo);
          registers_this_regset++;
        }
      }
    }

    // Copy all the metadata RegisterInfos into our combined combined
    // RegisterInfo array.
    // The metadata may add registers to one of the LC_THREAD register sets,
    // or its own newly added register sets.  metadata_regset_to_combined_regset
    // has the association of the RegisterSet indexes between the two.
    if (additional_reginfo_up) {
      // Find the register set in the metadata that matches this register
      // set, then copy all its RegisterInfos.
      for (size_t setidx = 0;
           setidx < additional_reginfo_up->GetNumRegisterSets(); setidx++) {
        if (metadata_regset_to_combined_regset[setidx] == combined_regset_idx) {
          const RegisterSet *regset =
              additional_reginfo_up->GetRegisterSet(setidx);
          for (size_t j = 0; j < regset->num_registers; j++) {
            uint32_t reg_idx = regset->registers[j];
            const RegisterInfo *reginfo =
                additional_reginfo_up->GetRegisterInfoAtIndex(reg_idx);
            m_regset_regnum_collection[combined_regset_idx].push_back(
                m_register_infos.size());
            m_register_infos.push_back(*reginfo);
            registers_this_regset++;
          }
        }
      }
    }
    m_register_sets[combined_regset_idx].num_registers = registers_this_regset;
    m_register_sets[combined_regset_idx].registers =
        m_regset_regnum_collection[combined_regset_idx].data();
  }
}

size_t RegisterContextUnifiedCore::GetRegisterCount() {
  return m_register_infos.size();
}

const RegisterInfo *
RegisterContextUnifiedCore::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < m_register_infos.size())
    return &m_register_infos[reg];
  return nullptr;
}

size_t RegisterContextUnifiedCore::GetRegisterSetCount() {
  return m_register_sets.size();
}

const RegisterSet *RegisterContextUnifiedCore::GetRegisterSet(size_t set) {
  if (set < m_register_sets.size())
    return &m_register_sets[set];
  return nullptr;
}

bool RegisterContextUnifiedCore::ReadRegister(
    const lldb_private::RegisterInfo *reg_info,
    lldb_private::RegisterValue &value) {
  if (!reg_info)
    return false;
  if (ProcessSP process_sp = m_thread.GetProcess()) {
    DataExtractor regdata(m_register_data.data(), m_register_data.size(),
                          process_sp->GetByteOrder(),
                          process_sp->GetAddressByteSize());
    offset_t offset = reg_info->byte_offset;
    switch (reg_info->byte_size) {
    case 2:
      value.SetUInt16(regdata.GetU16(&offset));
      break;
    case 4:
      value.SetUInt32(regdata.GetU32(&offset));
      break;
    case 8:
      value.SetUInt64(regdata.GetU64(&offset));
      break;
    default:
      return false;
    }
    return true;
  }
  return false;
}

bool RegisterContextUnifiedCore::WriteRegister(
    const lldb_private::RegisterInfo *reg_info,
    const lldb_private::RegisterValue &value) {
  return false;
}
