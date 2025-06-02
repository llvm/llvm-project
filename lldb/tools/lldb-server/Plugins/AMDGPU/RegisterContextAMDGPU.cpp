//===-- RegisterContextAMDGPU.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextAMDGPU.h"

#include "LLDBServerPluginAMDGPU.h"
#include "ProcessAMDGPU.h"
#include "ThreadAMDGPU.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <unordered_map>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

static size_t kNumRegs = 0;
static std::vector<RegisterSet> g_reg_sets;
/// Define all of the information about all registers. The register info structs
/// are accessed by the LLDB register numbers, which are defined above.
static std::vector<RegisterInfo> g_reg_infos;
size_t g_register_buffer_size = 0;
static std::unordered_map<uint32_t, amd_dbgapi_register_id_t>
    g_lldb_num_to_amd_reg_id;

bool RegisterContextAMDGPU::InitRegisterInfos() {
  if (!g_reg_infos.empty())
    return true;
  amd_dbgapi_status_t status;
  ThreadAMDGPU *thread = (ThreadAMDGPU *)&m_thread;
  amd_dbgapi_architecture_id_t architecture_id =
      thread->GetProcess().m_debugger->m_architecture_id;
  // Define custom hash functions for register IDs
  struct RegisterClassIdHash {
    std::size_t operator()(const amd_dbgapi_register_class_id_t &id) const {
      return std::hash<uint64_t>{}(id.handle);
    }
  };
  struct RegisterClassIdEqual {
    bool operator()(const amd_dbgapi_register_class_id_t &lhs,
                    const amd_dbgapi_register_class_id_t &rhs) const {
      return lhs.handle == rhs.handle;
    }
  };

  struct RegisterIdHash {
    std::size_t operator()(const amd_dbgapi_register_id_t &id) const {
      return std::hash<uint64_t>{}(id.handle);
    }
  };
  struct RegisterIdEqual {
    bool operator()(const amd_dbgapi_register_id_t &lhs,
                    const amd_dbgapi_register_id_t &rhs) const {
      return lhs.handle == rhs.handle;
    }
  };

  /* Get register class ids.  */
  size_t register_class_count;
  amd_dbgapi_register_class_id_t *register_class_ids;
  status = amd_dbgapi_architecture_register_class_list(
      architecture_id, &register_class_count, &register_class_ids);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "Failed to get register class list from amd-dbgapi");
    return false;
  }

  // Get register class names.
  std::unordered_map<amd_dbgapi_register_class_id_t, std::string,
                     RegisterClassIdHash, RegisterClassIdEqual>
      register_class_names;
  for (size_t i = 0; i < register_class_count; ++i) {
    char *bytes;
    status = amd_dbgapi_architecture_register_class_get_info(
        register_class_ids[i], AMD_DBGAPI_REGISTER_CLASS_INFO_NAME,
        sizeof(bytes), &bytes);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "Failed to get register class name from amd-dbgapi");
      return false;
    }

    // gdb::unique_xmalloc_ptr<char> name(bytes);
    register_class_names.emplace(register_class_ids[i], bytes);
  }

  /* Get all register count. */
  size_t register_count;
  amd_dbgapi_register_id_t *register_ids;
  status = amd_dbgapi_architecture_register_list(
      architecture_id, &register_count, &register_ids);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "Failed to get register list from amd-dbgapi");
    return false;
  }
  kNumRegs = register_count;

  std::unordered_map<amd_dbgapi_register_class_id_t,
                     std::vector<amd_dbgapi_register_id_t>, RegisterClassIdHash,
                     RegisterClassIdEqual>
      register_class_to_register_ids;
  for (size_t i = 0; i < register_class_count; ++i) {
    for (size_t j = 0; j < register_count; ++j) {
      amd_dbgapi_register_class_state_t register_class_state;
      status = amd_dbgapi_register_is_in_register_class(
          register_class_ids[i], register_ids[j], &register_class_state);
      if (status == AMD_DBGAPI_STATUS_SUCCESS &&
          register_class_state == AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER) {
        register_class_to_register_ids[register_class_ids[i]].push_back(
            register_ids[j]);
        break; // TODO: can a register be in multiple classes?
      }
    }
  }

  std::vector<amd_dbgapi_register_properties_t> all_register_properties;
  all_register_properties.resize(register_count);
  for (size_t regnum = 0; regnum < register_count; ++regnum) {
    auto &register_properties = all_register_properties[regnum];
    if (amd_dbgapi_register_get_info(
            register_ids[regnum], AMD_DBGAPI_REGISTER_INFO_PROPERTIES,
            sizeof(register_properties),
            &register_properties) != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "Failed to get register properties from amd-dbgapi");
      return false;
    }
  }

  std::vector<int> dwarf_regnum_to_gdb_regnum;
  std::unordered_map<amd_dbgapi_register_id_t, std::string, RegisterIdHash,
                     RegisterIdEqual>
      register_names;
  for (size_t i = 0; i < register_count; ++i) {
    /* Get register name.  */
    char *bytes;
    status = amd_dbgapi_register_get_info(
        register_ids[i], AMD_DBGAPI_REGISTER_INFO_NAME, sizeof(bytes), &bytes);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      register_names[register_ids[i]] = bytes;
      free(bytes);
    }

    /* Get register DWARF number.  */
    uint64_t dwarf_num;
    status = amd_dbgapi_register_get_info(register_ids[i],
                                          AMD_DBGAPI_REGISTER_INFO_DWARF,
                                          sizeof(dwarf_num), &dwarf_num);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      if (dwarf_num >= dwarf_regnum_to_gdb_regnum.size())
        dwarf_regnum_to_gdb_regnum.resize(dwarf_num + 1, -1);

      dwarf_regnum_to_gdb_regnum[dwarf_num] = i;
    }
  }

  amd_dbgapi_register_id_t pc_register_id;
  status = amd_dbgapi_architecture_get_info(
      architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER,
      sizeof(pc_register_id), &pc_register_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "Failed to get PC register from amd-dbgapi");
    return false;
  }
  // Initialize g_reg_infos with register information from AMD dbgapi
  g_reg_infos.resize(register_count);

  // Map from register class ID to register numbers for that class
  std::unordered_map<amd_dbgapi_register_class_id_t, std::vector<uint32_t>,
                     RegisterClassIdHash, RegisterClassIdEqual>
      register_class_to_lldb_regnums;
  // Populate g_reg_infos with register information from AMD dbgapi
  for (size_t i = 0; i < register_count; ++i) {
    amd_dbgapi_register_id_t reg_id = register_ids[i];
    RegisterInfo &reg_info = g_reg_infos[i];

    // Set register name from AMD dbgapi
    auto name_it = register_names.find(reg_id);
    if (name_it != register_names.end()) {
      reg_info.name = strdup(name_it->second.c_str());
      reg_info.alt_name = nullptr;
    } else {
      // Fallback name if not found
      char name[16];
      snprintf(name, sizeof(name), "reg%zu", i);
      reg_info.name = strdup(name);
      reg_info.alt_name = nullptr;
    }

    // Get register size from AMD dbgapi
    uint64_t reg_size;
    status = amd_dbgapi_register_get_info(reg_id, AMD_DBGAPI_REGISTER_INFO_SIZE,
                                          sizeof(reg_size), &reg_size);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      reg_info.byte_size = reg_size;
    } else {
      reg_info.byte_size = 8; // Default to 64-bit registers
    }
    reg_info.byte_offset = g_register_buffer_size; // Simple offset calculation
    g_register_buffer_size += reg_info.byte_size;

    // Set encoding and format based on register name
    std::string reg_name =
        name_it != register_names.end() ? name_it->second : "";

    // Check if register name contains indicators of its type
    // TODO: is this the correct way to do this?
    if (reg_name.find("float") != std::string::npos ||
        reg_name.find("fp") != std::string::npos) {
      reg_info.encoding = eEncodingIEEE754;
      reg_info.format = eFormatFloat;
    } else if (reg_name.find("vec") != std::string::npos ||
               reg_name.find("simd") != std::string::npos) {
      reg_info.encoding = eEncodingVector;
      reg_info.format = eFormatVectorOfUInt8;
    } else if (reg_info.byte_size > 8) {
      // TODO: check AMD_DBGAPI_REGISTER_INFO_TYPE and assign encoding/format.
      reg_info.encoding = eEncodingVector;
      reg_info.format = eFormatVectorOfUInt8;
    } else {
      // Default for other types
      reg_info.encoding = eEncodingUint;
      reg_info.format = eFormatHex;
    }

    // Set register kinds
    reg_info.kinds[eRegisterKindLLDB] = i; // LLDB register number is the index
    g_lldb_num_to_amd_reg_id[i] =
        reg_id; // Map from LLDB register number to AMD

    // Set DWARF register number if available
    uint64_t dwarf_num;
    status = amd_dbgapi_register_get_info(
        reg_id, AMD_DBGAPI_REGISTER_INFO_DWARF, sizeof(dwarf_num), &dwarf_num);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      reg_info.kinds[eRegisterKindDWARF] = dwarf_num;
    } else {
      reg_info.kinds[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    }

    // Set EH_FRAME register number (same as DWARF for now)
    reg_info.kinds[eRegisterKindEHFrame] = reg_info.kinds[eRegisterKindDWARF];

    // Set generic register kind
    reg_info.kinds[eRegisterKindGeneric] = LLDB_INVALID_REGNUM;

    // Check if this is the PC register
    if (reg_id.handle == pc_register_id.handle) {
      reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
    }

    // Add this register indices belong to its register classes
    for (size_t j = 0; j < register_class_count; ++j) {
      amd_dbgapi_register_class_state_t register_class_state;
      status = amd_dbgapi_register_is_in_register_class(
          register_class_ids[j], reg_id, &register_class_state);
      if (status == AMD_DBGAPI_STATUS_SUCCESS &&
          register_class_state == AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER) {
        register_class_to_lldb_regnums[register_class_ids[j]].push_back(i);
      }
    }
  }

  // Create register sets from register classes
  g_reg_sets.clear();

  for (size_t i = 0; i < register_class_count; ++i) {
    auto class_id = register_class_ids[i];
    auto name_it = register_class_names.find(class_id);
    if (name_it == register_class_names.end()) {
      continue; // Skip if no name found
    }

    auto regnums_it = register_class_to_lldb_regnums.find(class_id);
    if (regnums_it == register_class_to_lldb_regnums.end() ||
        regnums_it->second.empty()) {
      continue; // Skip if no registers in this class
    }

    // Create a new register set for this class
    RegisterSet reg_set;
    reg_set.name = strdup(name_it->second.c_str());

    // Create short name from the full name (use first word or first few chars)
    std::string short_name = name_it->second;
    size_t space_pos = short_name.find(' ');
    if (space_pos != std::string::npos) {
      short_name = short_name.substr(0, space_pos);
    } else if (short_name.length() > 3) {
      short_name = short_name.substr(0, 3);
    }
    std::transform(short_name.begin(), short_name.end(), short_name.begin(),
                   ::tolower);
    reg_set.short_name = strdup(short_name.c_str());

    // Get register numbers for this class
    const auto &regnums = regnums_it->second;

    // Store register numbers in a static container to ensure they live
    // for the duration of the program
    static std::vector<std::vector<uint32_t>> all_reg_nums;
    all_reg_nums.push_back(regnums);

    // Point the RegisterSet's registers field to the data in our static vector
    reg_set.registers = all_reg_nums.back().data();
    reg_set.num_registers = all_reg_nums.back().size();
    g_reg_sets.push_back(reg_set);
  }
  return true;
}

RegisterContextAMDGPU::RegisterContextAMDGPU(
    NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread) {
  InitRegisterInfos();
  InitRegisters();
  // Only doing this for the Mock GPU class, don't do this in real GPU classes.
  // ReadRegs();
}

void RegisterContextAMDGPU::InitRegisters() {
  m_regs.data.resize(g_register_buffer_size);
  m_regs_valid.resize(kNumRegs, false);
}

void RegisterContextAMDGPU::InvalidateAllRegisters() {
  // Do what ever book keeping we need to do to indicate that all register
  // values are now invalid.
  for (uint32_t i = 0; i < kNumRegs; ++i)
    m_regs_valid[i] = false;
}

Status RegisterContextAMDGPU::ReadReg(const RegisterInfo *reg_info) {
  Status error;
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  assert(lldb_reg_num < kNumRegs);
  auto amd_reg_id = g_lldb_num_to_amd_reg_id[lldb_reg_num];
  ThreadAMDGPU *thread = (ThreadAMDGPU *)&m_thread;
  auto wave_id = thread->GetWaveId();
  if (!wave_id) {
    // Swallow the error because so that we are returning the dummy register
    // vlaues.
    return error;
  }
  amd_dbgapi_register_exists_t exists;
  amd_dbgapi_status_t amd_status =
      amd_dbgapi_wave_register_exists(wave_id.value(), amd_reg_id, &exists);
  if (amd_status != AMD_DBGAPI_STATUS_SUCCESS) {
    error.FromErrorStringWithFormat(
        "Failed to check register %s existence  due to error %d",
        reg_info->name, amd_status);
    return error;
  }
  if (exists != AMD_DBGAPI_REGISTER_PRESENT) {
    error = Status::FromErrorStringWithFormat(
        "Failed to read register %s due to register not present",
        reg_info->name);
    return error;
  }

  amd_status = amd_dbgapi_prefetch_register(wave_id.value(), amd_reg_id,
                                            m_regs.data.size() - lldb_reg_num);
  if (amd_status != AMD_DBGAPI_STATUS_SUCCESS) {
    error = Status::FromErrorStringWithFormat(
        "Failed to prefetch register %s due to error %d", reg_info->name,
        amd_status);
    return error;
  }

  // Read the register value
  amd_status = amd_dbgapi_read_register(
      wave_id.value(), amd_reg_id, 0, reg_info->byte_size,
      m_regs.data.data() + reg_info->byte_offset);
  if (amd_status != AMD_DBGAPI_STATUS_SUCCESS) {
    error = Status::FromErrorStringWithFormat(
        "Failed to read register %s due to error %d", reg_info->name,
        amd_status);
    return error;
  }
  m_regs_valid[lldb_reg_num] = true;
  return error;
}

Status RegisterContextAMDGPU::ReadRegs() {
  ThreadAMDGPU *thread = (ThreadAMDGPU *)&m_thread;
  if (thread != nullptr) {
    auto wave_id = thread->GetWaveId();
    bool wave_stopped = wave_id.has_value();
    if (wave_stopped) {
      for (uint32_t i = 0; i < g_reg_infos.size(); ++i) {
        Status error = ReadReg(&g_reg_infos[i]);
        if (error.Fail())
          return error;
      }
    }
  } else {
    // Fill all registers with unique values.
    for (uint32_t i = 0; i < g_reg_infos.size(); ++i) {
      memcpy(m_regs.data.data() + g_reg_infos[i].byte_offset, &i, sizeof(i));
    }
  }
  return Status();
}

uint32_t RegisterContextAMDGPU::GetRegisterSetCount() const {
  return g_reg_sets.size();
}

uint32_t RegisterContextAMDGPU::GetRegisterCount() const { return kNumRegs; }

uint32_t RegisterContextAMDGPU::GetUserRegisterCount() const {
  return GetRegisterCount();
}

const RegisterInfo *
RegisterContextAMDGPU::GetRegisterInfoAtIndex(uint32_t reg) const {
  if (reg < kNumRegs)
    return &g_reg_infos[reg];
  return nullptr;
}

const RegisterSet *
RegisterContextAMDGPU::GetRegisterSet(uint32_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets[set_index];
  return nullptr;
}

Status RegisterContextAMDGPU::ReadRegister(const RegisterInfo *reg_info,
                                           RegisterValue &reg_value) {
  Status error;
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  if (!m_regs_valid[lldb_reg_num]) {
    error = ReadReg(reg_info);
  }
  if (error.Fail())
    return error;
  reg_value.SetBytes(m_regs.data.data() + reg_info->byte_offset,
                     reg_info->byte_size, lldb::eByteOrderLittle);
  return Status();
}

Status RegisterContextAMDGPU::WriteRegister(const RegisterInfo *reg_info,
                                            const RegisterValue &reg_value) {
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  const void *new_value = reg_value.GetBytes();
  memcpy(m_regs.data.data() + reg_info->byte_offset, new_value,
         reg_info->byte_size);
  m_regs_valid[lldb_reg_num] = true;
  return Status();
}

Status RegisterContextAMDGPU::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  ReadRegs(); // Read all registers first
  const size_t regs_byte_size = m_regs.data.size();
  data_sp.reset(new DataBufferHeap(regs_byte_size, 0));
  uint8_t *dst = data_sp->GetBytes();
  memcpy(dst, &m_regs.data[0], regs_byte_size);
  return Status();
}

Status RegisterContextAMDGPU::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  const size_t regs_byte_size = m_regs.data.size();

  if (!data_sp) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextAMDGPU::%s invalid data_sp provided", __FUNCTION__);
  }

  if (data_sp->GetByteSize() != regs_byte_size) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextAMDGPU::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, regs_byte_size, data_sp->GetByteSize());
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextAMDGPU::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
  }
  memcpy(&m_regs.data[0], src, regs_byte_size);
  return Status();
}

std::vector<uint32_t>
RegisterContextAMDGPU::GetExpeditedRegisters(ExpeditedRegs expType) const {
  static std::vector<uint32_t> g_expedited_regs;
  if (g_expedited_regs.empty()) {
    // TODO: is this the correct way to do this?
    // g_expedited_regs.push_back(LLDB_REGNUM_GENERIC_PC);
    // g_expedited_regs.push_back(LLDB_SP);
    // g_expedited_regs.push_back(LLDB_FP);
  }
  return g_expedited_regs;
}
