//===-- RegisterContextMockGPU.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMockGPU.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

/// LLDB register numbers must start at 0 and be contiguous with no gaps.
enum LLDBRegNum : uint32_t {
  LLDB_R0 = 0,
  LLDB_R1,
  LLDB_R2,
  LLDB_R3,
  LLDB_R4,
  LLDB_R5,
  LLDB_R6,
  LLDB_R7,
  LLDB_SP,
  LLDB_FP,
  LLDB_PC,
  LLDB_Flags,
  LLDB_V0,
  LLDB_V1,
  LLDB_V2,
  LLDB_V3,
  LLDB_V4,
  LLDB_V5,
  LLDB_V6,
  LLDB_V7,
  kNumRegs
};

/// DWARF register numbers should match the register numbers that the compiler
/// uses in the DWARF debug info. They can be any number and do not need to
/// be in increasing order or consective, there can be gaps. The compiler has
/// dedicated register numbers for any DWARF that references registers, like
/// location expressions and .debug_frame unwind info.
enum DWARFRegNum : uint32_t {
  DWARF_R0 = 128,
  DWARF_R1,
  DWARF_R2,
  DWARF_R3,
  DWARF_R4,
  DWARF_R5,
  DWARF_R6,
  DWARF_R7,
  DWARF_SP,
  DWARF_FP,
  DWARF_PC,
  DWARF_Flags = LLDB_INVALID_REGNUM, // This register does not exist in DWARF
  DWARF_V0 = 256,
  DWARF_V1,
  DWARF_V2,
  DWARF_V3,
  DWARF_V4,
  DWARF_V5,
  DWARF_V6,
  DWARF_V7,
};

/// Compiler registers should match the register numbers that the compiler
/// uses in runtime information. They can be any number and do not need to
/// be in increasing order or consective, there can be gaps. The compiler has
/// dedicated register numbers for any runtime information that references
/// registers, like .eh_frame unwind info. Many times these numbers match the
/// DWARF register numbers, but not always.
enum CompilerRegNum : uint32_t {
  EH_FRAME_R0 = 1000,
  EH_FRAME_R1,
  EH_FRAME_R2,
  EH_FRAME_R3,
  EH_FRAME_R4,
  EH_FRAME_R5,
  EH_FRAME_R6,
  EH_FRAME_R7,
  EH_FRAME_SP,
  EH_FRAME_FP,
  EH_FRAME_PC,
  EH_FRAME_Flags = LLDB_INVALID_REGNUM, // Not accessed by runtime info.
  EH_FRAME_V0 = 2000,
  EH_FRAME_V1,
  EH_FRAME_V2,
  EH_FRAME_V3,
  EH_FRAME_V4,
  EH_FRAME_V5,
  EH_FRAME_V6,
  EH_FRAME_V7,
};

uint32_t g_gpr_regnums[] = {LLDB_R0, LLDB_R1, LLDB_R2, LLDB_R3,
                            LLDB_R4, LLDB_R5, LLDB_R6, LLDB_R7,
                            LLDB_SP, LLDB_FP, LLDB_PC, LLDB_Flags};
uint32_t g_vec_regnums[] = {LLDB_V0, LLDB_V1, LLDB_V2, LLDB_V3,
                            LLDB_V4, LLDB_V5, LLDB_V6, LLDB_V7};

static const RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     sizeof(g_gpr_regnums) / sizeof(g_gpr_regnums[0]), g_gpr_regnums},
    {"Vector Registers", "vector",
     sizeof(g_vec_regnums) / sizeof(g_vec_regnums[0]), g_vec_regnums},
};

/// Define all of the information about all registers. The register info structs
/// are accessed by the LLDB register numbers, which are defined above.
static const RegisterInfo g_reg_infos[LLDBRegNum::kNumRegs] = {
    {
        "R0",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        0,             // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R0, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R0,    // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_ARG1, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R0, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R0  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R1",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        8,             // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R1, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R1,    // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_ARG2, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R1, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R1  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R2",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        16,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R2, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R2,    // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_ARG3, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R2, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R2  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R3",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        24,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R3, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R3,    // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_ARG4, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R3, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R3  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R4",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        32,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R4,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R4,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R4, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R4  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R5",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        40,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R5,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R5,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R5, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R5  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R6",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        48,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R6,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R6,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R6, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R6  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "R7",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        56,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_R7,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R7,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_R7, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_R7  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "SP",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        64,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_SP,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_SP,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_SP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_SP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_SP  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "FP",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        72,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_FP,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_FP,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_FP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_FP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_FP  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "PC",          // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        80,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_PC,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_PC,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_PC, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_PC, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_PC  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "Flags",       // RegisterInfo::name
        nullptr,       // RegisterInfo::alt_name
        8,             // RegisterInfo::byte_size
        88,            // RegisterInfo::byte_offset
        eEncodingUint, // RegisterInfo::encoding
        eFormatHex,    // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_Flags, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_Flags,    // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_FLAGS, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_Flags, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_Flags  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V0",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        96,                    // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V0,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V0,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V0, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V0  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V1",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        104,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V1,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V1,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V1, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V1  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V2",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        112,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V2,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V2,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V2, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V2  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V3",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        120,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V3,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V3,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V3, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V3  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V4",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        128,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V4,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V4,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V4, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V4  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V5",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        136,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V5,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V5,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V5, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V5  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V6",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        144,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V6,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V6,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V6, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V6  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "V7",                  // RegisterInfo::name
        nullptr,               // RegisterInfo::alt_name
        8,                     // RegisterInfo::byte_size
        152,                   // RegisterInfo::byte_offset
        eEncodingVector,       // RegisterInfo::encoding
        eFormatVectorOfUInt32, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_V7,         // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_V7,            // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_V7, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_V7  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
};
RegisterContextMockGPU::RegisterContextMockGPU(
    NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread) {
  InitRegisters();
  // Only doing this for the Mock GPU class, don't do this in real GPU classes.
  ReadRegs();
}

void RegisterContextMockGPU::InitRegisters() {
  for (size_t i = 0; i < kNumRegs; ++i)
    m_regs.data[i] = 0;
  m_regs_valid.resize(kNumRegs, false);
}

void RegisterContextMockGPU::InvalidateAllRegisters() {
  // Do what ever book keeping we need to do to indicate that all register
  // values are now invalid.
  for (uint32_t i = 0; i < kNumRegs; ++i)
    m_regs_valid[i] = false;
}

Status RegisterContextMockGPU::ReadRegs() {
  // Fill all registers with unique values.
  for (uint32_t i = 0; i < kNumRegs; ++i) {
    m_regs_valid[i] = true;
    m_regs.data[i] = i;
  }
  return Status();
}

uint32_t RegisterContextMockGPU::GetRegisterSetCount() const {
  return sizeof(g_reg_sets) / sizeof(g_reg_sets[0]);
}

uint32_t RegisterContextMockGPU::GetRegisterCount() const { return kNumRegs; }

uint32_t RegisterContextMockGPU::GetUserRegisterCount() const {
  return GetRegisterCount();
}

const RegisterInfo *
RegisterContextMockGPU::GetRegisterInfoAtIndex(uint32_t reg) const {
  if (reg < kNumRegs)
    return &g_reg_infos[reg];
  return nullptr;
}

const RegisterSet *
RegisterContextMockGPU::GetRegisterSet(uint32_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets[set_index];
  return nullptr;
}

Status RegisterContextMockGPU::ReadRegister(const RegisterInfo *reg_info,
                                            RegisterValue &reg_value) {
  Status error;
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  if (!m_regs_valid[lldb_reg_num])
    error = ReadRegs();
  if (error.Fail())
    return error;
  reg_value.SetUInt64(m_regs.data[lldb_reg_num]);
  return Status();
}

Status RegisterContextMockGPU::WriteRegister(const RegisterInfo *reg_info,
                                             const RegisterValue &reg_value) {
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  bool success = false;
  uint64_t new_value = reg_value.GetAsUInt64(UINT64_MAX, &success);
  if (!success)
    return Status::FromErrorString("register write failed");
  m_regs.data[lldb_reg_num] = new_value;
  m_regs_valid[lldb_reg_num] = true;
  return Status();
}

Status RegisterContextMockGPU::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  ReadRegs(); // Read all registers first
  const size_t regs_byte_size = sizeof(m_regs);
  data_sp.reset(new DataBufferHeap(regs_byte_size, 0));
  uint8_t *dst = data_sp->GetBytes();
  memcpy(dst, &m_regs.data[0], regs_byte_size);
  return Status();
}

Status RegisterContextMockGPU::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  const size_t regs_byte_size = sizeof(m_regs);

  if (!data_sp) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextMockGPU::%s invalid data_sp provided", __FUNCTION__);
  }

  if (data_sp->GetByteSize() != regs_byte_size) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextMockGPU::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, regs_byte_size, data_sp->GetByteSize());
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextMockGPU::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
  }
  memcpy(&m_regs.data[0], src, regs_byte_size);
  return Status();
}

std::vector<uint32_t>
RegisterContextMockGPU::GetExpeditedRegisters(ExpeditedRegs expType) const {
  static std::vector<uint32_t> g_expedited_regs;
  if (g_expedited_regs.empty()) {
    g_expedited_regs.push_back(LLDB_PC);
    g_expedited_regs.push_back(LLDB_SP);
    g_expedited_regs.push_back(LLDB_FP);
  }
  return g_expedited_regs;
}
