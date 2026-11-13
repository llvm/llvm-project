//===-- RegisterContextDarwin_arm.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextDarwin_arm.h"
#include "RegisterContextDarwinConstants.h"

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/Compiler.h"

#include "Plugins/Process/Utility/InstructionUtils.h"

#include <memory>


#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

enum {
  gpr_r0 = 0,
  gpr_r1,
  gpr_r2,
  gpr_r3,
  gpr_r4,
  gpr_r5,
  gpr_r6,
  gpr_r7,
  gpr_r8,
  gpr_r9,
  gpr_r10,
  gpr_r11,
  gpr_r12,
  gpr_r13,
  gpr_sp = gpr_r13,
  gpr_r14,
  gpr_lr = gpr_r14,
  gpr_r15,
  gpr_pc = gpr_r15,
  gpr_cpsr,

  fpu_s0,
  fpu_s1,
  fpu_s2,
  fpu_s3,
  fpu_s4,
  fpu_s5,
  fpu_s6,
  fpu_s7,
  fpu_s8,
  fpu_s9,
  fpu_s10,
  fpu_s11,
  fpu_s12,
  fpu_s13,
  fpu_s14,
  fpu_s15,
  fpu_s16,
  fpu_s17,
  fpu_s18,
  fpu_s19,
  fpu_s20,
  fpu_s21,
  fpu_s22,
  fpu_s23,
  fpu_s24,
  fpu_s25,
  fpu_s26,
  fpu_s27,
  fpu_s28,
  fpu_s29,
  fpu_s30,
  fpu_s31,
  fpu_fpscr,

  exc_exception,
  exc_fsr,
  exc_far,

  dbg_bvr0,
  dbg_bvr1,
  dbg_bvr2,
  dbg_bvr3,
  dbg_bvr4,
  dbg_bvr5,
  dbg_bvr6,
  dbg_bvr7,
  dbg_bvr8,
  dbg_bvr9,
  dbg_bvr10,
  dbg_bvr11,
  dbg_bvr12,
  dbg_bvr13,
  dbg_bvr14,
  dbg_bvr15,

  dbg_bcr0,
  dbg_bcr1,
  dbg_bcr2,
  dbg_bcr3,
  dbg_bcr4,
  dbg_bcr5,
  dbg_bcr6,
  dbg_bcr7,
  dbg_bcr8,
  dbg_bcr9,
  dbg_bcr10,
  dbg_bcr11,
  dbg_bcr12,
  dbg_bcr13,
  dbg_bcr14,
  dbg_bcr15,

  dbg_wvr0,
  dbg_wvr1,
  dbg_wvr2,
  dbg_wvr3,
  dbg_wvr4,
  dbg_wvr5,
  dbg_wvr6,
  dbg_wvr7,
  dbg_wvr8,
  dbg_wvr9,
  dbg_wvr10,
  dbg_wvr11,
  dbg_wvr12,
  dbg_wvr13,
  dbg_wvr14,
  dbg_wvr15,

  dbg_wcr0,
  dbg_wcr1,
  dbg_wcr2,
  dbg_wcr3,
  dbg_wcr4,
  dbg_wcr5,
  dbg_wcr6,
  dbg_wcr7,
  dbg_wcr8,
  dbg_wcr9,
  dbg_wcr10,
  dbg_wcr11,
  dbg_wcr12,
  dbg_wcr13,
  dbg_wcr14,
  dbg_wcr15,

  k_num_registers
};

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_arm::GPR) +                                    \
   sizeof(RegisterContextDarwin_arm::FPU) +                                    \
   sizeof(RegisterContextDarwin_arm::EXC))

// General purpose registers
static uint32_t g_gpr_regnums[] = {
    gpr_r0, gpr_r1,  gpr_r2,  gpr_r3,  gpr_r4, gpr_r5, gpr_r6, gpr_r7,  gpr_r8,
    gpr_r9, gpr_r10, gpr_r11, gpr_r12, gpr_sp, gpr_lr, gpr_pc, gpr_cpsr};

// Floating point registers
static uint32_t g_fpu_regnums[] = {
    fpu_s0,  fpu_s1,  fpu_s2,  fpu_s3,  fpu_s4,    fpu_s5,  fpu_s6,
    fpu_s7,  fpu_s8,  fpu_s9,  fpu_s10, fpu_s11,   fpu_s12, fpu_s13,
    fpu_s14, fpu_s15, fpu_s16, fpu_s17, fpu_s18,   fpu_s19, fpu_s20,
    fpu_s21, fpu_s22, fpu_s23, fpu_s24, fpu_s25,   fpu_s26, fpu_s27,
    fpu_s28, fpu_s29, fpu_s30, fpu_s31, fpu_fpscr,
};

// Exception registers

static uint32_t g_exc_regnums[] = {
    exc_exception, exc_fsr, exc_far,
};

static std::shared_ptr<const RegisterDescription> CreateContextDescription();

RegisterContextDarwin_arm::RegisterContextDarwin_arm(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpu(), exc() {
  SetRegisterDescription(CreateContextDescription());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpu_errs[i] = -1;
    exc_errs[i] = -1;
  }
}

RegisterContextDarwin_arm::~RegisterContextDarwin_arm() = default;

void RegisterContextDarwin_arm::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

// Number of registers in each register set
const size_t k_num_gpr_registers = std::size(g_gpr_regnums);
const size_t k_num_fpu_registers = std::size(g_fpu_regnums);
const size_t k_num_exc_registers = std::size(g_exc_regnums);

// Register set definitions. The first definitions at register set index of
// zero is for all registers, followed by other registers sets. The register
// information for the all register set need not be filled in.
static const RegisterGroup g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums,
                             g_gpr_regnums + k_num_gpr_registers)},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(g_fpu_regnums,
                             g_fpu_regnums + k_num_fpu_registers)},
    {"Exception State Registers", "exc",
     std::vector<RegisterID>(g_exc_regnums,
                             g_exc_regnums + k_num_exc_registers)}};

static std::shared_ptr<const RegisterDescription> CreateContextDescription() {
  static const auto description = [] {
    auto cpu = RegisterInfo_arm::CreateDescription();
    std::vector<RegisterDescriptor> registers;
    // Mach selectors are an explicit projection of the CPU catalog. These
    // contexts expose complete independent values, without the catalog's
    // unexposed subregister aliases.
    auto append = [&](uint32_t selector, const RegisterDescriptor &definition)
        -> RegisterDescriptor & {
      auto reg = definition;
      reg.id = selector;
      reg.numbers[eRegisterKindLLDB] = selector;
      reg.numbers[eRegisterKindProcessPlugin] = LLDB_INVALID_REGNUM;
      reg.value_registers.clear();
      reg.invalidate_registers.clear();
      reg.value_byte_offset = 0;
      reg.zero_extend_parent = LLDB_INVALID_REGNUM;
      registers.push_back(std::move(reg));
      return registers.back();
    };
    for (uint32_t i = 0; i <= gpr_cpsr; ++i) {
      auto &reg = append(i, *cpu->GetRegisterAtIndex(i));
      // Mach uses r7 as its frame pointer and does not expose argument roles.
      reg.numbers[eRegisterKindGeneric] =
          i == gpr_r7 ? LLDB_REGNUM_GENERIC_FP
          : i == gpr_sp ? LLDB_REGNUM_GENERIC_SP
          : i == gpr_lr ? LLDB_REGNUM_GENERIC_RA
          : i == gpr_pc ? LLDB_REGNUM_GENERIC_PC
          : i == gpr_cpsr ? LLDB_REGNUM_GENERIC_FLAGS
                          : LLDB_INVALID_REGNUM;
    }
    for (uint32_t i = 0; i < 32; ++i)
      append(fpu_s0 + i, *cpu->Find("s" + std::to_string(i)));
    append(fpu_fpscr, *cpu->Find("fpscr"));
    for (auto entry : {std::make_pair(exc_exception, "exception"),
                       std::make_pair(exc_fsr, "fsr"),
                       std::make_pair(exc_far, "far")})
      registers.emplace_back(
          entry.first, entry.second, nullptr, 4, eEncodingUint, eFormatHex,
          RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       static_cast<uint32_t>(entry.first)});
    for (auto entry : {std::make_pair(dbg_bvr0, "bvr"),
                       std::make_pair(dbg_bcr0, "bcr"),
                       std::make_pair(dbg_wvr0, "wvr"),
                       std::make_pair(dbg_wcr0, "wcr")})
      for (uint32_t i = 0; i < 16; ++i)
        registers.emplace_back(
            entry.first + i, entry.second, nullptr, 4, eEncodingUint, eFormatHex,
            RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM,
                                         LLDB_INVALID_REGNUM,
                                         LLDB_INVALID_REGNUM,
                                         LLDB_INVALID_REGNUM,
                                         LLDB_INVALID_REGNUM});
    return std::make_shared<RegisterDescription>(
        llvm::cantFail(RegisterDescription::Create(
            std::move(registers),
            {std::begin(g_reg_sets), std::end(g_reg_sets)},
            {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
  }();
  return description;
}

// Register information definitions for 32 bit i386.
int RegisterContextDarwin_arm::GetSetForNativeRegNum(int reg) {
  if (reg < fpu_s0)
    return GPRRegSet;
  else if (reg < exc_exception)
    return FPURegSet;
  else if (reg < k_num_registers)
    return EXCRegSet;
  return -1;
}

int RegisterContextDarwin_arm::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(GPRRegSet, Read);
}

int RegisterContextDarwin_arm::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
  }
  return GetError(FPURegSet, Read);
}

int RegisterContextDarwin_arm::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(EXCRegSet, Read);
}

int RegisterContextDarwin_arm::ReadDBG(bool force) {
  int set = DBGRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadDBG(GetThreadID(), set, dbg));
  }
  return GetError(DBGRegSet, Read);
}

int RegisterContextDarwin_arm::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(GPRRegSet, Write);
}

int RegisterContextDarwin_arm::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpu));
  SetError(set, Read, -1);
  return GetError(FPURegSet, Write);
}

int RegisterContextDarwin_arm::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(EXCRegSet, Write);
}

int RegisterContextDarwin_arm::WriteDBG() {
  int set = DBGRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteDBG(GetThreadID(), set, dbg));
  SetError(set, Read, -1);
  return GetError(DBGRegSet, Write);
}

int RegisterContextDarwin_arm::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case GPRAltRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  case DBGRegSet:
    return ReadDBG(force);
  default:
    break;
  }
  return KERN_INVALID_ARGUMENT;
}

int RegisterContextDarwin_arm::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case GPRAltRegSet:
      return WriteGPR();
    case FPURegSet:
      return WriteFPU();
    case EXCRegSet:
      return WriteEXC();
    case DBGRegSet:
      return WriteDBG();
    default:
      break;
    }
  }
  return KERN_INVALID_ARGUMENT;
}

void RegisterContextDarwin_arm::LogDBGRegisters(Log *log, const DBG &dbg) {
  if (log) {
    for (uint32_t i = 0; i < 16; i++)
      LLDB_LOGF(log,
                "BVR%-2u/BCR%-2u = { 0x%8.8x, 0x%8.8x } WVR%-2u/WCR%-2u = { "
                "0x%8.8x, 0x%8.8x }",
                i, i, dbg.bvr[i], dbg.bcr[i], i, i, dbg.wvr[i], dbg.wcr[i]);
  }
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_arm::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  int set = RegisterContextDarwin_arm::GetSetForNativeRegNum(reg);

  if (set == -1)
    return llvm::createStringError("register value is unavailable");

  if (ReadRegisterSet(set, false) != KERN_SUCCESS)
    return llvm::createStringError("register value is unavailable");

  switch (reg) {
  case gpr_r0:
  case gpr_r1:
  case gpr_r2:
  case gpr_r3:
  case gpr_r4:
  case gpr_r5:
  case gpr_r6:
  case gpr_r7:
  case gpr_r8:
  case gpr_r9:
  case gpr_r10:
  case gpr_r11:
  case gpr_r12:
  case gpr_sp:
  case gpr_lr:
  case gpr_pc:
    value.SetUInt32(gpr.r[reg - gpr_r0]);
    break;
  case gpr_cpsr:
    value.SetUInt32(gpr.cpsr);
    break;
  case fpu_s0:
  case fpu_s1:
  case fpu_s2:
  case fpu_s3:
  case fpu_s4:
  case fpu_s5:
  case fpu_s6:
  case fpu_s7:
  case fpu_s8:
  case fpu_s9:
  case fpu_s10:
  case fpu_s11:
  case fpu_s12:
  case fpu_s13:
  case fpu_s14:
  case fpu_s15:
  case fpu_s16:
  case fpu_s17:
  case fpu_s18:
  case fpu_s19:
  case fpu_s20:
  case fpu_s21:
  case fpu_s22:
  case fpu_s23:
  case fpu_s24:
  case fpu_s25:
  case fpu_s26:
  case fpu_s27:
  case fpu_s28:
  case fpu_s29:
  case fpu_s30:
  case fpu_s31:
    value.SetUInt32(fpu.floats.s[reg], RegisterValue::eTypeFloat);
    break;

  case fpu_fpscr:
    value.SetUInt32(fpu.fpscr);
    break;

  case exc_exception:
    value.SetUInt32(exc.exception);
    break;
  case exc_fsr:
    value.SetUInt32(exc.fsr);
    break;
  case exc_far:
    value.SetUInt32(exc.far);
    break;

  default:
    value.SetValueToInvalid();
    return llvm::createStringError("register value is unavailable");
  }
  return value;
}

llvm::Error
RegisterContextDarwin_arm::WriteRegister(RegisterID id,
                                              const RegisterValue &value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  int set = GetSetForNativeRegNum(reg);

  if (set == -1)
    return llvm::createStringError("register write is unavailable");

  if (ReadRegisterSet(set, false) != KERN_SUCCESS)
    return llvm::createStringError("register write is unavailable");

  switch (reg) {
  case gpr_r0:
  case gpr_r1:
  case gpr_r2:
  case gpr_r3:
  case gpr_r4:
  case gpr_r5:
  case gpr_r6:
  case gpr_r7:
  case gpr_r8:
  case gpr_r9:
  case gpr_r10:
  case gpr_r11:
  case gpr_r12:
  case gpr_sp:
  case gpr_lr:
  case gpr_pc:
  case gpr_cpsr:
    gpr.r[reg - gpr_r0] = value.GetAsUInt32();
    break;

  case fpu_s0:
  case fpu_s1:
  case fpu_s2:
  case fpu_s3:
  case fpu_s4:
  case fpu_s5:
  case fpu_s6:
  case fpu_s7:
  case fpu_s8:
  case fpu_s9:
  case fpu_s10:
  case fpu_s11:
  case fpu_s12:
  case fpu_s13:
  case fpu_s14:
  case fpu_s15:
  case fpu_s16:
  case fpu_s17:
  case fpu_s18:
  case fpu_s19:
  case fpu_s20:
  case fpu_s21:
  case fpu_s22:
  case fpu_s23:
  case fpu_s24:
  case fpu_s25:
  case fpu_s26:
  case fpu_s27:
  case fpu_s28:
  case fpu_s29:
  case fpu_s30:
  case fpu_s31:
    fpu.floats.s[reg] = value.GetAsUInt32();
    break;

  case fpu_fpscr:
    fpu.fpscr = value.GetAsUInt32();
    break;

  case exc_exception:
    exc.exception = value.GetAsUInt32();
    break;
  case exc_fsr:
    exc.fsr = value.GetAsUInt32();
    break;
  case exc_far:
    exc.far = value.GetAsUInt32();
    break;

  default:
    return llvm::createStringError("register write is unavailable");
  }
  {
    if (!(WriteRegisterSet(set) == KERN_SUCCESS))
      return llvm::createStringError("register write failed");
    return llvm::Error::success();
  }
}

bool RegisterContextDarwin_arm::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (data_sp && ReadGPR(false) == KERN_SUCCESS &&
      ReadFPU(false) == KERN_SUCCESS && ReadEXC(false) == KERN_SUCCESS) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpu, sizeof(fpu));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_arm::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpu, src, sizeof(fpu));
    src += sizeof(gpr);

    ::memcpy(&exc, src, sizeof(exc));
    uint32_t success_count = 0;
    if (WriteGPR() == KERN_SUCCESS)
      ++success_count;
    if (WriteFPU() == KERN_SUCCESS)
      ++success_count;
    if (WriteEXC() == KERN_SUCCESS)
      ++success_count;
    return success_count == 3;
  }
  return false;
}

uint32_t RegisterContextDarwin_arm::NumSupportedHardwareBreakpoints() {
#if defined(__APPLE__) && defined(__arm__)
  // Set the init value to something that will let us know that we need to
  // autodetect how many breakpoints are supported dynamically...
  static uint32_t g_num_supported_hw_breakpoints = UINT32_MAX;
  if (g_num_supported_hw_breakpoints == UINT32_MAX) {
    // Set this to zero in case we can't tell if there are any HW breakpoints
    g_num_supported_hw_breakpoints = 0;

    uint32_t register_DBGDIDR;

    asm("mrc p14, 0, %0, c0, c0, 0" : "=r"(register_DBGDIDR));
    g_num_supported_hw_breakpoints = Bits32(register_DBGDIDR, 27, 24);
    // Zero is reserved for the BRP count, so don't increment it if it is zero
    if (g_num_supported_hw_breakpoints > 0)
      g_num_supported_hw_breakpoints++;
  }
  return g_num_supported_hw_breakpoints;
#else
  // TODO: figure out remote case here!
  return 6;
#endif
}

uint32_t RegisterContextDarwin_arm::SetHardwareBreakpoint(lldb::addr_t addr,
                                                          size_t size) {
  // Make sure our address isn't bogus
  if (addr & 1)
    return LLDB_INVALID_INDEX32;

  int kret = ReadDBG(false);

  if (kret == KERN_SUCCESS) {
    const uint32_t num_hw_breakpoints = NumSupportedHardwareBreakpoints();
    uint32_t i;
    for (i = 0; i < num_hw_breakpoints; ++i) {
      if ((dbg.bcr[i] & BCR_ENABLE) == 0)
        break; // We found an available hw breakpoint slot (in i)
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_breakpoints) {
      // Make sure bits 1:0 are clear in our address
      dbg.bvr[i] = addr & ~((lldb::addr_t)3);

      if (size == 2 || addr & 2) {
        uint32_t byte_addr_select = (addr & 2) ? BAS_IMVA_2_3 : BAS_IMVA_0_1;

        // We have a thumb breakpoint
        // We have an ARM breakpoint
        dbg.bcr[i] = BCR_M_IMVA_MATCH | // Stop on address match
                     byte_addr_select | // Set the correct byte address select
                                        // so we only trigger on the correct
                                        // opcode
                     S_USER |    // Which modes should this breakpoint stop in?
                     BCR_ENABLE; // Enable this hardware breakpoint
                                 //                if (log) log->Printf
        //                ("RegisterContextDarwin_arm::EnableHardwareBreakpoint(
        //                addr = %8.8p, size = %u ) - BVR%u/BCR%u = 0x%8.8x /
        //                0x%8.8x (Thumb)",
        //                        addr,
        //                        size,
        //                        i,
        //                        i,
        //                        dbg.bvr[i],
        //                        dbg.bcr[i]);
      } else if (size == 4) {
        // We have an ARM breakpoint
        dbg.bcr[i] =
            BCR_M_IMVA_MATCH | // Stop on address match
            BAS_IMVA_ALL | // Stop on any of the four bytes following the IMVA
            S_USER |       // Which modes should this breakpoint stop in?
            BCR_ENABLE;    // Enable this hardware breakpoint
                           //                if (log) log->Printf
        //                ("RegisterContextDarwin_arm::EnableHardwareBreakpoint(
        //                addr = %8.8p, size = %u ) - BVR%u/BCR%u = 0x%8.8x /
        //                0x%8.8x (ARM)",
        //                        addr,
        //                        size,
        //                        i,
        //                        i,
        //                        dbg.bvr[i],
        //                        dbg.bcr[i]);
      }

      kret = WriteDBG();
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm::EnableHardwareBreakpoint()
      //            WriteDBG() => 0x%8.8x.", kret);

      if (kret == KERN_SUCCESS)
        return i;
    }
    //        else
    //        {
    //            if (log) log->Printf
    //            ("RegisterContextDarwin_arm::EnableHardwareBreakpoint(addr =
    //            %8.8p, size = %u) => all hardware breakpoint resources are
    //            being used.", addr, size);
    //        }
  }

  return LLDB_INVALID_INDEX32;
}

bool RegisterContextDarwin_arm::ClearHardwareBreakpoint(uint32_t hw_index) {
  int kret = ReadDBG(false);

  const uint32_t num_hw_points = NumSupportedHardwareBreakpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      dbg.bcr[hw_index] = 0;
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm::SetHardwareBreakpoint( %u ) -
      //            BVR%u = 0x%8.8x  BCR%u = 0x%8.8x",
      //                    hw_index,
      //                    hw_index,
      //                    dbg.bvr[hw_index],
      //                    hw_index,
      //                    dbg.bcr[hw_index]);

      kret = WriteDBG();

      if (kret == KERN_SUCCESS)
        return true;
    }
  }
  return false;
}

uint32_t RegisterContextDarwin_arm::NumSupportedHardwareWatchpoints() {
#if defined(__APPLE__) && defined(__arm__)
  // Set the init value to something that will let us know that we need to
  // autodetect how many watchpoints are supported dynamically...
  static uint32_t g_num_supported_hw_watchpoints = UINT32_MAX;
  if (g_num_supported_hw_watchpoints == UINT32_MAX) {
    // Set this to zero in case we can't tell if there are any HW breakpoints
    g_num_supported_hw_watchpoints = 0;

    uint32_t register_DBGDIDR;
    asm("mrc p14, 0, %0, c0, c0, 0" : "=r"(register_DBGDIDR));
    g_num_supported_hw_watchpoints = Bits32(register_DBGDIDR, 31, 28) + 1;
  }
  return g_num_supported_hw_watchpoints;
#else
  // TODO: figure out remote case here!
  return 2;
#endif
}

uint32_t RegisterContextDarwin_arm::SetHardwareWatchpoint(lldb::addr_t addr,
                                                          size_t size,
                                                          bool read,
                                                          bool write) {
  const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

  // Can't watch zero bytes
  if (size == 0)
    return LLDB_INVALID_INDEX32;

  // We must watch for either read or write
  if (!read && !write)
    return LLDB_INVALID_INDEX32;

  // Can't watch more than 4 bytes per WVR/WCR pair
  if (size > 4)
    return LLDB_INVALID_INDEX32;

  // We can only watch up to four bytes that follow a 4 byte aligned address
  // per watchpoint register pair. Since we have at most so we can only watch
  // until the next 4 byte boundary and we need to make sure we can properly
  // encode this.
  uint32_t addr_word_offset = addr % 4;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm::EnableHardwareWatchpoint() -
  //    addr_word_offset = 0x%8.8x", addr_word_offset);

  uint32_t byte_mask = ((1u << size) - 1u) << addr_word_offset;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm::EnableHardwareWatchpoint() - byte_mask =
  //    0x%8.8x", byte_mask);
  if (byte_mask > 0xfu)
    return LLDB_INVALID_INDEX32;

  // Read the debug state
  int kret = ReadDBG(false);

  if (kret == KERN_SUCCESS) {
    // Check to make sure we have the needed hardware support
    uint32_t i = 0;

    for (i = 0; i < num_hw_watchpoints; ++i) {
      if ((dbg.wcr[i] & WCR_ENABLE) == 0)
        break; // We found an available hw breakpoint slot (in i)
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_watchpoints) {
      // Make the byte_mask into a valid Byte Address Select mask
      uint32_t byte_address_select = byte_mask << 5;
      // Make sure bits 1:0 are clear in our address
      dbg.wvr[i] = addr & ~((lldb::addr_t)3);
      dbg.wcr[i] = byte_address_select |     // Which bytes that follow the IMVA
                                             // that we will watch
                   S_USER |                  // Stop only in user mode
                   (read ? WCR_LOAD : 0) |   // Stop on read access?
                   (write ? WCR_STORE : 0) | // Stop on write access?
                   WCR_ENABLE;               // Enable this watchpoint;

      kret = WriteDBG();
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm::EnableHardwareWatchpoint()
      //            WriteDBG() => 0x%8.8x.", kret);

      if (kret == KERN_SUCCESS)
        return i;
    } else {
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm::EnableHardwareWatchpoint(): All
      //            hardware resources (%u) are in use.", num_hw_watchpoints);
    }
  }
  return LLDB_INVALID_INDEX32;
}

bool RegisterContextDarwin_arm::ClearHardwareWatchpoint(uint32_t hw_index) {
  int kret = ReadDBG(false);

  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      dbg.wcr[hw_index] = 0;
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm::ClearHardwareWatchpoint( %u ) -
      //            WVR%u = 0x%8.8x  WCR%u = 0x%8.8x",
      //                    hw_index,
      //                    hw_index,
      //                    dbg.wvr[hw_index],
      //                    hw_index,
      //                    dbg.wcr[hw_index]);

      kret = WriteDBG();

      if (kret == KERN_SUCCESS)
        return true;
    }
  }
  return false;
}
