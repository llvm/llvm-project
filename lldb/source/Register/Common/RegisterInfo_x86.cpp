//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_x86.h"
#include "lldb/Register/Arch/lldb-x86_64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-ehframe-register-enums.h"
#include "lldb/lldb-private-types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <set>

using namespace lldb_private;
using namespace lldb;
namespace {
namespace x86_register_info {
static uint32_t g_contained_eax[] = {registers::x86::id::eax};
static uint32_t g_contained_ebx[] = {registers::x86::id::ebx};
static uint32_t g_contained_ecx[] = {registers::x86::id::ecx};
static uint32_t g_contained_edx[] = {registers::x86::id::edx};
static uint32_t g_contained_edi[] = {registers::x86::id::edi};
static uint32_t g_contained_esi[] = {registers::x86::id::esi};
static uint32_t g_contained_ebp[] = {registers::x86::id::ebp};
static uint32_t g_contained_esp[] = {registers::x86::id::esp};
static uint32_t g_invalidate_eax[] = {
    registers::x86::id::eax, registers::x86::id::ax, registers::x86::id::ah,
    registers::x86::id::al};
static uint32_t g_invalidate_ebx[] = {
    registers::x86::id::ebx, registers::x86::id::bx, registers::x86::id::bh,
    registers::x86::id::bl};
static uint32_t g_invalidate_ecx[] = {
    registers::x86::id::ecx, registers::x86::id::cx, registers::x86::id::ch,
    registers::x86::id::cl};
static uint32_t g_invalidate_edx[] = {
    registers::x86::id::edx, registers::x86::id::dx, registers::x86::id::dh,
    registers::x86::id::dl};
static uint32_t g_invalidate_edi[] = {registers::x86::id::edi,
                                      registers::x86::id::di};
static uint32_t g_invalidate_esi[] = {registers::x86::id::esi,
                                      registers::x86::id::si};
static uint32_t g_invalidate_ebp[] = {registers::x86::id::ebp,
                                      registers::x86::id::bp};
static uint32_t g_invalidate_esp[] = {registers::x86::id::esp,
                                      registers::x86::id::sp};
static uint32_t g_contained_rax[] = {registers::x86_64::id::rax};
static uint32_t g_contained_rbx[] = {registers::x86_64::id::rbx};
static uint32_t g_contained_rcx[] = {registers::x86_64::id::rcx};
static uint32_t g_contained_rdx[] = {registers::x86_64::id::rdx};
static uint32_t g_contained_rdi[] = {registers::x86_64::id::rdi};
static uint32_t g_contained_rsi[] = {registers::x86_64::id::rsi};
static uint32_t g_contained_rbp[] = {registers::x86_64::id::rbp};
static uint32_t g_contained_rsp[] = {registers::x86_64::id::rsp};
static uint32_t g_contained_r8[] = {registers::x86_64::id::r8};
static uint32_t g_contained_r9[] = {registers::x86_64::id::r9};
static uint32_t g_contained_r10[] = {registers::x86_64::id::r10};
static uint32_t g_contained_r11[] = {registers::x86_64::id::r11};
static uint32_t g_contained_r12[] = {registers::x86_64::id::r12};
static uint32_t g_contained_r13[] = {registers::x86_64::id::r13};
static uint32_t g_contained_r14[] = {registers::x86_64::id::r14};
static uint32_t g_contained_r15[] = {registers::x86_64::id::r15};
static uint32_t g_invalidate_rax[] = {
    registers::x86_64::id::rax, registers::x86_64::id::eax,
    registers::x86_64::id::ax, registers::x86_64::id::ah,
    registers::x86_64::id::al};
static uint32_t g_invalidate_rbx[] = {
    registers::x86_64::id::rbx, registers::x86_64::id::ebx,
    registers::x86_64::id::bx, registers::x86_64::id::bh,
    registers::x86_64::id::bl};
static uint32_t g_invalidate_rcx[] = {
    registers::x86_64::id::rcx, registers::x86_64::id::ecx,
    registers::x86_64::id::cx, registers::x86_64::id::ch,
    registers::x86_64::id::cl};
static uint32_t g_invalidate_rdx[] = {
    registers::x86_64::id::rdx, registers::x86_64::id::edx,
    registers::x86_64::id::dx, registers::x86_64::id::dh,
    registers::x86_64::id::dl};
static uint32_t g_invalidate_rdi[] = {
    registers::x86_64::id::rdi, registers::x86_64::id::edi,
    registers::x86_64::id::di, registers::x86_64::id::dil};
static uint32_t g_invalidate_rsi[] = {
    registers::x86_64::id::rsi, registers::x86_64::id::esi,
    registers::x86_64::id::si, registers::x86_64::id::sil};
static uint32_t g_invalidate_rbp[] = {
    registers::x86_64::id::rbp, registers::x86_64::id::ebp,
    registers::x86_64::id::bp, registers::x86_64::id::bpl};
static uint32_t g_invalidate_rsp[] = {
    registers::x86_64::id::rsp, registers::x86_64::id::esp,
    registers::x86_64::id::sp, registers::x86_64::id::spl};
static uint32_t g_invalidate_r8[] = {
    registers::x86_64::id::r8, registers::x86_64::id::r8d,
    registers::x86_64::id::r8w, registers::x86_64::id::r8l};
static uint32_t g_invalidate_r9[] = {
    registers::x86_64::id::r9, registers::x86_64::id::r9d,
    registers::x86_64::id::r9w, registers::x86_64::id::r9l};
static uint32_t g_invalidate_r10[] = {
    registers::x86_64::id::r10, registers::x86_64::id::r10d,
    registers::x86_64::id::r10w, registers::x86_64::id::r10l};
static uint32_t g_invalidate_r11[] = {
    registers::x86_64::id::r11, registers::x86_64::id::r11d,
    registers::x86_64::id::r11w, registers::x86_64::id::r11l};
static uint32_t g_invalidate_r12[] = {
    registers::x86_64::id::r12, registers::x86_64::id::r12d,
    registers::x86_64::id::r12w, registers::x86_64::id::r12l};
static uint32_t g_invalidate_r13[] = {
    registers::x86_64::id::r13, registers::x86_64::id::r13d,
    registers::x86_64::id::r13w, registers::x86_64::id::r13l};
static uint32_t g_invalidate_r14[] = {
    registers::x86_64::id::r14, registers::x86_64::id::r14d,
    registers::x86_64::id::r14w, registers::x86_64::id::r14l};
static uint32_t g_invalidate_r15[] = {
    registers::x86_64::id::r15, registers::x86_64::id::r15d,
    registers::x86_64::id::r15w, registers::x86_64::id::r15l};
static uint32_t g_contained_fip[] = {registers::x86_64::id::fip};
static uint32_t g_contained_fdp[] = {registers::x86_64::id::fdp};
static uint32_t g_invalidate_fip[] = {registers::x86_64::id::fip,
                                      registers::x86_64::id::fioff,
                                      registers::x86_64::id::fiseg};
static uint32_t g_invalidate_fdp[] = {registers::x86_64::id::fdp,
                                      registers::x86_64::id::fooff,
                                      registers::x86_64::id::foseg};
static uint32_t g_contained_st0_32[] = {registers::x86::id::st0};
static uint32_t g_contained_st1_32[] = {registers::x86::id::st1};
static uint32_t g_contained_st2_32[] = {registers::x86::id::st2};
static uint32_t g_contained_st3_32[] = {registers::x86::id::st3};
static uint32_t g_contained_st4_32[] = {registers::x86::id::st4};
static uint32_t g_contained_st5_32[] = {registers::x86::id::st5};
static uint32_t g_contained_st6_32[] = {registers::x86::id::st6};
static uint32_t g_contained_st7_32[] = {registers::x86::id::st7};
static uint32_t g_invalidate_st0_32[] = {registers::x86::id::st0,
                                         registers::x86::id::mm0};
static uint32_t g_invalidate_st1_32[] = {registers::x86::id::st1,
                                         registers::x86::id::mm1};
static uint32_t g_invalidate_st2_32[] = {registers::x86::id::st2,
                                         registers::x86::id::mm2};
static uint32_t g_invalidate_st3_32[] = {registers::x86::id::st3,
                                         registers::x86::id::mm3};
static uint32_t g_invalidate_st4_32[] = {registers::x86::id::st4,
                                         registers::x86::id::mm4};
static uint32_t g_invalidate_st5_32[] = {registers::x86::id::st5,
                                         registers::x86::id::mm5};
static uint32_t g_invalidate_st6_32[] = {registers::x86::id::st6,
                                         registers::x86::id::mm6};
static uint32_t g_invalidate_st7_32[] = {registers::x86::id::st7,
                                         registers::x86::id::mm7};
static uint32_t g_contained_st0_64[] = {registers::x86_64::id::st0};
static uint32_t g_contained_st1_64[] = {registers::x86_64::id::st1};
static uint32_t g_contained_st2_64[] = {registers::x86_64::id::st2};
static uint32_t g_contained_st3_64[] = {registers::x86_64::id::st3};
static uint32_t g_contained_st4_64[] = {registers::x86_64::id::st4};
static uint32_t g_contained_st5_64[] = {registers::x86_64::id::st5};
static uint32_t g_contained_st6_64[] = {registers::x86_64::id::st6};
static uint32_t g_contained_st7_64[] = {registers::x86_64::id::st7};
static uint32_t g_invalidate_st0_64[] = {registers::x86_64::id::st0,
                                         registers::x86_64::id::mm0};
static uint32_t g_invalidate_st1_64[] = {registers::x86_64::id::st1,
                                         registers::x86_64::id::mm1};
static uint32_t g_invalidate_st2_64[] = {registers::x86_64::id::st2,
                                         registers::x86_64::id::mm2};
static uint32_t g_invalidate_st3_64[] = {registers::x86_64::id::st3,
                                         registers::x86_64::id::mm3};
static uint32_t g_invalidate_st4_64[] = {registers::x86_64::id::st4,
                                         registers::x86_64::id::mm4};
static uint32_t g_invalidate_st5_64[] = {registers::x86_64::id::st5,
                                         registers::x86_64::id::mm5};
static uint32_t g_invalidate_st6_64[] = {registers::x86_64::id::st6,
                                         registers::x86_64::id::mm6};
static uint32_t g_invalidate_st7_64[] = {registers::x86_64::id::st7,
                                         registers::x86_64::id::mm7};
} // namespace x86_register_info
namespace RegisterInfos_x86_64_with_base_shared {
static uint32_t g_contained_rax[] = {registers::x86_64::id::with_base::rax};
static uint32_t g_contained_rbx[] = {registers::x86_64::id::with_base::rbx};
static uint32_t g_contained_rcx[] = {registers::x86_64::id::with_base::rcx};
static uint32_t g_contained_rdx[] = {registers::x86_64::id::with_base::rdx};
static uint32_t g_contained_rdi[] = {registers::x86_64::id::with_base::rdi};
static uint32_t g_contained_rsi[] = {registers::x86_64::id::with_base::rsi};
static uint32_t g_contained_rbp[] = {registers::x86_64::id::with_base::rbp};
static uint32_t g_contained_rsp[] = {registers::x86_64::id::with_base::rsp};
static uint32_t g_contained_r8[] = {registers::x86_64::id::with_base::r8};
static uint32_t g_contained_r9[] = {registers::x86_64::id::with_base::r9};
static uint32_t g_contained_r10[] = {registers::x86_64::id::with_base::r10};
static uint32_t g_contained_r11[] = {registers::x86_64::id::with_base::r11};
static uint32_t g_contained_r12[] = {registers::x86_64::id::with_base::r12};
static uint32_t g_contained_r13[] = {registers::x86_64::id::with_base::r13};
static uint32_t g_contained_r14[] = {registers::x86_64::id::with_base::r14};
static uint32_t g_contained_r15[] = {registers::x86_64::id::with_base::r15};
static uint32_t g_invalidate_rax[] = {
    registers::x86_64::id::with_base::rax,
    registers::x86_64::id::with_base::eax, registers::x86_64::id::with_base::ax,
    registers::x86_64::id::with_base::ah, registers::x86_64::id::with_base::al};
static uint32_t g_invalidate_rbx[] = {
    registers::x86_64::id::with_base::rbx,
    registers::x86_64::id::with_base::ebx, registers::x86_64::id::with_base::bx,
    registers::x86_64::id::with_base::bh, registers::x86_64::id::with_base::bl};
static uint32_t g_invalidate_rcx[] = {
    registers::x86_64::id::with_base::rcx,
    registers::x86_64::id::with_base::ecx, registers::x86_64::id::with_base::cx,
    registers::x86_64::id::with_base::ch, registers::x86_64::id::with_base::cl};
static uint32_t g_invalidate_rdx[] = {
    registers::x86_64::id::with_base::rdx, registers::x86_64::id::with_base::edx,
    registers::x86_64::id::with_base::dx, registers::x86_64::id::with_base::dh,
    registers::x86_64::id::with_base::dl};
static uint32_t g_invalidate_rdi[] = {
    registers::x86_64::id::with_base::rdi, registers::x86_64::id::with_base::edi,
    registers::x86_64::id::with_base::di, registers::x86_64::id::with_base::dil};
static uint32_t g_invalidate_rsi[] = {
    registers::x86_64::id::with_base::rsi, registers::x86_64::id::with_base::esi,
    registers::x86_64::id::with_base::si, registers::x86_64::id::with_base::sil};
static uint32_t g_invalidate_rbp[] = {
    registers::x86_64::id::with_base::rbp, registers::x86_64::id::with_base::ebp,
    registers::x86_64::id::with_base::bp, registers::x86_64::id::with_base::bpl};
static uint32_t g_invalidate_rsp[] = {
    registers::x86_64::id::with_base::rsp, registers::x86_64::id::with_base::esp,
    registers::x86_64::id::with_base::sp, registers::x86_64::id::with_base::spl};
static uint32_t g_invalidate_r8[] = {
    registers::x86_64::id::with_base::r8, registers::x86_64::id::with_base::r8d,
    registers::x86_64::id::with_base::r8w, registers::x86_64::id::with_base::r8l};
static uint32_t g_invalidate_r9[] = {
    registers::x86_64::id::with_base::r9, registers::x86_64::id::with_base::r9d,
    registers::x86_64::id::with_base::r9w, registers::x86_64::id::with_base::r9l};
static uint32_t g_invalidate_r10[] = {
    registers::x86_64::id::with_base::r10, registers::x86_64::id::with_base::r10d,
    registers::x86_64::id::with_base::r10w, registers::x86_64::id::with_base::r10l};
static uint32_t g_invalidate_r11[] = {
    registers::x86_64::id::with_base::r11, registers::x86_64::id::with_base::r11d,
    registers::x86_64::id::with_base::r11w, registers::x86_64::id::with_base::r11l};
static uint32_t g_invalidate_r12[] = {
    registers::x86_64::id::with_base::r12, registers::x86_64::id::with_base::r12d,
    registers::x86_64::id::with_base::r12w, registers::x86_64::id::with_base::r12l};
static uint32_t g_invalidate_r13[] = {
    registers::x86_64::id::with_base::r13, registers::x86_64::id::with_base::r13d,
    registers::x86_64::id::with_base::r13w, registers::x86_64::id::with_base::r13l};
static uint32_t g_invalidate_r14[] = {
    registers::x86_64::id::with_base::r14, registers::x86_64::id::with_base::r14d,
    registers::x86_64::id::with_base::r14w, registers::x86_64::id::with_base::r14l};
static uint32_t g_invalidate_r15[] = {
    registers::x86_64::id::with_base::r15, registers::x86_64::id::with_base::r15d,
    registers::x86_64::id::with_base::r15w, registers::x86_64::id::with_base::r15l};
static uint32_t g_contained_fip[] = {registers::x86_64::id::with_base::fip};
static uint32_t g_contained_fdp[] = {registers::x86_64::id::with_base::fdp};
static uint32_t g_invalidate_fip[] = {registers::x86_64::id::with_base::fip,
                                      registers::x86_64::id::with_base::fioff,
                                      registers::x86_64::id::with_base::fiseg};
static uint32_t g_invalidate_fdp[] = {registers::x86_64::id::with_base::fdp,
                                      registers::x86_64::id::with_base::fooff,
                                      registers::x86_64::id::with_base::foseg};
static uint32_t g_contained_st0_64[] = {registers::x86_64::id::with_base::st0};
static uint32_t g_contained_st1_64[] = {registers::x86_64::id::with_base::st1};
static uint32_t g_contained_st2_64[] = {registers::x86_64::id::with_base::st2};
static uint32_t g_contained_st3_64[] = {registers::x86_64::id::with_base::st3};
static uint32_t g_contained_st4_64[] = {registers::x86_64::id::with_base::st4};
static uint32_t g_contained_st5_64[] = {registers::x86_64::id::with_base::st5};
static uint32_t g_contained_st6_64[] = {registers::x86_64::id::with_base::st6};
static uint32_t g_contained_st7_64[] = {registers::x86_64::id::with_base::st7};
static uint32_t g_invalidate_st0_64[] = {registers::x86_64::id::with_base::st0,
                                         registers::x86_64::id::with_base::mm0};
static uint32_t g_invalidate_st1_64[] = {registers::x86_64::id::with_base::st1,
                                         registers::x86_64::id::with_base::mm1};
static uint32_t g_invalidate_st2_64[] = {registers::x86_64::id::with_base::st2,
                                         registers::x86_64::id::with_base::mm2};
static uint32_t g_invalidate_st3_64[] = {registers::x86_64::id::with_base::st3,
                                         registers::x86_64::id::with_base::mm3};
static uint32_t g_invalidate_st4_64[] = {registers::x86_64::id::with_base::st4,
                                         registers::x86_64::id::with_base::mm4};
static uint32_t g_invalidate_st5_64[] = {registers::x86_64::id::with_base::st5,
                                         registers::x86_64::id::with_base::mm5};
static uint32_t g_invalidate_st6_64[] = {registers::x86_64::id::with_base::st6,
                                         registers::x86_64::id::with_base::mm6};
static uint32_t g_invalidate_st7_64[] = {registers::x86_64::id::with_base::st7,
                                         registers::x86_64::id::with_base::mm7};
} // namespace RegisterInfos_x86_64_with_base_shared
} // namespace

// Convert the 8-bit abridged FPU Tag Word (as found in FXSAVE) to the full
// 16-bit FPU Tag Word (as found in FSAVE, and used by gdb protocol).  This
// requires knowing the values of the ST(i) registers and the FPU Status Word.
namespace lldb_private {} // namespace lldb_private

static uint32_t FPRValueByteSize(llvm::StringRef field) {
  if (field == "ptr.x86_64.fip" || field == "ptr.x86_64.fdp")
    return 8;
  if (field.starts_with("ptr.i386_.") || field == "mxcsr" ||
      field == "mxcsrmask")
    return 4;
  return 2;
}

static llvm::ArrayRef<RegisterDescriptor> Metadata_i386() {
  constexpr uint32_t word_size = 4;
#include "Register/Arch/RegisterDescriptions_i386.inc"
  return g_register_infos_i386;
}
static llvm::ArrayRef<RegisterDescriptor> Metadata_x86_64() {
  constexpr uint32_t word_size = 8;
#include "Register/Arch/RegisterDescriptions_x86_64.inc"
  return g_register_infos_x86_64;
}
static llvm::ArrayRef<RegisterDescriptor> Metadata_x86_64_with_base() {
  constexpr uint32_t word_size = 8;
#include "Register/Arch/RegisterDescriptions_x86_64_with_base.inc"
  return g_register_infos_x86_64_with_base;
}

namespace {
// A value profile selects exposed CPU registers and widths, not storage slots.
struct ValueSpec {
  llvm::StringRef name;
  uint32_t byte_size;
};
std::shared_ptr<const RegisterDescription>
BuildDescription(const ArchSpec &arch, llvm::ArrayRef<ValueSpec> entries,
                 bool with_segment_bases, X86RegisterABI abi,
                 X86RegisterMode mode) {
  const bool is_64 = mode == X86RegisterMode::X86_64 ||
                     (mode == X86RegisterMode::Target &&
                      arch.GetMachine() == llvm::Triple::x86_64);
  auto base_rows = is_64 ? Metadata_x86_64() : Metadata_i386();
  auto rows =
      is_64 && with_segment_bases ? Metadata_x86_64_with_base() : base_rows;
  auto base = llvm::cantFail(RegisterDescription::Create(
      std::vector<RegisterDescriptor>(base_rows.begin(), base_rows.end())));
  auto source = llvm::cantFail(RegisterDescription::Create(
      std::vector<RegisterDescriptor>(rows.begin(), rows.end())));
  std::map<RegisterID, RegisterID> identities;
  RegisterID next_id = 0;
  for (const auto &reg : base.GetRegisters())
    next_id = std::max(next_id, reg.id + 1);
  for (const auto &reg : source.GetRegisters()) {
    const auto *canonical = base.Find(reg.name);
    identities[reg.id] = canonical ? canonical->id : next_id++;
  }
  std::vector<RegisterDescriptor> registers;
  std::set<RegisterID> selected;
  for (const auto &entry : entries) {
    const auto *original =
        entry.name.empty() ? nullptr : source.Find(entry.name);
    if (!original)
      llvm::report_fatal_error("unknown x86 register in saved-state layout");
    auto reg = *original;
    reg.id = identities.at(reg.id);
    reg.byte_size = entry.byte_size;
    if (reg.byte_size == 1 && (reg.name == "ah" || reg.name == "bh" ||
                               reg.name == "ch" || reg.name == "dh"))
      reg.value_byte_offset = 1;
    if ((reg.name == "fiseg" || reg.name == "foseg") &&
        reg.value_registers.size() == 1)
      reg.value_byte_offset = 4;
    for (auto *references : {&reg.value_registers, &reg.invalidate_registers})
      for (auto &id : *references)
        id = identities.at(id);
    if (abi == X86RegisterABI::Windows && is_64) {
      uint32_t &generic = reg.numbers[eRegisterKindGeneric];
      if (generic >= LLDB_REGNUM_GENERIC_ARG1 &&
          generic <= LLDB_REGNUM_GENERIC_ARG8)
        generic = LLDB_INVALID_REGNUM;
      for (auto pair : {std::make_pair("rcx", LLDB_REGNUM_GENERIC_ARG1),
                        std::make_pair("rdx", LLDB_REGNUM_GENERIC_ARG2),
                        std::make_pair("r8", LLDB_REGNUM_GENERIC_ARG3),
                        std::make_pair("r9", LLDB_REGNUM_GENERIC_ARG4)})
        if (reg.name == pair.first)
          generic = pair.second;
    }
    if (abi == X86RegisterABI::Windows &&
        llvm::StringRef(reg.name).starts_with("xmm")) {
      reg.encoding = eEncodingUint;
      reg.format = eFormatVectorOfUInt64;
    }
    if (is_64 && reg.byte_size == 4 && !reg.value_registers.empty() &&
        (llvm::StringRef(reg.name).starts_with("e") ||
         (llvm::StringRef(reg.name).starts_with("r") &&
          llvm::StringRef(reg.name).ends_with("d"))))
      reg.zero_extend_parent = reg.value_registers.front();
    selected.insert(reg.id);
    registers.push_back(std::move(reg));
  }
  for (auto &reg : registers) {
    llvm::erase_if(reg.invalidate_registers,
                   [&](RegisterID id) { return !selected.count(id); });
    llvm::erase_if(reg.value_registers,
                   [&](RegisterID id) { return !selected.count(id); });
  }
  std::vector<RegisterGroup> groups = {
      {"General Purpose Registers", "gpr", {}},
      {"Floating Point Registers", "fpu", {}},
      {"Advanced Vector Extensions", "avx", {}},
      {"Memory Protection Extensions", "mpx", {}},
      {"Hardware Debug Registers", "dbr", {}}};
  const uint32_t first_fpr = !is_64 ? registers::x86::id::k_first_fpr
                             : with_segment_bases
                                 ? registers::x86_64::id::with_base::k_first_fpr
                                 : registers::x86_64::id::k_first_fpr;
  for (const auto &reg : registers) {
    llvm::StringRef name(reg.name);
    size_t group = reg.numbers[eRegisterKindLLDB] < first_fpr ? 0 : 1;
    if (name.starts_with("ymm"))
      group = 2;
    if (name.starts_with("bnd"))
      group = 3;
    if (name.size() > 2 && name.starts_with("dr"))
      group = 4;
    groups[group].registers.push_back(reg.id);
  }
  llvm::erase_if(groups, [](const RegisterGroup &group) {
    return group.registers.empty();
  });
  return std::make_shared<RegisterDescription>(llvm::cantFail(
      RegisterDescription::Create(std::move(registers), std::move(groups))));
}

} // namespace

std::shared_ptr<const RegisterDescription> RegisterInfo_x86::CreateDescription(
    const ArchSpec &arch, bool with_segment_bases, X86RegisterABI abi,
    X86RegisterMode mode, X86RegisterProfile profile) {
  bool is_64 = mode == X86RegisterMode::X86_64 ||
               (mode == X86RegisterMode::Target &&
                arch.GetMachine() == llvm::Triple::x86_64);
  auto rows = is_64 ? (with_segment_bases ? Metadata_x86_64_with_base()
                                          : Metadata_x86_64())
                    : Metadata_i386();
  std::vector<ValueSpec> values;
  for (const auto &row : rows)
    values.push_back({row.name, row.byte_size});
  auto complete = BuildDescription(arch, values, with_segment_bases, abi, mode);
  std::vector<RegisterGroup> groups;
  std::set<RegisterID> selected;
  for (auto group : complete->GetGroups()) {
    llvm::erase_if(group.registers, [&](RegisterID id) {
      const auto *reg = complete->Find(id);
      if (group.short_name == "gpr") {
        const bool segment = llvm::is_contained(
            {"cs", "ds", "es", "fs", "gs", "ss", "fs_base", "gs_base"},
            llvm::StringRef(reg->name));
        return (!profile.segments && segment) ||
               (!profile.gpr_aliases && !reg->value_registers.empty());
      }
      if (group.short_name == "fpu")
        return llvm::StringRef(reg->name).starts_with("xmm") ? !profile.simd
                                                             : !profile.x87;
      if (group.short_name == "avx")
        return !profile.avx;
      if (group.short_name == "mpx")
        return !profile.mpx;
      if (group.short_name == "dbr")
        return !profile.debug;
      return false;
    });
    selected.insert(group.registers.begin(), group.registers.end());
    if (!group.registers.empty())
      groups.push_back(std::move(group));
  }
  std::vector<RegisterDescriptor> regs;
  for (auto reg : complete->GetRegisters()) {
    if (!selected.count(reg.id))
      continue;
    for (auto *refs : {&reg.value_registers, &reg.invalidate_registers})
      llvm::erase_if(*refs, [&](RegisterID id) { return !selected.count(id); });
    if (!selected.count(reg.zero_extend_parent))
      reg.zero_extend_parent = LLDB_INVALID_REGNUM;
    regs.push_back(std::move(reg));
  }
  return std::make_shared<RegisterDescription>(llvm::cantFail(
      RegisterDescription::Create(std::move(regs), std::move(groups))));
}

uint16_t lldb_private::AbridgedToFullTagWord(uint8_t abridged_tw, uint16_t sw,
                                             llvm::ArrayRef<MMSReg> st_regs) {
  // Tag word is using internal FPU register numbering rather than ST(i).
  // Mapping to ST(i): i = FPU regno - TOP (Status Word, bits 11:13).
  // Here we start with FPU reg 7 and go down.
  int st = 7 - ((sw >> 11) & 7);
  uint16_t tw = 0;
  for (uint8_t mask = 0x80; mask != 0; mask >>= 1) {
    tw <<= 2;
    if (abridged_tw & mask) {
      // The register is non-empty, so we need to check the value of ST(i).
      uint16_t exp =
          st_regs[st].comp.sign_exp & 0x7fff; // Discard the sign bit.
      if (exp == 0) {
        if (st_regs[st].comp.mantissa == 0)
          tw |= 1; // Zero
        else
          tw |= 2; // Denormal
      } else if (exp == 0x7fff)
        tw |= 2; // Infinity or NaN
      // 0 if normal number
    } else
      tw |= 3; // Empty register

    // Rotate ST down.
    st = (st - 1) & 7;
  }

  return tw;
}

// Convert the 16-bit FPU Tag Word to the abridged 8-bit value, to be written
// into FXSAVE.
uint8_t lldb_private::FullToAbridgedTagWord(uint16_t tw) {
  uint8_t abridged_tw = 0;
  for (uint16_t mask = 0xc000; mask != 0; mask >>= 2) {
    abridged_tw <<= 1;
    // full TW uses 11 for empty registers, aTW uses 0
    if ((tw & mask) != mask)
      abridged_tw |= 1;
  }
  return abridged_tw;
}

RegisterLayout lldb_private::CreateX86RegisterLayout(
    const ArchSpec &arch, llvm::ArrayRef<RegisterStorageField> gpr_fields,
    size_t gpr_size, bool gpr_is_64, bool segment_bases, X86RegisterABI abi) {
  using namespace x86_layout;
  auto cpu = RegisterInfo_x86::CreateDescription(arch, segment_bases, abi);
  std::vector<RegisterDescriptor> regs(cpu->GetRegisters().begin(),
                                       cpu->GetRegisters().end());
  std::vector<RegisterGroup> groups(cpu->GetGroups().begin(),
                                    cpu->GetGroups().end());
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(GP, gpr_size));
  llvm::cantFail(layout.AddBank(FP, sizeof(FXSAVE)));
  llvm::cantFail(layout.AddBank(YMMHigh, 256));
  llvm::cantFail(layout.AddBank(Bounds, 64));
  llvm::cantFail(layout.AddBank(BoundsControl, 16));
  llvm::cantFail(layout.AddBank(Debug, 8 * (gpr_is_64 ? 8 : 4)));
  auto field = [&](llvm::StringRef name) -> const RegisterStorageField * {
    if (gpr_is_64 && arch.GetMachine() == llvm::Triple::x86) {
      if (name == "eflags")
        name = "rflags";
      else if (name.starts_with("e")) {
        std::string wide = "r" + name.drop_front().str();
        for (const auto &value : gpr_fields)
          if (wide == value.name)
            return &value;
      }
    }
    for (const auto &value : gpr_fields)
      if (name == value.name)
        return &value;
    return nullptr;
  };
  std::map<RegisterID, RegisterLocation> locations;
  auto stored = [&](RegisterID id, RegisterBankID bank, uint32_t offset,
                    uint32_t size) {
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, size});
    locations.emplace(id, std::move(location));
  };
  for (auto &reg : regs) {
    if (auto *slot = field(reg.name)) {
      reg.byte_size = std::min(reg.byte_size, slot->byte_size);
      stored(reg.id, GP, slot->byte_offset, reg.byte_size);
      continue;
    }
    llvm::StringRef name(reg.name);
    unsigned index;
    auto indexed = [&](llvm::StringRef prefix) {
      return name.consume_front(prefix) && !name.getAsInteger(10, index);
    };
    name = reg.name;
    if (indexed("ymm")) {
      RegisterLocation location;
      location.kind = RegisterLocation::Stored;
      location.pieces = {{FP, offsetof(FXSAVE, xmm) + index * 16, 16, 0},
                         {YMMHigh, index * 16, 16, 16}};
      locations.emplace(reg.id, std::move(location));
      continue;
    }
    name = reg.name;
    if (indexed("xmm")) {
      stored(reg.id, FP, offsetof(FXSAVE, xmm) + index * 16, reg.byte_size);
      continue;
    }
    name = reg.name;
    if (indexed("st")) {
      stored(reg.id, FP, offsetof(FXSAVE, stmm) + index * 16, reg.byte_size);
      continue;
    }
    name = reg.name;
    if (indexed("mm")) {
      stored(reg.id, FP, offsetof(FXSAVE, stmm) + index * 16, reg.byte_size);
      continue;
    }
    name = reg.name;
    if (indexed("dr")) {
      stored(reg.id, Debug, index * (gpr_is_64 ? 8 : 4), reg.byte_size);
      continue;
    }
    name = reg.name;
    if (indexed("bnd")) {
      stored(reg.id, Bounds, index * 16, reg.byte_size);
      continue;
    }
    if (reg.name == "bndcfgu") {
      stored(reg.id, BoundsControl, 0, reg.byte_size);
      continue;
    }
    if (reg.name == "bndstatus") {
      stored(reg.id, BoundsControl, 8, reg.byte_size);
      continue;
    }
    struct FPField {
      const char *name;
      uint32_t offset;
    };
    const FPField scalars[] = {{"fctrl", offsetof(FXSAVE, fctrl)},
                               {"fstat", offsetof(FXSAVE, fstat)},
                               {"ftag", offsetof(FXSAVE, ftag)},
                               {"fop", offsetof(FXSAVE, fop)},
                               {"fiseg", offsetof(FXSAVE, ptr.i386_.fiseg)},
                               {"fioff", offsetof(FXSAVE, ptr.i386_.fioff)},
                               {"foseg", offsetof(FXSAVE, ptr.i386_.foseg)},
                               {"fooff", offsetof(FXSAVE, ptr.i386_.fooff)},
                               {"fip", offsetof(FXSAVE, ptr.x86_64.fip)},
                               {"fdp", offsetof(FXSAVE, ptr.x86_64.fdp)},
                               {"mxcsr", offsetof(FXSAVE, mxcsr)},
                               {"mxcsrmask", offsetof(FXSAVE, mxcsrmask)}};
    for (const auto &slot : scalars)
      if (reg.name == slot.name)
        stored(reg.id, FP, slot.offset, reg.byte_size);
  }
  // GPR aliases use their architecture value relation, never slot subtraction.
  for (size_t pass = 0; pass < regs.size(); ++pass)
    for (const auto &reg : regs) {
      if (locations.count(reg.id) || reg.value_registers.size() != 1)
        continue;
      auto parent = locations.find(reg.value_registers.front());
      if (parent == locations.end() || parent->second.pieces.size() != 1 ||
          parent->second.pieces.front().bank != GP)
        continue;
      const auto &piece = parent->second.pieces.front();
      stored(reg.id, GP, piece.byte_offset + reg.value_byte_offset,
             reg.byte_size);
    }
  uint32_t user_count = regs.size();
  for (const auto &reg : regs)
    if (llvm::StringRef(reg.name).starts_with("dr"))
      --user_count;
  auto description = std::make_shared<RegisterDescription>(llvm::cantFail(
      RegisterDescription::Create(std::move(regs), std::move(groups))));
  for (auto &[id, location] : locations)
    llvm::cantFail(layout.Add(id, std::move(location)));
  std::vector<RegisterID> visible;
  for (const auto &reg : description->GetRegisters().take_front(user_count))
    visible.push_back(reg.id);
  description = std::make_shared<RegisterDescription>(
      llvm::cantFail(description->WithUserRegisters(visible)));
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
