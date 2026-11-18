//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_riscv.h"
#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <set>

using namespace lldb_private;
using namespace lldb_private::registers::riscv;

namespace {

struct CSRName {
  uint32_t address;
  const char *name;
  bool rv32_only = false;
};

constexpr CSRName csr_names[] = {
    {0x001, "fflags"},
    {0x002, "frm"},
    {0x003, "fcsr"},
    {0x008, "vstart"},
    {0x009, "vxsat"},
    {0x00a, "vxrm"},
    {0x00f, "vcsr"},
    {0x100, "sstatus"},
    {0x104, "sie"},
    {0x105, "stvec"},
    {0x106, "scounteren"},
    {0x10a, "senvcfg"},
    {0x10c, "sstateen0"},
    {0x10d, "sstateen1"},
    {0x10e, "sstateen2"},
    {0x10f, "sstateen3"},
    {0x120, "scountinhibit"},
    {0x140, "sscratch"},
    {0x141, "sepc"},
    {0x142, "scause"},
    {0x143, "stval"},
    {0x180, "satp"},
    {0x200, "vsstatus"},
    {0x204, "vsie"},
    {0x205, "vstvec"},
    {0x240, "vsscratch"},
    {0x241, "vsepc"},
    {0x242, "vscause"},
    {0x243, "vstval"},
    {0x244, "vsip"},
    {0x280, "vsatp"},
    {0x300, "mstatus"},
    {0x301, "misa"},
    {0x302, "medeleg"},
    {0x303, "mideleg"},
    {0x304, "mie"},
    {0x305, "mtvec"},
    {0x306, "mcounteren"},
    {0x30a, "menvcfg"},
    {0x30c, "mstateen0"},
    {0x30d, "mstateen1"},
    {0x30e, "mstateen2"},
    {0x30f, "mstateen3"},
    {0x310, "mstatush", true},
    {0x312, "medelegh", true},
    {0x31a, "menvcfgh", true},
    {0x31c, "mstateen0h", true},
    {0x31d, "mstateen1h", true},
    {0x31e, "mstateen2h", true},
    {0x31f, "mstateen3h", true},
    {0x320, "mcountinhibit"},
    {0x323, "mhpmevent3"},
    {0x324, "mhpmevent4"},
    {0x325, "mhpmevent5"},
    {0x326, "mhpmevent6"},
    {0x327, "mhpmevent7"},
    {0x328, "mhpmevent8"},
    {0x329, "mhpmevent9"},
    {0x32a, "mhpmevent10"},
    {0x32b, "mhpmevent11"},
    {0x32c, "mhpmevent12"},
    {0x32d, "mhpmevent13"},
    {0x32e, "mhpmevent14"},
    {0x32f, "mhpmevent15"},
    {0x330, "mhpmevent16"},
    {0x331, "mhpmevent17"},
    {0x332, "mhpmevent18"},
    {0x333, "mhpmevent19"},
    {0x334, "mhpmevent20"},
    {0x335, "mhpmevent21"},
    {0x336, "mhpmevent22"},
    {0x337, "mhpmevent23"},
    {0x338, "mhpmevent24"},
    {0x339, "mhpmevent25"},
    {0x33a, "mhpmevent26"},
    {0x33b, "mhpmevent27"},
    {0x33c, "mhpmevent28"},
    {0x33d, "mhpmevent29"},
    {0x33e, "mhpmevent30"},
    {0x33f, "mhpmevent31"},
    {0x340, "mscratch"},
    {0x341, "mepc"},
    {0x342, "mcause"},
    {0x343, "mtval"},
    {0x344, "mip"},
    {0x34a, "mtinst"},
    {0x34b, "mtval2"},
    {0x3a0, "pmpcfg0"},
    {0x3a1, "pmpcfg1", true},
    {0x3a2, "pmpcfg2"},
    {0x3a3, "pmpcfg3", true},
    {0x3a4, "pmpcfg4"},
    {0x3a5, "pmpcfg5", true},
    {0x3a6, "pmpcfg6"},
    {0x3a7, "pmpcfg7", true},
    {0x3a8, "pmpcfg8"},
    {0x3a9, "pmpcfg9", true},
    {0x3aa, "pmpcfg10"},
    {0x3ab, "pmpcfg11", true},
    {0x3ac, "pmpcfg12"},
    {0x3ad, "pmpcfg13", true},
    {0x3ae, "pmpcfg14"},
    {0x3af, "pmpcfg15", true},
    {0x3b0, "pmpaddr0"},
    {0x3b1, "pmpaddr1"},
    {0x3b2, "pmpaddr2"},
    {0x3b3, "pmpaddr3"},
    {0x3b4, "pmpaddr4"},
    {0x3b5, "pmpaddr5"},
    {0x3b6, "pmpaddr6"},
    {0x3b7, "pmpaddr7"},
    {0x3b8, "pmpaddr8"},
    {0x3b9, "pmpaddr9"},
    {0x3ba, "pmpaddr10"},
    {0x3bb, "pmpaddr11"},
    {0x3bc, "pmpaddr12"},
    {0x3bd, "pmpaddr13"},
    {0x3be, "pmpaddr14"},
    {0x3bf, "pmpaddr15"},
    {0x3c0, "pmpaddr16"},
    {0x3c1, "pmpaddr17"},
    {0x3c2, "pmpaddr18"},
    {0x3c3, "pmpaddr19"},
    {0x3c4, "pmpaddr20"},
    {0x3c5, "pmpaddr21"},
    {0x3c6, "pmpaddr22"},
    {0x3c7, "pmpaddr23"},
    {0x3c8, "pmpaddr24"},
    {0x3c9, "pmpaddr25"},
    {0x3ca, "pmpaddr26"},
    {0x3cb, "pmpaddr27"},
    {0x3cc, "pmpaddr28"},
    {0x3cd, "pmpaddr29"},
    {0x3ce, "pmpaddr30"},
    {0x3cf, "pmpaddr31"},
    {0x3d0, "pmpaddr32"},
    {0x3d1, "pmpaddr33"},
    {0x3d2, "pmpaddr34"},
    {0x3d3, "pmpaddr35"},
    {0x3d4, "pmpaddr36"},
    {0x3d5, "pmpaddr37"},
    {0x3d6, "pmpaddr38"},
    {0x3d7, "pmpaddr39"},
    {0x3d8, "pmpaddr40"},
    {0x3d9, "pmpaddr41"},
    {0x3da, "pmpaddr42"},
    {0x3db, "pmpaddr43"},
    {0x3dc, "pmpaddr44"},
    {0x3dd, "pmpaddr45"},
    {0x3de, "pmpaddr46"},
    {0x3df, "pmpaddr47"},
    {0x3e0, "pmpaddr48"},
    {0x3e1, "pmpaddr49"},
    {0x3e2, "pmpaddr50"},
    {0x3e3, "pmpaddr51"},
    {0x3e4, "pmpaddr52"},
    {0x3e5, "pmpaddr53"},
    {0x3e6, "pmpaddr54"},
    {0x3e7, "pmpaddr55"},
    {0x3e8, "pmpaddr56"},
    {0x3e9, "pmpaddr57"},
    {0x3ea, "pmpaddr58"},
    {0x3eb, "pmpaddr59"},
    {0x3ec, "pmpaddr60"},
    {0x3ed, "pmpaddr61"},
    {0x3ee, "pmpaddr62"},
    {0x3ef, "pmpaddr63"},
    {0x5a8, "scontext"},
    {0x600, "hstatus"},
    {0x602, "hedeleg"},
    {0x603, "hideleg"},
    {0x604, "hie"},
    {0x605, "htimedelta"},
    {0x606, "hcounteren"},
    {0x607, "hgeie"},
    {0x60a, "henvcfg"},
    {0x60c, "hstateen0"},
    {0x60d, "hstateen1"},
    {0x60e, "hstateen2"},
    {0x60f, "hstateen3"},
    {0x612, "hedelegh", true},
    {0x615, "htimedeltah", true},
    {0x61a, "henvcfgh", true},
    {0x61c, "hstateen0h", true},
    {0x61d, "hstateen1h", true},
    {0x61e, "hstateen2h", true},
    {0x61f, "hstateen3h", true},
    {0x643, "htval"},
    {0x644, "hip"},
    {0x645, "hvip"},
    {0x64a, "htinst"},
    {0x680, "hgatp"},
    {0x6a8, "hcontext"},
    {0x723, "mhpmevent3h", true},
    {0x724, "mhpmevent4h", true},
    {0x725, "mhpmevent5h", true},
    {0x726, "mhpmevent6h", true},
    {0x727, "mhpmevent7h", true},
    {0x728, "mhpmevent8h", true},
    {0x729, "mhpmevent9h", true},
    {0x72a, "mhpmevent10h", true},
    {0x72b, "mhpmevent11h", true},
    {0x72c, "mhpmevent12h", true},
    {0x72d, "mhpmevent13h", true},
    {0x72e, "mhpmevent14h", true},
    {0x72f, "mhpmevent15h", true},
    {0x730, "mhpmevent16h", true},
    {0x731, "mhpmevent17h", true},
    {0x732, "mhpmevent18h", true},
    {0x733, "mhpmevent19h", true},
    {0x734, "mhpmevent20h", true},
    {0x735, "mhpmevent21h", true},
    {0x736, "mhpmevent22h", true},
    {0x737, "mhpmevent23h", true},
    {0x738, "mhpmevent24h", true},
    {0x739, "mhpmevent25h", true},
    {0x73a, "mhpmevent26h", true},
    {0x73b, "mhpmevent27h", true},
    {0x73c, "mhpmevent28h", true},
    {0x73d, "mhpmevent29h", true},
    {0x73e, "mhpmevent30h", true},
    {0x73f, "mhpmevent31h", true},
    {0x740, "mnscratch"},
    {0x741, "mnepc"},
    {0x742, "mncause"},
    {0x744, "mnstatus"},
    {0x747, "mseccfg"},
    {0x757, "mseccfgh", true},
    {0x7a0, "tselect"},
    {0x7a1, "tdata1"},
    {0x7a2, "tdata2"},
    {0x7a3, "tdata3"},
    {0x7a8, "mcontext"},
    {0x7b0, "dcsr"},
    {0x7b1, "dpc"},
    {0x7b2, "dscratch0"},
    {0x7b3, "dscratch1"},
    {0xb00, "mcycle"},
    {0xb02, "minstret"},
    {0xb03, "mhpmcounter3"},
    {0xb04, "mhpmcounter4"},
    {0xb05, "mhpmcounter5"},
    {0xb06, "mhpmcounter6"},
    {0xb07, "mhpmcounter7"},
    {0xb08, "mhpmcounter8"},
    {0xb09, "mhpmcounter9"},
    {0xb0a, "mhpmcounter10"},
    {0xb0b, "mhpmcounter11"},
    {0xb0c, "mhpmcounter12"},
    {0xb0d, "mhpmcounter13"},
    {0xb0e, "mhpmcounter14"},
    {0xb0f, "mhpmcounter15"},
    {0xb10, "mhpmcounter16"},
    {0xb11, "mhpmcounter17"},
    {0xb12, "mhpmcounter18"},
    {0xb13, "mhpmcounter19"},
    {0xb14, "mhpmcounter20"},
    {0xb15, "mhpmcounter21"},
    {0xb16, "mhpmcounter22"},
    {0xb17, "mhpmcounter23"},
    {0xb18, "mhpmcounter24"},
    {0xb19, "mhpmcounter25"},
    {0xb1a, "mhpmcounter26"},
    {0xb1b, "mhpmcounter27"},
    {0xb1c, "mhpmcounter28"},
    {0xb1d, "mhpmcounter29"},
    {0xb1e, "mhpmcounter30"},
    {0xb1f, "mhpmcounter31"},
    {0xb80, "mcycleh", true},
    {0xb82, "minstreth", true},
    {0xb83, "mhpmcounter3h", true},
    {0xb84, "mhpmcounter4h", true},
    {0xb85, "mhpmcounter5h", true},
    {0xb86, "mhpmcounter6h", true},
    {0xb87, "mhpmcounter7h", true},
    {0xb88, "mhpmcounter8h", true},
    {0xb89, "mhpmcounter9h", true},
    {0xb8a, "mhpmcounter10h", true},
    {0xb8b, "mhpmcounter11h", true},
    {0xb8c, "mhpmcounter12h", true},
    {0xb8d, "mhpmcounter13h", true},
    {0xb8e, "mhpmcounter14h", true},
    {0xb8f, "mhpmcounter15h", true},
    {0xb90, "mhpmcounter16h", true},
    {0xb91, "mhpmcounter17h", true},
    {0xb92, "mhpmcounter18h", true},
    {0xb93, "mhpmcounter19h", true},
    {0xb94, "mhpmcounter20h", true},
    {0xb95, "mhpmcounter21h", true},
    {0xb96, "mhpmcounter22h", true},
    {0xb97, "mhpmcounter23h", true},
    {0xb98, "mhpmcounter24h", true},
    {0xb99, "mhpmcounter25h", true},
    {0xb9a, "mhpmcounter26h", true},
    {0xb9b, "mhpmcounter27h", true},
    {0xb9c, "mhpmcounter28h", true},
    {0xb9d, "mhpmcounter29h", true},
    {0xb9e, "mhpmcounter30h", true},
    {0xb9f, "mhpmcounter31h", true},
    {0xc00, "cycle"},
    {0xc01, "time"},
    {0xc02, "instret"},
    {0xc03, "hpmcounter3"},
    {0xc04, "hpmcounter4"},
    {0xc05, "hpmcounter5"},
    {0xc06, "hpmcounter6"},
    {0xc07, "hpmcounter7"},
    {0xc08, "hpmcounter8"},
    {0xc09, "hpmcounter9"},
    {0xc0a, "hpmcounter10"},
    {0xc0b, "hpmcounter11"},
    {0xc0c, "hpmcounter12"},
    {0xc0d, "hpmcounter13"},
    {0xc0e, "hpmcounter14"},
    {0xc0f, "hpmcounter15"},
    {0xc10, "hpmcounter16"},
    {0xc11, "hpmcounter17"},
    {0xc12, "hpmcounter18"},
    {0xc13, "hpmcounter19"},
    {0xc14, "hpmcounter20"},
    {0xc15, "hpmcounter21"},
    {0xc16, "hpmcounter22"},
    {0xc17, "hpmcounter23"},
    {0xc18, "hpmcounter24"},
    {0xc19, "hpmcounter25"},
    {0xc1a, "hpmcounter26"},
    {0xc1b, "hpmcounter27"},
    {0xc1c, "hpmcounter28"},
    {0xc1d, "hpmcounter29"},
    {0xc1e, "hpmcounter30"},
    {0xc1f, "hpmcounter31"},
    {0xc20, "vl"},
    {0xc21, "vtype"},
    {0xc22, "vlenb"},
    {0xc80, "cycleh", true},
    {0xc81, "timeh", true},
    {0xc82, "instreth", true},
    {0xc83, "hpmcounter3h", true},
    {0xc84, "hpmcounter4h", true},
    {0xc85, "hpmcounter5h", true},
    {0xc86, "hpmcounter6h", true},
    {0xc87, "hpmcounter7h", true},
    {0xc88, "hpmcounter8h", true},
    {0xc89, "hpmcounter9h", true},
    {0xc8a, "hpmcounter10h", true},
    {0xc8b, "hpmcounter11h", true},
    {0xc8c, "hpmcounter12h", true},
    {0xc8d, "hpmcounter13h", true},
    {0xc8e, "hpmcounter14h", true},
    {0xc8f, "hpmcounter15h", true},
    {0xc90, "hpmcounter16h", true},
    {0xc91, "hpmcounter17h", true},
    {0xc92, "hpmcounter18h", true},
    {0xc93, "hpmcounter19h", true},
    {0xc94, "hpmcounter20h", true},
    {0xc95, "hpmcounter21h", true},
    {0xc96, "hpmcounter22h", true},
    {0xc97, "hpmcounter23h", true},
    {0xc98, "hpmcounter24h", true},
    {0xc99, "hpmcounter25h", true},
    {0xc9a, "hpmcounter26h", true},
    {0xc9b, "hpmcounter27h", true},
    {0xc9c, "hpmcounter28h", true},
    {0xc9d, "hpmcounter29h", true},
    {0xc9e, "hpmcounter30h", true},
    {0xc9f, "hpmcounter31h", true},
    {0xda0, "scountovf"},
    {0xe12, "hgeip"},
    {0xf11, "mvendorid"},
    {0xf12, "marchid"},
    {0xf13, "mimpid"},
    {0xf14, "mhartid"},
    {0xf15, "mconfigptr"},
};

constexpr CSRName xqciint_names[] = {
    {0x7c0, "qc.mmcr"},          {0x7c3, "qc.mntvec"},
    {0x7c4, "qc.mstktopaddr"},   {0x7c5, "qc.mstkbottomaddr"},
    {0x7c8, "qc.mthreadptr"},    {0x7c9, "qc.mcause"},
    {0x7d0, "qc.mwpstartaddr0"}, {0x7d1, "qc.mwpstartaddr1"},
    {0x7d2, "qc.mwpstartaddr2"}, {0x7d3, "qc.mwpstartaddr3"},
    {0x7d4, "qc.mwpendaddr0"},   {0x7d5, "qc.mwpendaddr1"},
    {0x7d6, "qc.mwpendaddr2"},   {0x7d7, "qc.mwpendaddr3"},
    {0x7f0, "qc.mclicip0"},      {0x7f1, "qc.mclicip1"},
    {0x7f2, "qc.mclicip2"},      {0x7f3, "qc.mclicip3"},
    {0x7f4, "qc.mclicip4"},      {0x7f5, "qc.mclicip5"},
    {0x7f6, "qc.mclicip6"},      {0x7f7, "qc.mclicip7"},
    {0x7f8, "qc.mclicie0"},      {0x7f9, "qc.mclicie1"},
    {0x7fa, "qc.mclicie2"},      {0x7fb, "qc.mclicie3"},
    {0x7fc, "qc.mclicie4"},      {0x7fd, "qc.mclicie5"},
    {0x7fe, "qc.mclicie6"},      {0x7ff, "qc.mclicie7"},
    {0xbc0, "qc.mclicilvl00"},   {0xbc1, "qc.mclicilvl01"},
    {0xbc2, "qc.mclicilvl02"},   {0xbc3, "qc.mclicilvl03"},
    {0xbc4, "qc.mclicilvl04"},   {0xbc5, "qc.mclicilvl05"},
    {0xbc6, "qc.mclicilvl06"},   {0xbc7, "qc.mclicilvl07"},
    {0xbc8, "qc.mclicilvl08"},   {0xbc9, "qc.mclicilvl09"},
    {0xbca, "qc.mclicilvl10"},   {0xbcb, "qc.mclicilvl11"},
    {0xbcc, "qc.mclicilvl12"},   {0xbcd, "qc.mclicilvl13"},
    {0xbce, "qc.mclicilvl14"},   {0xbcf, "qc.mclicilvl15"},
    {0xbd0, "qc.mclicilvl16"},   {0xbd1, "qc.mclicilvl17"},
    {0xbd2, "qc.mclicilvl18"},   {0xbd3, "qc.mclicilvl19"},
    {0xbd4, "qc.mclicilvl20"},   {0xbd5, "qc.mclicilvl21"},
    {0xbd6, "qc.mclicilvl22"},   {0xbd7, "qc.mclicilvl23"},
    {0xbd8, "qc.mclicilvl24"},   {0xbd9, "qc.mclicilvl25"},
    {0xbda, "qc.mclicilvl26"},   {0xbdb, "qc.mclicilvl27"},
    {0xbdc, "qc.mclicilvl28"},   {0xbdd, "qc.mclicilvl29"},
    {0xbde, "qc.mclicilvl30"},   {0xbdf, "qc.mclicilvl31"},
};

} // namespace

std::optional<RegisterDescriptor>
RegisterDescription_riscv::CreateCSRDescriptor(
    uint32_t address, uint32_t byte_size,
    llvm::ArrayRef<std::string> features) {
  if (address > id::csr_last - id::csr_first)
    return std::nullopt;

  assert(byte_size == 4 || byte_size == 8);

  RegisterDescriptor csr;
  csr.id = id::csr_first + address;
  csr.name = llvm::formatv("csr_{0:x}", address).str();
  csr.alt_name = csr.name;
  csr.byte_size = byte_size;
  csr.numbers[lldb::eRegisterKindLLDB] = csr.id;
  csr.numbers[lldb::eRegisterKindDWARF] = dwarf::first_csr + address;
  csr.numbers[lldb::eRegisterKindEHFrame] = ehframe::first_csr + address;

  for (const auto &entry : csr_names) {
    if (entry.address == address && (!entry.rv32_only || byte_size == 4)) {
      csr.name = entry.name;
      csr.alt_name = llvm::formatv("csr_{0:x3}", address).str();
      break;
    }
  }

  if (byte_size == 4 && llvm::is_contained(features, "xqciint")) {
    for (const auto &entry : xqciint_names) {
      if (entry.address == address) {
        csr.name = entry.name;
        csr.alt_name = llvm::formatv("csr_{0:x3}", address).str();
        break;
      }
    }
  }

  return csr;
}

RegisterDescription_riscv::RegisterDescription_riscv(
    llvm::ArrayRef<RegisterDescriptor> definitions,
    llvm::ArrayRef<RegisterID> gpr, llvm::ArrayRef<RegisterID> fpr,
    llvm::ArrayRef<RegisterID> vector, Flags features,
    llvm::ArrayRef<uint32_t> csr_addresses, uint32_t byte_size,
    llvm::ArrayRef<std::string> extensions) {
  std::vector<RegisterGroup> groups = {
      MakeRegisterGroup("General Purpose Registers", "gpr", gpr),
  };

  if (features.AnySet(eRegsetMaskFP))
    groups.push_back(MakeRegisterGroup("Floating Point Registers", "fpr", fpr));

  if (features.AnySet(eRegsetMaskVector))
    groups.push_back(MakeRegisterGroup("Vector Registers", "vpr", vector));

  auto values = SelectRegisterDefinitions(definitions, groups);

  RegisterGroup csrs{"Control and Status Registers", "csr", {}};
  std::set<uint32_t> seen;

  for (uint32_t address : csr_addresses) {
    if (address > id::csr_last - id::csr_first || !seen.insert(address).second)
      continue;

    if (auto csr = CreateCSRDescriptor(address, byte_size, extensions)) {
      csrs.registers.push_back(csr->id);
      values.push_back(std::move(*csr));
    }
  }

  if (!csrs.registers.empty())
    groups.push_back(std::move(csrs));

  RegisterDescription::operator=(llvm::cantFail(
      RegisterDescription::Create(std::move(values), std::move(groups))));
}
