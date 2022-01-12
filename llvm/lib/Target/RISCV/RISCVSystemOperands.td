//===- RISCVSystemOperands.td ------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the symbolic operands permitted for various kinds of
// RISC-V system instruction.
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

//===----------------------------------------------------------------------===//
// CSR (control and status register read/write) instruction options.
//===----------------------------------------------------------------------===//

class SysReg<string name, bits<12> op> {
  string Name = name;
  // A maximum of one alias is supported right now.
  string AltName = name;
  // A maximum of one deprecated name is supported right now.  Unlike the
  // `AltName` alias, a `DeprecatedName` generates a diagnostic when the name is
  // used to encourage software to migrate away from the name.
  string DeprecatedName = "";
  bits<12> Encoding = op;
  // FIXME: add these additional fields when needed.
  // Privilege Access: Read and Write = 0, 1, 2; Read-Only = 3.
  // Privilege Mode: User = 0, System = 1 or Machine = 3.
  // bits<2> ReadWrite = op{11 - 10};
  // bits<2> XMode = op{9 - 8};
  // Check Extra field name and what bits 7-6 correspond to.
  // bits<2> Extra = op{7 - 6};
  // Register number without the privilege bits.
  // bits<6> Number = op{5 - 0};
  code FeaturesRequired = [{ {} }];
  bit isRV32Only = 0;
}

def SysRegsList : GenericTable {
  let FilterClass = "SysReg";
  // FIXME: add "ReadWrite", "Mode", "Extra", "Number" fields when needed.
  let Fields = [
    "Name", "AltName", "DeprecatedName", "Encoding", "FeaturesRequired",
    "isRV32Only",
  ];

  let PrimaryKey = [ "Encoding" ];
  let PrimaryKeyName = "lookupSysRegByEncoding";
}

def lookupSysRegByName : SearchIndex {
  let Table = SysRegsList;
  let Key = [ "Name" ];
}

def lookupSysRegByAltName : SearchIndex {
  let Table = SysRegsList;
  let Key = [ "AltName" ];
}

def lookupSysRegByDeprecatedName : SearchIndex {
  let Table = SysRegsList;
  let Key = [ "DeprecatedName" ];
}

// The following CSR encodings match those given in Tables 2.2,
// 2.3, 2.4 and  2.5 in the RISC-V Instruction Set Manual
// Volume II: Privileged Architecture.

//===----------------------------------------------------------------------===//
// User Trap Setup
//===----------------------------------------------------------------------===//
def : SysReg<"ustatus", 0x000>;
def : SysReg<"uie", 0x004>;
def : SysReg<"utvec", 0x005>;

//===----------------------------------------------------------------------===//
// User Trap Handling
//===----------------------------------------------------------------------===//
def : SysReg<"uscratch", 0x040>;
def : SysReg<"uepc", 0x041>;
def : SysReg<"ucause", 0x042>;
let DeprecatedName = "ubadaddr" in
def : SysReg<"utval", 0x043>;
def : SysReg<"uip", 0x044>;

//===----------------------------------------------------------------------===//
// User Floating-Point CSRs
//===----------------------------------------------------------------------===//

def SysRegFFLAGS : SysReg<"fflags", 0x001>;
def SysRegFRM    : SysReg<"frm", 0x002>;
def SysRegFCSR   : SysReg<"fcsr", 0x003>;

//===----------------------------------------------------------------------===//
// User Counter/Timers
//===----------------------------------------------------------------------===//
def CYCLE   : SysReg<"cycle", 0xC00>;
def TIME    : SysReg<"time", 0xC01>;
def INSTRET : SysReg<"instret", 0xC02>;

def : SysReg<"hpmcounter3", 0xC03>;
def : SysReg<"hpmcounter4", 0xC04>;
def : SysReg<"hpmcounter5", 0xC05>;
def : SysReg<"hpmcounter6", 0xC06>;
def : SysReg<"hpmcounter7", 0xC07>;
def : SysReg<"hpmcounter8", 0xC08>;
def : SysReg<"hpmcounter9", 0xC09>;
def : SysReg<"hpmcounter10", 0xC0A>;
def : SysReg<"hpmcounter11", 0xC0B>;
def : SysReg<"hpmcounter12", 0xC0C>;
def : SysReg<"hpmcounter13", 0xC0D>;
def : SysReg<"hpmcounter14", 0xC0E>;
def : SysReg<"hpmcounter15", 0xC0F>;
def : SysReg<"hpmcounter16", 0xC10>;
def : SysReg<"hpmcounter17", 0xC11>;
def : SysReg<"hpmcounter18", 0xC12>;
def : SysReg<"hpmcounter19", 0xC13>;
def : SysReg<"hpmcounter20", 0xC14>;
def : SysReg<"hpmcounter21", 0xC15>;
def : SysReg<"hpmcounter22", 0xC16>;
def : SysReg<"hpmcounter23", 0xC17>;
def : SysReg<"hpmcounter24", 0xC18>;
def : SysReg<"hpmcounter25", 0xC19>;
def : SysReg<"hpmcounter26", 0xC1A>;
def : SysReg<"hpmcounter27", 0xC1B>;
def : SysReg<"hpmcounter28", 0xC1C>;
def : SysReg<"hpmcounter29", 0xC1D>;
def : SysReg<"hpmcounter30", 0xC1E>;
def : SysReg<"hpmcounter31", 0xC1F>;

let isRV32Only = 1 in {
def CYCLEH   : SysReg<"cycleh", 0xC80>;
def TIMEH    : SysReg<"timeh", 0xC81>;
def INSTRETH : SysReg<"instreth", 0xC82>;

def: SysReg<"hpmcounter3h", 0xC83>;
def: SysReg<"hpmcounter4h", 0xC84>;
def: SysReg<"hpmcounter5h", 0xC85>;
def: SysReg<"hpmcounter6h", 0xC86>;
def: SysReg<"hpmcounter7h", 0xC87>;
def: SysReg<"hpmcounter8h", 0xC88>;
def: SysReg<"hpmcounter9h", 0xC89>;
def: SysReg<"hpmcounter10h", 0xC8A>;
def: SysReg<"hpmcounter11h", 0xC8B>;
def: SysReg<"hpmcounter12h", 0xC8C>;
def: SysReg<"hpmcounter13h", 0xC8D>;
def: SysReg<"hpmcounter14h", 0xC8E>;
def: SysReg<"hpmcounter15h", 0xC8F>;
def: SysReg<"hpmcounter16h", 0xC90>;
def: SysReg<"hpmcounter17h", 0xC91>;
def: SysReg<"hpmcounter18h", 0xC92>;
def: SysReg<"hpmcounter19h", 0xC93>;
def: SysReg<"hpmcounter20h", 0xC94>;
def: SysReg<"hpmcounter21h", 0xC95>;
def: SysReg<"hpmcounter22h", 0xC96>;
def: SysReg<"hpmcounter23h", 0xC97>;
def: SysReg<"hpmcounter24h", 0xC98>;
def: SysReg<"hpmcounter25h", 0xC99>;
def: SysReg<"hpmcounter26h", 0xC9A>;
def: SysReg<"hpmcounter27h", 0xC9B>;
def: SysReg<"hpmcounter28h", 0xC9C>;
def: SysReg<"hpmcounter29h", 0xC9D>;
def: SysReg<"hpmcounter30h", 0xC9E>;
def: SysReg<"hpmcounter31h", 0xC9F>;
}

//===----------------------------------------------------------------------===//
// Supervisor Trap Setup
//===----------------------------------------------------------------------===//
def : SysReg<"sstatus", 0x100>;
def : SysReg<"sedeleg", 0x102>;
def : SysReg<"sideleg", 0x103>;
def : SysReg<"sie", 0x104>;
def : SysReg<"stvec", 0x105>;
def : SysReg<"scounteren", 0x106>;

//===----------------------------------------------------------------------===//
// Supervisor Configuration
//===----------------------------------------------------------------------===//

def : SysReg<"senvcfg", 0x10A>;

//===----------------------------------------------------------------------===//
// Supervisor Trap Handling
//===----------------------------------------------------------------------===//
def : SysReg<"sscratch", 0x140>;
def : SysReg<"sepc", 0x141>;
def : SysReg<"scause", 0x142>;
let DeprecatedName = "sbadaddr" in
def : SysReg<"stval", 0x143>;
def : SysReg<"sip", 0x144>;

//===----------------------------------------------------------------------===//
// Supervisor Protection and Translation
//===----------------------------------------------------------------------===//
let DeprecatedName = "sptbr" in
def : SysReg<"satp", 0x180>;

//===----------------------------------------------------------------------===//
// Debug/Trace Registers
//===----------------------------------------------------------------------===//

def : SysReg<"scontext", 0x5A8>;

//===----------------------------------------------------------------------===//
// Hypervisor Trap Setup
//===----------------------------------------------------------------------===//

def : SysReg<"hstatus", 0x600>;
def : SysReg<"hedeleg", 0x602>;
def : SysReg<"hideleg", 0x603>;
def : SysReg<"hie", 0x604>;
def : SysReg<"hcounteren", 0x606>;
def : SysReg<"hgeie", 0x607>;

//===----------------------------------------------------------------------===//
// Hypervisor Trap Handling
//===----------------------------------------------------------------------===//

def : SysReg<"htval", 0x643>;
def : SysReg<"hip", 0x644>;
def : SysReg<"hvip", 0x645>;
def : SysReg<"htinst", 0x64A>;
def : SysReg<"hgeip", 0xE12>;

//===----------------------------------------------------------------------===//
// Hypervisor Configuration
//===----------------------------------------------------------------------===//

def : SysReg<"henvcfg", 0x60A>;
let isRV32Only = 1 in
def : SysReg<"henvcfgh", 0x61A>;

//===----------------------------------------------------------------------===//
// Hypervisor Protection and Translation
//===----------------------------------------------------------------------===//

def : SysReg<"hgatp", 0x680>;

//===----------------------------------------------------------------------===//
// Debug/Trace Registers
//===----------------------------------------------------------------------===//

def : SysReg<"hcontext", 0x6A8>;

//===----------------------------------------------------------------------===//
// Hypervisor Counter/Timer Virtualization Registers
//===----------------------------------------------------------------------===//

def : SysReg<"htimedelta", 0x605>;
let isRV32Only = 1 in
def : SysReg<"htimedeltah", 0x615>;

//===----------------------------------------------------------------------===//
// Virtual Supervisor Registers
//===----------------------------------------------------------------------===//

def : SysReg<"vsstatus", 0x200>;
def : SysReg<"vsie", 0x204>;
def : SysReg<"vstvec", 0x205>;
def : SysReg<"vsscratch", 0x240>;
def : SysReg<"vsepc", 0x241>;
def : SysReg<"vscause", 0x242>;
def : SysReg<"vstval", 0x243>;
def : SysReg<"vsip", 0x244>;
def : SysReg<"vsatp", 0x280>;

//===----------------------------------------------------------------------===//
// Machine Information Registers
//===----------------------------------------------------------------------===//

def : SysReg<"mvendorid", 0xF11>;
def : SysReg<"marchid", 0xF12>;
def : SysReg<"mimpid", 0xF13>;
def : SysReg<"mhartid", 0xF14>;
def : SysReg<"mconfigptr", 0xF15>;

//===----------------------------------------------------------------------===//
// Machine Trap Setup
//===----------------------------------------------------------------------===//
def : SysReg<"mstatus", 0x300>;
def : SysReg<"misa", 0x301>;
def : SysReg<"medeleg", 0x302>;
def : SysReg<"mideleg", 0x303>;
def : SysReg<"mie", 0x304>;
def : SysReg<"mtvec", 0x305>;
def : SysReg<"mcounteren", 0x306>;
let isRV32Only = 1 in
def : SysReg<"mstatush", 0x310>;

//===----------------------------------------------------------------------===//
// Machine Trap Handling
//===----------------------------------------------------------------------===//
def : SysReg<"mscratch", 0x340>;
def : SysReg<"mepc", 0x341>;
def : SysReg<"mcause", 0x342>;
let DeprecatedName = "mbadaddr" in
def : SysReg<"mtval", 0x343>;
def : SysReg<"mip", 0x344>;
def : SysReg<"mtinst", 0x34A>;
def : SysReg<"mtval2", 0x34B>;

//===----------------------------------------------------------------------===//
// Machine Configuration
//===----------------------------------------------------------------------===//

def : SysReg<"menvcfg", 0x30A>;
let isRV32Only = 1 in
def : SysReg<"menvcfgh", 0x31A>;
def : SysReg<"mseccfg", 0x747>;
let isRV32Only = 1 in
def : SysReg<"mseccfgh", 0x757>;

//===----------------------------------------------------------------------===//
// Machine Protection and Translation
//===----------------------------------------------------------------------===//
def : SysReg<"pmpcfg0", 0x3A0>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg1", 0x3A1>;
def : SysReg<"pmpcfg2", 0x3A2>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg3", 0x3A3>;
def : SysReg<"pmpcfg4", 0x3A4>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg5", 0x3A5>;
def : SysReg<"pmpcfg6", 0x3A6>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg7", 0x3A7>;
def : SysReg<"pmpcfg8", 0x3A8>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg9", 0x3A9>;
def : SysReg<"pmpcfg10", 0x3AA>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg11", 0x3AB>;
def : SysReg<"pmpcfg12", 0x3AC>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg13", 0x3AD>;
def : SysReg<"pmpcfg14", 0x3AE>;
let isRV32Only = 1 in
def : SysReg<"pmpcfg15", 0x3AF>;

def : SysReg<"pmpaddr0", 0x3B0>;
def : SysReg<"pmpaddr1", 0x3B1>;
def : SysReg<"pmpaddr2", 0x3B2>;
def : SysReg<"pmpaddr3", 0x3B3>;
def : SysReg<"pmpaddr4", 0x3B4>;
def : SysReg<"pmpaddr5", 0x3B5>;
def : SysReg<"pmpaddr6", 0x3B6>;
def : SysReg<"pmpaddr7", 0x3B7>;
def : SysReg<"pmpaddr8", 0x3B8>;
def : SysReg<"pmpaddr9", 0x3B9>;
def : SysReg<"pmpaddr10", 0x3BA>;
def : SysReg<"pmpaddr11", 0x3BB>;
def : SysReg<"pmpaddr12", 0x3BC>;
def : SysReg<"pmpaddr13", 0x3BD>;
def : SysReg<"pmpaddr14", 0x3BE>;
def : SysReg<"pmpaddr15", 0x3BF>;
def : SysReg<"pmpaddr16", 0x3C0>;
def : SysReg<"pmpaddr17", 0x3C1>;
def : SysReg<"pmpaddr18", 0x3C2>;
def : SysReg<"pmpaddr19", 0x3C3>;
def : SysReg<"pmpaddr20", 0x3C4>;
def : SysReg<"pmpaddr21", 0x3C5>;
def : SysReg<"pmpaddr22", 0x3C6>;
def : SysReg<"pmpaddr23", 0x3C7>;
def : SysReg<"pmpaddr24", 0x3C8>;
def : SysReg<"pmpaddr25", 0x3C9>;
def : SysReg<"pmpaddr26", 0x3CA>;
def : SysReg<"pmpaddr27", 0x3CB>;
def : SysReg<"pmpaddr28", 0x3CC>;
def : SysReg<"pmpaddr29", 0x3CD>;
def : SysReg<"pmpaddr30", 0x3CE>;
def : SysReg<"pmpaddr31", 0x3CF>;
def : SysReg<"pmpaddr32", 0x3D0>;
def : SysReg<"pmpaddr33", 0x3D1>;
def : SysReg<"pmpaddr34", 0x3D2>;
def : SysReg<"pmpaddr35", 0x3D3>;
def : SysReg<"pmpaddr36", 0x3D4>;
def : SysReg<"pmpaddr37", 0x3D5>;
def : SysReg<"pmpaddr38", 0x3D6>;
def : SysReg<"pmpaddr39", 0x3D7>;
def : SysReg<"pmpaddr40", 0x3D8>;
def : SysReg<"pmpaddr41", 0x3D9>;
def : SysReg<"pmpaddr42", 0x3DA>;
def : SysReg<"pmpaddr43", 0x3DB>;
def : SysReg<"pmpaddr44", 0x3DC>;
def : SysReg<"pmpaddr45", 0x3DD>;
def : SysReg<"pmpaddr46", 0x3DE>;
def : SysReg<"pmpaddr47", 0x3DF>;
def : SysReg<"pmpaddr48", 0x3E0>;
def : SysReg<"pmpaddr49", 0x3E1>;
def : SysReg<"pmpaddr50", 0x3E2>;
def : SysReg<"pmpaddr51", 0x3E3>;
def : SysReg<"pmpaddr52", 0x3E4>;
def : SysReg<"pmpaddr53", 0x3E5>;
def : SysReg<"pmpaddr54", 0x3E6>;
def : SysReg<"pmpaddr55", 0x3E7>;
def : SysReg<"pmpaddr56", 0x3E8>;
def : SysReg<"pmpaddr57", 0x3E9>;
def : SysReg<"pmpaddr58", 0x3EA>;
def : SysReg<"pmpaddr59", 0x3EB>;
def : SysReg<"pmpaddr60", 0x3EC>;
def : SysReg<"pmpaddr61", 0x3ED>;
def : SysReg<"pmpaddr62", 0x3EE>;
def : SysReg<"pmpaddr63", 0x3EF>;


//===----------------------------------------------------------------------===//
// Machine Counter and Timers
//===----------------------------------------------------------------------===//
def : SysReg<"mcycle", 0xB00>;
def : SysReg<"minstret", 0xB02>;

def : SysReg<"mhpmcounter3", 0xB03>;
def : SysReg<"mhpmcounter4", 0xB04>;
def : SysReg<"mhpmcounter5", 0xB05>;
def : SysReg<"mhpmcounter6", 0xB06>;
def : SysReg<"mhpmcounter7", 0xB07>;
def : SysReg<"mhpmcounter8", 0xB08>;
def : SysReg<"mhpmcounter9", 0xB09>;
def : SysReg<"mhpmcounter10", 0xB0A>;
def : SysReg<"mhpmcounter11", 0xB0B>;
def : SysReg<"mhpmcounter12", 0xB0C>;
def : SysReg<"mhpmcounter13", 0xB0D>;
def : SysReg<"mhpmcounter14", 0xB0E>;
def : SysReg<"mhpmcounter15", 0xB0F>;
def : SysReg<"mhpmcounter16", 0xB10>;
def : SysReg<"mhpmcounter17", 0xB11>;
def : SysReg<"mhpmcounter18", 0xB12>;
def : SysReg<"mhpmcounter19", 0xB13>;
def : SysReg<"mhpmcounter20", 0xB14>;
def : SysReg<"mhpmcounter21", 0xB15>;
def : SysReg<"mhpmcounter22", 0xB16>;
def : SysReg<"mhpmcounter23", 0xB17>;
def : SysReg<"mhpmcounter24", 0xB18>;
def : SysReg<"mhpmcounter25", 0xB19>;
def : SysReg<"mhpmcounter26", 0xB1A>;
def : SysReg<"mhpmcounter27", 0xB1B>;
def : SysReg<"mhpmcounter28", 0xB1C>;
def : SysReg<"mhpmcounter29", 0xB1D>;
def : SysReg<"mhpmcounter30", 0xB1E>;
def : SysReg<"mhpmcounter31", 0xB1F>;

let isRV32Only = 1 in {
def: SysReg<"mcycleh", 0xB80>;
def: SysReg<"minstreth", 0xB82>;

def: SysReg<"mhpmcounter3h", 0xB83>;
def: SysReg<"mhpmcounter4h", 0xB84>;
def: SysReg<"mhpmcounter5h", 0xB85>;
def: SysReg<"mhpmcounter6h", 0xB86>;
def: SysReg<"mhpmcounter7h", 0xB87>;
def: SysReg<"mhpmcounter8h", 0xB88>;
def: SysReg<"mhpmcounter9h", 0xB89>;
def: SysReg<"mhpmcounter10h", 0xB8A>;
def: SysReg<"mhpmcounter11h", 0xB8B>;
def: SysReg<"mhpmcounter12h", 0xB8C>;
def: SysReg<"mhpmcounter13h", 0xB8D>;
def: SysReg<"mhpmcounter14h", 0xB8E>;
def: SysReg<"mhpmcounter15h", 0xB8F>;
def: SysReg<"mhpmcounter16h", 0xB90>;
def: SysReg<"mhpmcounter17h", 0xB91>;
def: SysReg<"mhpmcounter18h", 0xB92>;
def: SysReg<"mhpmcounter19h", 0xB93>;
def: SysReg<"mhpmcounter20h", 0xB94>;
def: SysReg<"mhpmcounter21h", 0xB95>;
def: SysReg<"mhpmcounter22h", 0xB96>;
def: SysReg<"mhpmcounter23h", 0xB97>;
def: SysReg<"mhpmcounter24h", 0xB98>;
def: SysReg<"mhpmcounter25h", 0xB99>;
def: SysReg<"mhpmcounter26h", 0xB9A>;
def: SysReg<"mhpmcounter27h", 0xB9B>;
def: SysReg<"mhpmcounter28h", 0xB9C>;
def: SysReg<"mhpmcounter29h", 0xB9D>;
def: SysReg<"mhpmcounter30h", 0xB9E>;
def: SysReg<"mhpmcounter31h", 0xB9F>;
}

//===----------------------------------------------------------------------===//
// Machine Counter Setup
//===----------------------------------------------------------------------===//
let AltName = "mucounteren" in // Privileged spec v1.9.1 Name
def : SysReg<"mcountinhibit", 0x320>;

def : SysReg<"mhpmevent3", 0x323>;
def : SysReg<"mhpmevent4", 0x324>;
def : SysReg<"mhpmevent5", 0x325>;
def : SysReg<"mhpmevent6", 0x326>;
def : SysReg<"mhpmevent7", 0x327>;
def : SysReg<"mhpmevent8", 0x328>;
def : SysReg<"mhpmevent9", 0x329>;
def : SysReg<"mhpmevent10", 0x32A>;
def : SysReg<"mhpmevent11", 0x32B>;
def : SysReg<"mhpmevent12", 0x32C>;
def : SysReg<"mhpmevent13", 0x32D>;
def : SysReg<"mhpmevent14", 0x32E>;
def : SysReg<"mhpmevent15", 0x32F>;
def : SysReg<"mhpmevent16", 0x330>;
def : SysReg<"mhpmevent17", 0x331>;
def : SysReg<"mhpmevent18", 0x332>;
def : SysReg<"mhpmevent19", 0x333>;
def : SysReg<"mhpmevent20", 0x334>;
def : SysReg<"mhpmevent21", 0x335>;
def : SysReg<"mhpmevent22", 0x336>;
def : SysReg<"mhpmevent23", 0x337>;
def : SysReg<"mhpmevent24", 0x338>;
def : SysReg<"mhpmevent25", 0x339>;
def : SysReg<"mhpmevent26", 0x33A>;
def : SysReg<"mhpmevent27", 0x33B>;
def : SysReg<"mhpmevent28", 0x33C>;
def : SysReg<"mhpmevent29", 0x33D>;
def : SysReg<"mhpmevent30", 0x33E>;
def : SysReg<"mhpmevent31", 0x33F>;

//===----------------------------------------------------------------------===//
// Debug/ Trace Registers (shared with Debug Mode)
//===----------------------------------------------------------------------===//
def : SysReg<"tselect", 0x7A0>;
def : SysReg<"tdata1", 0x7A1>;
def : SysReg<"tdata2", 0x7A2>;
def : SysReg<"tdata3", 0x7A3>;
def : SysReg<"mcontext", 0x7A8>;

//===----------------------------------------------------------------------===//
// Debug Mode Registers
//===----------------------------------------------------------------------===//
def : SysReg<"dcsr", 0x7B0>;
def : SysReg<"dpc", 0x7B1>;

// "dscratch" is an alternative name for "dscratch0" which appeared in earlier
// drafts of the RISC-V debug spec
let AltName = "dscratch" in
def : SysReg<"dscratch0", 0x7B2>;
def : SysReg<"dscratch1", 0x7B3>;

//===----------------------------------------------------------------------===//
// User Vector CSRs
//===----------------------------------------------------------------------===//
def : SysReg<"vstart", 0x008>;
def : SysReg<"vxsat", 0x009>;
def : SysReg<"vxrm", 0x00A>;
def : SysReg<"vcsr", 0x00F>;
def : SysReg<"vl", 0xC20>;
def : SysReg<"vtype", 0xC21>;
def SysRegVLENB: SysReg<"vlenb", 0xC22>;
