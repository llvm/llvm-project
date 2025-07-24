//===- AMDGPUMIRFormatter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of AMDGPU overrides of MIRFormatter.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMIRFormatter.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"

using namespace llvm;

bool parseAtomicOrdering(StringRef Src, unsigned &Order) {
  Src.consume_front(".");
  for (unsigned I = 0; I <= (unsigned)AtomicOrdering::LAST; ++I) {
    if (Src == toIRString((AtomicOrdering)I)) {
      Order = I;
      return true;
    }
  }
  Order = ~0u;
  return false;
}

static const char *fmtScope(unsigned Scope) {
  static const char *Names[] = {"none",      "singlethread", "wavefront",
                                "workgroup", "agent",        "system"};
  return Names[Scope];
}

bool parseAtomicScope(StringRef Src, unsigned &Scope) {
  Src.consume_front(".");
  for (unsigned I = 0;
       I != (unsigned)AMDGPU::SIAtomicScope::NUM_SI_ATOMIC_SCOPES; ++I) {
    if (Src == fmtScope(I)) {
      Scope = I;
      return true;
    }
  }
  Scope = ~0u;
  return false;
}

static const char *fmtAddrSpace(unsigned Space) {
  static const char *Names[] = {"none",    "global", "lds",
                                "scratch", "gds",    "other"};
  return Names[Space];
}

bool parseOneAddrSpace(StringRef Src, unsigned &AddrSpace) {
  if (Src == "none") {
    AddrSpace = (unsigned)AMDGPU::SIAtomicAddrSpace::NONE;
    return true;
  }
  if (Src == "flat") {
    AddrSpace = (unsigned)AMDGPU::SIAtomicAddrSpace::FLAT;
    return true;
  }
  if (Src == "atomic") {
    AddrSpace = (unsigned)AMDGPU::SIAtomicAddrSpace::ATOMIC;
    return true;
  }
  if (Src == "all") {
    AddrSpace = (unsigned)AMDGPU::SIAtomicAddrSpace::ALL;
    return true;
  }
  for (unsigned I = 1, A = 1; A <= (unsigned)AMDGPU::SIAtomicAddrSpace::LAST;
       A <<= 1, ++I) {
    if (Src == fmtAddrSpace(I)) {
      AddrSpace = A;
      return true;
    }
  }
  AddrSpace = ~0u;
  return false;
}

bool parseAddrSpace(StringRef Src, unsigned &AddrSpace) {
  Src = Src.trim();
  Src.consume_front(".");
  while (!Src.empty()) {
    auto [First, Rest] = Src.split('.');
    unsigned OneSpace;
    if (!parseOneAddrSpace(First, OneSpace))
      return false;
    AddrSpace |= OneSpace;
    Src = Rest;
  }
  return true;
}

static void fmtAddrSpace(raw_ostream &OS, int64_t Imm) {
  OS << '.';
  if (Imm == (unsigned)AMDGPU::SIAtomicAddrSpace::NONE) {
    OS << "none";
    return;
  }
  if (Imm == (unsigned)AMDGPU::SIAtomicAddrSpace::FLAT) {
    OS << "flat";
    return;
  }
  if (Imm == (unsigned)AMDGPU::SIAtomicAddrSpace::ATOMIC) {
    OS << "atomic";
    return;
  }
  if (Imm == (unsigned)AMDGPU::SIAtomicAddrSpace::ALL) {
    OS << "all";
    return;
  }

  ListSeparator LS{"."};
  auto AddrSpace = (AMDGPU::SIAtomicAddrSpace)Imm;
  const auto LAST = (unsigned)AMDGPU::SIAtomicAddrSpace::LAST;

  for (unsigned A = 1, I = 1; A <= LAST; A <<= 1, ++I) {
    if (any(AddrSpace & (AMDGPU::SIAtomicAddrSpace)A))
      OS << LS << StringRef(fmtAddrSpace(I));
  }
}

static void printFenceOperand(raw_ostream &OS, const MachineInstr &MI,
                              std::optional<unsigned int> OpIdx, int64_t Imm) {
#define GET_IDX(Name)                                                          \
  AMDGPU::getNamedOperandIdx(AMDGPU::S_WAITCNT_FENCE_soft, AMDGPU::OpName::Name)
  if (OpIdx == GET_IDX(Ordering)) {
    assert(Imm <= (unsigned)AtomicOrdering::LAST);
    OS << '.' << StringRef(toIRString((AtomicOrdering)Imm));
  } else if (OpIdx == GET_IDX(Scope)) {
    assert(Imm < (unsigned)AMDGPU::SIAtomicScope::NUM_SI_ATOMIC_SCOPES);
    OS << '.' << StringRef(fmtScope(Imm));
  } else if (OpIdx == GET_IDX(AddrSpace)) {
    fmtAddrSpace(OS, Imm);
  }
#undef GET_IDX
}

void AMDGPUMIRFormatter::printImm(raw_ostream &OS, const MachineInstr &MI,
                      std::optional<unsigned int> OpIdx, int64_t Imm) const {

  switch (MI.getOpcode()) {
  case AMDGPU::S_DELAY_ALU:
    assert(OpIdx == 0);
    printSDelayAluImm(Imm, OS);
    break;
  case AMDGPU::S_WAITCNT_FENCE_soft:
    printFenceOperand(OS, MI, OpIdx, Imm);
    break;
  default:
    MIRFormatter::printImm(OS, MI, OpIdx, Imm);
    break;
  }
}

static bool
parseFenceParameter(const unsigned int OpIdx, int64_t &Imm,
                    llvm::StringRef &Src,
                    llvm::MIRFormatter::ErrorCallbackType &ErrorCallback) {
#define GET_IDX(Name)                                                          \
  AMDGPU::getNamedOperandIdx(AMDGPU::S_WAITCNT_FENCE_soft, AMDGPU::OpName::Name)
  if (OpIdx == (unsigned)GET_IDX(Ordering)) {
    unsigned Order = 0;
    if (!parseAtomicOrdering(Src, Order))
      return ErrorCallback(Src.begin(), "Expected atomic ordering");
    Imm = Order;
    return false;
  }
  if (OpIdx == (unsigned)GET_IDX(Scope)) {
    unsigned Scope = 0;
    if (!parseAtomicScope(Src, Scope))
      return ErrorCallback(Src.begin(), "Expected atomic scope");
    Imm = Scope;
    return false;
  }
  if (OpIdx == (unsigned)GET_IDX(AddrSpace)) {
    unsigned AddrSpace = 0;
    if (!parseAddrSpace(Src, AddrSpace))
      return ErrorCallback(Src.begin(), "Expected address space");
    Imm = AddrSpace;
    return false;
  }
  return true;
#undef GET_IDX
}

/// Implement target specific parsing of immediate mnemonics. The mnemonic is
/// a string with a leading dot.
bool AMDGPUMIRFormatter::parseImmMnemonic(const unsigned OpCode,
                              const unsigned OpIdx,
                              StringRef Src, int64_t &Imm,
                              ErrorCallbackType ErrorCallback) const
{

  switch (OpCode) {
  case AMDGPU::S_DELAY_ALU:
    return parseSDelayAluImmMnemonic(OpIdx, Imm, Src, ErrorCallback);
  case AMDGPU::S_WAITCNT_FENCE_soft:
    return parseFenceParameter(OpIdx, Imm, Src, ErrorCallback);
  default:
    break;
  }
  return true; // Don't know what this is
}

void AMDGPUMIRFormatter::printSDelayAluImm(int64_t Imm,
                                           llvm::raw_ostream &OS) const {
  // Construct an immediate string to represent the information encoded in the
  // s_delay_alu immediate.
  // .id0_<dep>[_skip_<count>_id1<dep>]
  constexpr int64_t None = 0;
  constexpr int64_t Same = 0;

  uint64_t Id0 = (Imm & 0xF);
  uint64_t Skip = ((Imm >> 4) & 0x7);
  uint64_t Id1 = ((Imm >> 7) & 0xF);
  auto Outdep = [&](uint64_t Id) {
    if (Id == None)
      OS << "NONE";
    else if (Id < 5)
      OS << "VALU_DEP_" << Id;
    else if (Id < 8)
      OS << "TRANS32_DEP_" << Id - 4;
    else
      OS << "SALU_CYCLE_" << Id - 8;
  };

  OS << ".id0_";
  Outdep(Id0);

  // If the second inst is "same" and "none", no need to print the rest of the
  // string.
  if (Skip == Same && Id1 == None)
    return;

  // Encode the second delay specification.
  OS << "_skip_";
  if (Skip == 0)
    OS << "SAME";
  else if (Skip == 1)
    OS << "NEXT";
  else
    OS << "SKIP_" << Skip - 1;

  OS << "_id1_";
  Outdep(Id1);
}

bool AMDGPUMIRFormatter::parseSDelayAluImmMnemonic(
    const unsigned int OpIdx, int64_t &Imm, llvm::StringRef &Src,
    llvm::MIRFormatter::ErrorCallbackType &ErrorCallback) const
{
  assert(OpIdx == 0);

  Imm = 0;
  bool Expected = Src.consume_front(".id0_");
  if (!Expected)
    return ErrorCallback(Src.begin(), "Expected .id0_");

  auto ExpectInt = [&](StringRef &Src, int64_t Offset) -> int64_t {
    int64_t Dep;
    if (!Src.consumeInteger(10, Dep))
      return Dep + Offset;

    return -1;
  };

  auto DecodeDelay = [&](StringRef &Src) -> int64_t {
    if (Src.consume_front("NONE"))
      return 0;
    if (Src.consume_front("VALU_DEP_"))
      return ExpectInt(Src, 0);
    if (Src.consume_front("TRANS32_DEP_"))
      return ExpectInt(Src, 4);
    if (Src.consume_front("SALU_CYCLE_"))
      return ExpectInt(Src, 8);

    return -1;
  };

  int64_t Delay0 = DecodeDelay(Src);
  int64_t Skip = 0;
  int64_t Delay1 = 0;
  if (Delay0 == -1)
    return ErrorCallback(Src.begin(), "Could not decode delay0");


  // Set the Imm so far, to that early return has the correct value.
  Imm = Delay0;

  // If that was the end of the string, the second instruction is "same" and
  // "none"
  if (Src.begin() == Src.end())
    return false;

  Expected = Src.consume_front("_skip_");
  if (!Expected)
    return ErrorCallback(Src.begin(), "Expected _skip_");


  if (Src.consume_front("SAME")) {
    Skip = 0;
  } else if (Src.consume_front("NEXT")) {
    Skip = 1;
  } else if (Src.consume_front("SKIP_")) {
    if (Src.consumeInteger(10, Skip)) {
      return ErrorCallback(Src.begin(), "Expected integer Skip value");
    }
    Skip += 1;
  } else {
    ErrorCallback(Src.begin(), "Unexpected Skip Value");
  }

  Expected = Src.consume_front("_id1_");
  if (!Expected)
    return ErrorCallback(Src.begin(), "Expected _id1_");

  Delay1 = DecodeDelay(Src);
  if (Delay1 == -1)
    return ErrorCallback(Src.begin(), "Could not decode delay1");

  Imm = Imm | (Skip << 4) | (Delay1 << 7);
  return false;
}

bool AMDGPUMIRFormatter::parseCustomPseudoSourceValue(
    StringRef Src, MachineFunction &MF, PerFunctionMIParsingState &PFS,
    const PseudoSourceValue *&PSV, ErrorCallbackType ErrorCallback) const {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const AMDGPUTargetMachine &TM =
      static_cast<const AMDGPUTargetMachine &>(MF.getTarget());
  if (Src == "GWSResource") {
    PSV = MFI->getGWSPSV(TM);
    return false;
  }
  llvm_unreachable("unknown MIR custom pseudo source value");
}
