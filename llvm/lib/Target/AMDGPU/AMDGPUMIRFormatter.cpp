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
#include "SIMachineFunctionInfo.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;

const char SWaitAluImmPrefix = '.';
StringLiteral SWaitAluDelim = "_";

StringLiteral VaVdstName = "VaVdst";
StringLiteral VaSdstName = "VaSdst";
StringLiteral VaSsrcName = "VaSsrc";
StringLiteral HoldCntName = "HoldCnt";
StringLiteral VmVsrcName = "VmVsrc";
StringLiteral VaVccName = "VaVcc";
StringLiteral SaSdstName = "SaSdst";

StringLiteral AllOff = "AllOff";

void AMDGPUMIRFormatter::printSWaitAluImm(uint64_t Imm, raw_ostream &OS) const {
  bool NonePrinted = true;
  ListSeparator Delim(SWaitAluDelim);
  auto PrintFieldIfNotMax = [&](StringRef Descr, uint64_t Num, unsigned Max) {
    if (Num != Max) {
      OS << Delim << Descr << SWaitAluDelim << Num;
      NonePrinted = false;
    }
  };
  OS << SWaitAluImmPrefix;
  PrintFieldIfNotMax(VaVdstName, AMDGPU::DepCtr::decodeFieldVaVdst(Imm),
                     AMDGPU::DepCtr::getVaVdstBitMask());
  PrintFieldIfNotMax(VaSdstName, AMDGPU::DepCtr::decodeFieldVaSdst(Imm),
                     AMDGPU::DepCtr::getVaSdstBitMask());
  PrintFieldIfNotMax(VaSsrcName, AMDGPU::DepCtr::decodeFieldVaSsrc(Imm),
                     AMDGPU::DepCtr::getVaSsrcBitMask());
  PrintFieldIfNotMax(
      HoldCntName,
      AMDGPU::DepCtr::decodeFieldHoldCnt(Imm,
                                         AMDGPU::getIsaVersion(STI.getCPU())),
      AMDGPU::DepCtr::getHoldCntBitMask(AMDGPU::getIsaVersion(STI.getCPU())));
  PrintFieldIfNotMax(VmVsrcName, AMDGPU::DepCtr::decodeFieldVmVsrc(Imm),
                     AMDGPU::DepCtr::getVmVsrcBitMask());
  PrintFieldIfNotMax(VaVccName, AMDGPU::DepCtr::decodeFieldVaVcc(Imm),
                     AMDGPU::DepCtr::getVaVccBitMask());
  PrintFieldIfNotMax(SaSdstName, AMDGPU::DepCtr::decodeFieldSaSdst(Imm),
                     AMDGPU::DepCtr::getSaSdstBitMask());
  if (NonePrinted)
    OS << AllOff;
}

void AMDGPUMIRFormatter::printImm(raw_ostream &OS, const MachineInstr &MI,
                      std::optional<unsigned int> OpIdx, int64_t Imm) const {

  switch (MI.getOpcode()) {
  case AMDGPU::S_WAITCNT_DEPCTR:
    printSWaitAluImm(Imm, OS);
    break;
  case AMDGPU::S_DELAY_ALU:
    assert(OpIdx == 0);
    printSDelayAluImm(Imm, OS);
    break;
  default:
    MIRFormatter::printImm(OS, MI, OpIdx, Imm);
    break;
  }
}

/// Implement target specific parsing of immediate mnemonics. The mnemonic is
/// a string with a leading dot.
bool AMDGPUMIRFormatter::parseImmMnemonic(const unsigned OpCode,
                              const unsigned OpIdx,
                              StringRef Src, int64_t &Imm,
                              ErrorCallbackType ErrorCallback) const
{

  switch (OpCode) {
  case AMDGPU::S_WAITCNT_DEPCTR:
    return parseSWaitAluImmMnemonic(OpIdx, Imm, Src, ErrorCallback);
  case AMDGPU::S_DELAY_ALU:
    return parseSDelayAluImmMnemonic(OpIdx, Imm, Src, ErrorCallback);
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

bool AMDGPUMIRFormatter::parseSWaitAluImmMnemonic(
    const unsigned int OpIdx, int64_t &Imm, StringRef &Src,
    MIRFormatter::ErrorCallbackType &ErrorCallback) const {
  // TODO: For now accept integer masks for compatibility with old MIR.
  if (!Src.consumeInteger(10, Imm))
    return false;

  // Initialize with all checks off.
  Imm = AMDGPU::DepCtr::getDefaultDepCtrEncoding(STI);
  // The input is in the form: .Name1_Num1_Name2_Num2
  // Drop the '.' prefix.
  bool ConsumePrefix = Src.consume_front(SWaitAluImmPrefix);
  if (!ConsumePrefix)
    return ErrorCallback(Src.begin(), "expected prefix");
  if (Src.empty())
    return ErrorCallback(Src.begin(), "expected <CounterName>_<CounterNum>");

  // Special case for all off.
  if (Src == AllOff)
    return false;

  // Parse a counter name, number pair in each iteration.
  while (!Src.empty()) {
    // Src: Name1_Num1_Name2_Num2
    //           ^
    size_t DelimIdx = Src.find(SWaitAluDelim);
    if (DelimIdx == StringRef::npos)
      return ErrorCallback(Src.begin(), "expected <CounterName>_<CounterNum>");
    // Src: Name1_Num1_Name2_Num2
    //      ^^^^^
    StringRef Name = Src.substr(0, DelimIdx);
    // Save the position of the name for accurate error reporting.
    StringRef::iterator NamePos = Src.begin();
    [[maybe_unused]] bool ConsumeName = Src.consume_front(Name);
    assert(ConsumeName && "Expected name");
    [[maybe_unused]] bool ConsumeDelim = Src.consume_front(SWaitAluDelim);
    assert(ConsumeDelim && "Expected delimiter");
    // Src:       Num1_Name2_Num2
    //                ^
    DelimIdx = Src.find(SWaitAluDelim);
    // Src:       Num1_Name2_Num2
    //            ^^^^
    int64_t Num;
    // Save the position of the number for accurate error reporting.
    StringRef::iterator NumPos = Src.begin();
    if (Src.consumeInteger(10, Num) || Num < 0)
      return ErrorCallback(NumPos,
                           "expected non-negative integer counter number");
    unsigned Max;
    if (Name == VaVdstName) {
      Max = AMDGPU::DepCtr::getVaVdstBitMask();
      Imm = AMDGPU::DepCtr::encodeFieldVaVdst(Imm, Num);
    } else if (Name == VmVsrcName) {
      Max = AMDGPU::DepCtr::getVmVsrcBitMask();
      Imm = AMDGPU::DepCtr::encodeFieldVmVsrc(Imm, Num);
    } else if (Name == VaSdstName) {
      Max = AMDGPU::DepCtr::getVaSdstBitMask();
      Imm = AMDGPU::DepCtr::encodeFieldVaSdst(Imm, Num);
    } else if (Name == VaSsrcName) {
      Max = AMDGPU::DepCtr::getVaSsrcBitMask();
      Imm = AMDGPU::DepCtr::encodeFieldVaSsrc(Imm, Num);
    } else if (Name == HoldCntName) {
      const AMDGPU::IsaVersion &Version = AMDGPU::getIsaVersion(STI.getCPU());
      Max = AMDGPU::DepCtr::getHoldCntBitMask(Version);
      Imm = AMDGPU::DepCtr::encodeFieldHoldCnt(Imm, Num, Version);
    } else if (Name == VaVccName) {
      Max = AMDGPU::DepCtr::getVaVccBitMask();
      Imm = AMDGPU::DepCtr::encodeFieldVaVcc(Imm, Num);
    } else if (Name == SaSdstName) {
      Max = AMDGPU::DepCtr::getSaSdstBitMask();
      Imm = AMDGPU::DepCtr::encodeFieldSaSdst(Imm, Num);
    } else {
      return ErrorCallback(NamePos, "invalid counter name");
    }
    // Don't allow the values to reach their maximum value.
    if (Num >= Max)
      return ErrorCallback(NumPos, "counter value too large");
    // Src:            Name2_Num2
    Src.consume_front(SWaitAluDelim);
  }
  return false;
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
