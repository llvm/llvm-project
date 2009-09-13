//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "X86GenInstrInfo.inc"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetAsmInfo.h"
using namespace llvm;

namespace {
  cl::opt<bool>
  NoFusing("disable-spill-fusing",
           cl::desc("Disable fusing of spill code into instructions"));
  cl::opt<bool>
  PrintFailedFusing("print-failed-fuse-candidates",
                    cl::desc("Print instructions that the allocator wants to"
                             " fuse, but the X86 backend currently can't"),
                    cl::Hidden);
  cl::opt<bool>
  ReMatPICStubLoad("remat-pic-stub-load",
                   cl::desc("Re-materialize load from stub in PIC mode"),
                   cl::init(false), cl::Hidden);
}

X86InstrInfo::X86InstrInfo(X86TargetMachine &tm)
  : TargetInstrInfoImpl(X86Insts, array_lengthof(X86Insts)),
    TM(tm), RI(tm, *this) {
  SmallVector<unsigned,16> AmbEntries;
  static const unsigned OpTbl2Addr[][2] = {
    { X86::ADC32ri,     X86::ADC32mi },
    { X86::ADC32ri8,    X86::ADC32mi8 },
    { X86::ADC32rr,     X86::ADC32mr },
    { X86::ADC64ri32,   X86::ADC64mi32 },
    { X86::ADC64ri8,    X86::ADC64mi8 },
    { X86::ADC64rr,     X86::ADC64mr },
    { X86::ADD16ri,     X86::ADD16mi },
    { X86::ADD16ri8,    X86::ADD16mi8 },
    { X86::ADD16rr,     X86::ADD16mr },
    { X86::ADD32ri,     X86::ADD32mi },
    { X86::ADD32ri8,    X86::ADD32mi8 },
    { X86::ADD32rr,     X86::ADD32mr },
    { X86::ADD64ri32,   X86::ADD64mi32 },
    { X86::ADD64ri8,    X86::ADD64mi8 },
    { X86::ADD64rr,     X86::ADD64mr },
    { X86::ADD8ri,      X86::ADD8mi },
    { X86::ADD8rr,      X86::ADD8mr },
    { X86::AND16ri,     X86::AND16mi },
    { X86::AND16ri8,    X86::AND16mi8 },
    { X86::AND16rr,     X86::AND16mr },
    { X86::AND32ri,     X86::AND32mi },
    { X86::AND32ri8,    X86::AND32mi8 },
    { X86::AND32rr,     X86::AND32mr },
    { X86::AND64ri32,   X86::AND64mi32 },
    { X86::AND64ri8,    X86::AND64mi8 },
    { X86::AND64rr,     X86::AND64mr },
    { X86::AND8ri,      X86::AND8mi },
    { X86::AND8rr,      X86::AND8mr },
    { X86::DEC16r,      X86::DEC16m },
    { X86::DEC32r,      X86::DEC32m },
    { X86::DEC64_16r,   X86::DEC64_16m },
    { X86::DEC64_32r,   X86::DEC64_32m },
    { X86::DEC64r,      X86::DEC64m },
    { X86::DEC8r,       X86::DEC8m },
    { X86::INC16r,      X86::INC16m },
    { X86::INC32r,      X86::INC32m },
    { X86::INC64_16r,   X86::INC64_16m },
    { X86::INC64_32r,   X86::INC64_32m },
    { X86::INC64r,      X86::INC64m },
    { X86::INC8r,       X86::INC8m },
    { X86::NEG16r,      X86::NEG16m },
    { X86::NEG32r,      X86::NEG32m },
    { X86::NEG64r,      X86::NEG64m },
    { X86::NEG8r,       X86::NEG8m },
    { X86::NOT16r,      X86::NOT16m },
    { X86::NOT32r,      X86::NOT32m },
    { X86::NOT64r,      X86::NOT64m },
    { X86::NOT8r,       X86::NOT8m },
    { X86::OR16ri,      X86::OR16mi },
    { X86::OR16ri8,     X86::OR16mi8 },
    { X86::OR16rr,      X86::OR16mr },
    { X86::OR32ri,      X86::OR32mi },
    { X86::OR32ri8,     X86::OR32mi8 },
    { X86::OR32rr,      X86::OR32mr },
    { X86::OR64ri32,    X86::OR64mi32 },
    { X86::OR64ri8,     X86::OR64mi8 },
    { X86::OR64rr,      X86::OR64mr },
    { X86::OR8ri,       X86::OR8mi },
    { X86::OR8rr,       X86::OR8mr },
    { X86::ROL16r1,     X86::ROL16m1 },
    { X86::ROL16rCL,    X86::ROL16mCL },
    { X86::ROL16ri,     X86::ROL16mi },
    { X86::ROL32r1,     X86::ROL32m1 },
    { X86::ROL32rCL,    X86::ROL32mCL },
    { X86::ROL32ri,     X86::ROL32mi },
    { X86::ROL64r1,     X86::ROL64m1 },
    { X86::ROL64rCL,    X86::ROL64mCL },
    { X86::ROL64ri,     X86::ROL64mi },
    { X86::ROL8r1,      X86::ROL8m1 },
    { X86::ROL8rCL,     X86::ROL8mCL },
    { X86::ROL8ri,      X86::ROL8mi },
    { X86::ROR16r1,     X86::ROR16m1 },
    { X86::ROR16rCL,    X86::ROR16mCL },
    { X86::ROR16ri,     X86::ROR16mi },
    { X86::ROR32r1,     X86::ROR32m1 },
    { X86::ROR32rCL,    X86::ROR32mCL },
    { X86::ROR32ri,     X86::ROR32mi },
    { X86::ROR64r1,     X86::ROR64m1 },
    { X86::ROR64rCL,    X86::ROR64mCL },
    { X86::ROR64ri,     X86::ROR64mi },
    { X86::ROR8r1,      X86::ROR8m1 },
    { X86::ROR8rCL,     X86::ROR8mCL },
    { X86::ROR8ri,      X86::ROR8mi },
    { X86::SAR16r1,     X86::SAR16m1 },
    { X86::SAR16rCL,    X86::SAR16mCL },
    { X86::SAR16ri,     X86::SAR16mi },
    { X86::SAR32r1,     X86::SAR32m1 },
    { X86::SAR32rCL,    X86::SAR32mCL },
    { X86::SAR32ri,     X86::SAR32mi },
    { X86::SAR64r1,     X86::SAR64m1 },
    { X86::SAR64rCL,    X86::SAR64mCL },
    { X86::SAR64ri,     X86::SAR64mi },
    { X86::SAR8r1,      X86::SAR8m1 },
    { X86::SAR8rCL,     X86::SAR8mCL },
    { X86::SAR8ri,      X86::SAR8mi },
    { X86::SBB32ri,     X86::SBB32mi },
    { X86::SBB32ri8,    X86::SBB32mi8 },
    { X86::SBB32rr,     X86::SBB32mr },
    { X86::SBB64ri32,   X86::SBB64mi32 },
    { X86::SBB64ri8,    X86::SBB64mi8 },
    { X86::SBB64rr,     X86::SBB64mr },
    { X86::SHL16rCL,    X86::SHL16mCL },
    { X86::SHL16ri,     X86::SHL16mi },
    { X86::SHL32rCL,    X86::SHL32mCL },
    { X86::SHL32ri,     X86::SHL32mi },
    { X86::SHL64rCL,    X86::SHL64mCL },
    { X86::SHL64ri,     X86::SHL64mi },
    { X86::SHL8rCL,     X86::SHL8mCL },
    { X86::SHL8ri,      X86::SHL8mi },
    { X86::SHLD16rrCL,  X86::SHLD16mrCL },
    { X86::SHLD16rri8,  X86::SHLD16mri8 },
    { X86::SHLD32rrCL,  X86::SHLD32mrCL },
    { X86::SHLD32rri8,  X86::SHLD32mri8 },
    { X86::SHLD64rrCL,  X86::SHLD64mrCL },
    { X86::SHLD64rri8,  X86::SHLD64mri8 },
    { X86::SHR16r1,     X86::SHR16m1 },
    { X86::SHR16rCL,    X86::SHR16mCL },
    { X86::SHR16ri,     X86::SHR16mi },
    { X86::SHR32r1,     X86::SHR32m1 },
    { X86::SHR32rCL,    X86::SHR32mCL },
    { X86::SHR32ri,     X86::SHR32mi },
    { X86::SHR64r1,     X86::SHR64m1 },
    { X86::SHR64rCL,    X86::SHR64mCL },
    { X86::SHR64ri,     X86::SHR64mi },
    { X86::SHR8r1,      X86::SHR8m1 },
    { X86::SHR8rCL,     X86::SHR8mCL },
    { X86::SHR8ri,      X86::SHR8mi },
    { X86::SHRD16rrCL,  X86::SHRD16mrCL },
    { X86::SHRD16rri8,  X86::SHRD16mri8 },
    { X86::SHRD32rrCL,  X86::SHRD32mrCL },
    { X86::SHRD32rri8,  X86::SHRD32mri8 },
    { X86::SHRD64rrCL,  X86::SHRD64mrCL },
    { X86::SHRD64rri8,  X86::SHRD64mri8 },
    { X86::SUB16ri,     X86::SUB16mi },
    { X86::SUB16ri8,    X86::SUB16mi8 },
    { X86::SUB16rr,     X86::SUB16mr },
    { X86::SUB32ri,     X86::SUB32mi },
    { X86::SUB32ri8,    X86::SUB32mi8 },
    { X86::SUB32rr,     X86::SUB32mr },
    { X86::SUB64ri32,   X86::SUB64mi32 },
    { X86::SUB64ri8,    X86::SUB64mi8 },
    { X86::SUB64rr,     X86::SUB64mr },
    { X86::SUB8ri,      X86::SUB8mi },
    { X86::SUB8rr,      X86::SUB8mr },
    { X86::XOR16ri,     X86::XOR16mi },
    { X86::XOR16ri8,    X86::XOR16mi8 },
    { X86::XOR16rr,     X86::XOR16mr },
    { X86::XOR32ri,     X86::XOR32mi },
    { X86::XOR32ri8,    X86::XOR32mi8 },
    { X86::XOR32rr,     X86::XOR32mr },
    { X86::XOR64ri32,   X86::XOR64mi32 },
    { X86::XOR64ri8,    X86::XOR64mi8 },
    { X86::XOR64rr,     X86::XOR64mr },
    { X86::XOR8ri,      X86::XOR8mi },
    { X86::XOR8rr,      X86::XOR8mr }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2Addr); i != e; ++i) {
    unsigned RegOp = OpTbl2Addr[i][0];
    unsigned MemOp = OpTbl2Addr[i][1];
    if (!RegOp2MemOpTable2Addr.insert(std::make_pair((unsigned*)RegOp,
                                               std::make_pair(MemOp,0))).second)
      assert(false && "Duplicated entries?");
    // Index 0, folded load and store, no alignment requirement.
    unsigned AuxInfo = 0 | (1 << 4) | (1 << 5);
    if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                                std::make_pair(RegOp,
                                                              AuxInfo))).second)
      AmbEntries.push_back(MemOp);
  }

  // If the third value is 1, then it's folding either a load or a store.
  static const unsigned OpTbl0[][4] = {
    { X86::BT16ri8,     X86::BT16mi8, 1, 0 },
    { X86::BT32ri8,     X86::BT32mi8, 1, 0 },
    { X86::BT64ri8,     X86::BT64mi8, 1, 0 },
    { X86::CALL32r,     X86::CALL32m, 1, 0 },
    { X86::CALL64r,     X86::CALL64m, 1, 0 },
    { X86::CMP16ri,     X86::CMP16mi, 1, 0 },
    { X86::CMP16ri8,    X86::CMP16mi8, 1, 0 },
    { X86::CMP16rr,     X86::CMP16mr, 1, 0 },
    { X86::CMP32ri,     X86::CMP32mi, 1, 0 },
    { X86::CMP32ri8,    X86::CMP32mi8, 1, 0 },
    { X86::CMP32rr,     X86::CMP32mr, 1, 0 },
    { X86::CMP64ri32,   X86::CMP64mi32, 1, 0 },
    { X86::CMP64ri8,    X86::CMP64mi8, 1, 0 },
    { X86::CMP64rr,     X86::CMP64mr, 1, 0 },
    { X86::CMP8ri,      X86::CMP8mi, 1, 0 },
    { X86::CMP8rr,      X86::CMP8mr, 1, 0 },
    { X86::DIV16r,      X86::DIV16m, 1, 0 },
    { X86::DIV32r,      X86::DIV32m, 1, 0 },
    { X86::DIV64r,      X86::DIV64m, 1, 0 },
    { X86::DIV8r,       X86::DIV8m, 1, 0 },
    { X86::EXTRACTPSrr, X86::EXTRACTPSmr, 0, 16 },
    { X86::FsMOVAPDrr,  X86::MOVSDmr, 0, 0 },
    { X86::FsMOVAPSrr,  X86::MOVSSmr, 0, 0 },
    { X86::IDIV16r,     X86::IDIV16m, 1, 0 },
    { X86::IDIV32r,     X86::IDIV32m, 1, 0 },
    { X86::IDIV64r,     X86::IDIV64m, 1, 0 },
    { X86::IDIV8r,      X86::IDIV8m, 1, 0 },
    { X86::IMUL16r,     X86::IMUL16m, 1, 0 },
    { X86::IMUL32r,     X86::IMUL32m, 1, 0 },
    { X86::IMUL64r,     X86::IMUL64m, 1, 0 },
    { X86::IMUL8r,      X86::IMUL8m, 1, 0 },
    { X86::JMP32r,      X86::JMP32m, 1, 0 },
    { X86::JMP64r,      X86::JMP64m, 1, 0 },
    { X86::MOV16ri,     X86::MOV16mi, 0, 0 },
    { X86::MOV16rr,     X86::MOV16mr, 0, 0 },
    { X86::MOV32ri,     X86::MOV32mi, 0, 0 },
    { X86::MOV32rr,     X86::MOV32mr, 0, 0 },
    { X86::MOV64ri32,   X86::MOV64mi32, 0, 0 },
    { X86::MOV64rr,     X86::MOV64mr, 0, 0 },
    { X86::MOV8ri,      X86::MOV8mi, 0, 0 },
    { X86::MOV8rr,      X86::MOV8mr, 0, 0 },
    { X86::MOV8rr_NOREX, X86::MOV8mr_NOREX, 0, 0 },
    { X86::MOVAPDrr,    X86::MOVAPDmr, 0, 16 },
    { X86::MOVAPSrr,    X86::MOVAPSmr, 0, 16 },
    { X86::MOVDQArr,    X86::MOVDQAmr, 0, 16 },
    { X86::MOVPDI2DIrr, X86::MOVPDI2DImr, 0, 0 },
    { X86::MOVPQIto64rr,X86::MOVPQI2QImr, 0, 0 },
    { X86::MOVPS2SSrr,  X86::MOVPS2SSmr, 0, 0 },
    { X86::MOVSDrr,     X86::MOVSDmr, 0, 0 },
    { X86::MOVSDto64rr, X86::MOVSDto64mr, 0, 0 },
    { X86::MOVSS2DIrr,  X86::MOVSS2DImr, 0, 0 },
    { X86::MOVSSrr,     X86::MOVSSmr, 0, 0 },
    { X86::MOVUPDrr,    X86::MOVUPDmr, 0, 0 },
    { X86::MOVUPSrr,    X86::MOVUPSmr, 0, 0 },
    { X86::MUL16r,      X86::MUL16m, 1, 0 },
    { X86::MUL32r,      X86::MUL32m, 1, 0 },
    { X86::MUL64r,      X86::MUL64m, 1, 0 },
    { X86::MUL8r,       X86::MUL8m, 1, 0 },
    { X86::SETAEr,      X86::SETAEm, 0, 0 },
    { X86::SETAr,       X86::SETAm, 0, 0 },
    { X86::SETBEr,      X86::SETBEm, 0, 0 },
    { X86::SETBr,       X86::SETBm, 0, 0 },
    { X86::SETEr,       X86::SETEm, 0, 0 },
    { X86::SETGEr,      X86::SETGEm, 0, 0 },
    { X86::SETGr,       X86::SETGm, 0, 0 },
    { X86::SETLEr,      X86::SETLEm, 0, 0 },
    { X86::SETLr,       X86::SETLm, 0, 0 },
    { X86::SETNEr,      X86::SETNEm, 0, 0 },
    { X86::SETNOr,      X86::SETNOm, 0, 0 },
    { X86::SETNPr,      X86::SETNPm, 0, 0 },
    { X86::SETNSr,      X86::SETNSm, 0, 0 },
    { X86::SETOr,       X86::SETOm, 0, 0 },
    { X86::SETPr,       X86::SETPm, 0, 0 },
    { X86::SETSr,       X86::SETSm, 0, 0 },
    { X86::TAILJMPr,    X86::TAILJMPm, 1, 0 },
    { X86::TEST16ri,    X86::TEST16mi, 1, 0 },
    { X86::TEST32ri,    X86::TEST32mi, 1, 0 },
    { X86::TEST64ri32,  X86::TEST64mi32, 1, 0 },
    { X86::TEST8ri,     X86::TEST8mi, 1, 0 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl0); i != e; ++i) {
    unsigned RegOp = OpTbl0[i][0];
    unsigned MemOp = OpTbl0[i][1];
    unsigned Align = OpTbl0[i][3];
    if (!RegOp2MemOpTable0.insert(std::make_pair((unsigned*)RegOp,
                                           std::make_pair(MemOp,Align))).second)
      assert(false && "Duplicated entries?");
    unsigned FoldedLoad = OpTbl0[i][2];
    // Index 0, folded load or store.
    unsigned AuxInfo = 0 | (FoldedLoad << 4) | ((FoldedLoad^1) << 5);
    if (RegOp != X86::FsMOVAPDrr && RegOp != X86::FsMOVAPSrr)
      if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                     std::make_pair(RegOp, AuxInfo))).second)
        AmbEntries.push_back(MemOp);
  }

  static const unsigned OpTbl1[][3] = {
    { X86::CMP16rr,         X86::CMP16rm, 0 },
    { X86::CMP32rr,         X86::CMP32rm, 0 },
    { X86::CMP64rr,         X86::CMP64rm, 0 },
    { X86::CMP8rr,          X86::CMP8rm, 0 },
    { X86::CVTSD2SSrr,      X86::CVTSD2SSrm, 0 },
    { X86::CVTSI2SD64rr,    X86::CVTSI2SD64rm, 0 },
    { X86::CVTSI2SDrr,      X86::CVTSI2SDrm, 0 },
    { X86::CVTSI2SS64rr,    X86::CVTSI2SS64rm, 0 },
    { X86::CVTSI2SSrr,      X86::CVTSI2SSrm, 0 },
    { X86::CVTSS2SDrr,      X86::CVTSS2SDrm, 0 },
    { X86::CVTTSD2SI64rr,   X86::CVTTSD2SI64rm, 0 },
    { X86::CVTTSD2SIrr,     X86::CVTTSD2SIrm, 0 },
    { X86::CVTTSS2SI64rr,   X86::CVTTSS2SI64rm, 0 },
    { X86::CVTTSS2SIrr,     X86::CVTTSS2SIrm, 0 },
    { X86::FsMOVAPDrr,      X86::MOVSDrm, 0 },
    { X86::FsMOVAPSrr,      X86::MOVSSrm, 0 },
    { X86::IMUL16rri,       X86::IMUL16rmi, 0 },
    { X86::IMUL16rri8,      X86::IMUL16rmi8, 0 },
    { X86::IMUL32rri,       X86::IMUL32rmi, 0 },
    { X86::IMUL32rri8,      X86::IMUL32rmi8, 0 },
    { X86::IMUL64rri32,     X86::IMUL64rmi32, 0 },
    { X86::IMUL64rri8,      X86::IMUL64rmi8, 0 },
    { X86::Int_CMPSDrr,     X86::Int_CMPSDrm, 0 },
    { X86::Int_CMPSSrr,     X86::Int_CMPSSrm, 0 },
    { X86::Int_COMISDrr,    X86::Int_COMISDrm, 0 },
    { X86::Int_COMISSrr,    X86::Int_COMISSrm, 0 },
    { X86::Int_CVTDQ2PDrr,  X86::Int_CVTDQ2PDrm, 16 },
    { X86::Int_CVTDQ2PSrr,  X86::Int_CVTDQ2PSrm, 16 },
    { X86::Int_CVTPD2DQrr,  X86::Int_CVTPD2DQrm, 16 },
    { X86::Int_CVTPD2PSrr,  X86::Int_CVTPD2PSrm, 16 },
    { X86::Int_CVTPS2DQrr,  X86::Int_CVTPS2DQrm, 16 },
    { X86::Int_CVTPS2PDrr,  X86::Int_CVTPS2PDrm, 0 },
    { X86::Int_CVTSD2SI64rr,X86::Int_CVTSD2SI64rm, 0 },
    { X86::Int_CVTSD2SIrr,  X86::Int_CVTSD2SIrm, 0 },
    { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm, 0 },
    { X86::Int_CVTSI2SD64rr,X86::Int_CVTSI2SD64rm, 0 },
    { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm, 0 },
    { X86::Int_CVTSI2SS64rr,X86::Int_CVTSI2SS64rm, 0 },
    { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm, 0 },
    { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm, 0 },
    { X86::Int_CVTSS2SI64rr,X86::Int_CVTSS2SI64rm, 0 },
    { X86::Int_CVTSS2SIrr,  X86::Int_CVTSS2SIrm, 0 },
    { X86::Int_CVTTPD2DQrr, X86::Int_CVTTPD2DQrm, 16 },
    { X86::Int_CVTTPS2DQrr, X86::Int_CVTTPS2DQrm, 16 },
    { X86::Int_CVTTSD2SI64rr,X86::Int_CVTTSD2SI64rm, 0 },
    { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm, 0 },
    { X86::Int_CVTTSS2SI64rr,X86::Int_CVTTSS2SI64rm, 0 },
    { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm, 0 },
    { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm, 0 },
    { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm, 0 },
    { X86::MOV16rr,         X86::MOV16rm, 0 },
    { X86::MOV32rr,         X86::MOV32rm, 0 },
    { X86::MOV64rr,         X86::MOV64rm, 0 },
    { X86::MOV64toPQIrr,    X86::MOVQI2PQIrm, 0 },
    { X86::MOV64toSDrr,     X86::MOV64toSDrm, 0 },
    { X86::MOV8rr,          X86::MOV8rm, 0 },
    { X86::MOVAPDrr,        X86::MOVAPDrm, 16 },
    { X86::MOVAPSrr,        X86::MOVAPSrm, 16 },
    { X86::MOVDDUPrr,       X86::MOVDDUPrm, 0 },
    { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm, 0 },
    { X86::MOVDI2SSrr,      X86::MOVDI2SSrm, 0 },
    { X86::MOVDQArr,        X86::MOVDQArm, 16 },
    { X86::MOVSD2PDrr,      X86::MOVSD2PDrm, 0 },
    { X86::MOVSDrr,         X86::MOVSDrm, 0 },
    { X86::MOVSHDUPrr,      X86::MOVSHDUPrm, 16 },
    { X86::MOVSLDUPrr,      X86::MOVSLDUPrm, 16 },
    { X86::MOVSS2PSrr,      X86::MOVSS2PSrm, 0 },
    { X86::MOVSSrr,         X86::MOVSSrm, 0 },
    { X86::MOVSX16rr8,      X86::MOVSX16rm8, 0 },
    { X86::MOVSX32rr16,     X86::MOVSX32rm16, 0 },
    { X86::MOVSX32rr8,      X86::MOVSX32rm8, 0 },
    { X86::MOVSX64rr16,     X86::MOVSX64rm16, 0 },
    { X86::MOVSX64rr32,     X86::MOVSX64rm32, 0 },
    { X86::MOVSX64rr8,      X86::MOVSX64rm8, 0 },
    { X86::MOVUPDrr,        X86::MOVUPDrm, 16 },
    { X86::MOVUPSrr,        X86::MOVUPSrm, 16 },
    { X86::MOVZDI2PDIrr,    X86::MOVZDI2PDIrm, 0 },
    { X86::MOVZQI2PQIrr,    X86::MOVZQI2PQIrm, 0 },
    { X86::MOVZPQILo2PQIrr, X86::MOVZPQILo2PQIrm, 16 },
    { X86::MOVZX16rr8,      X86::MOVZX16rm8, 0 },
    { X86::MOVZX32rr16,     X86::MOVZX32rm16, 0 },
    { X86::MOVZX32_NOREXrr8, X86::MOVZX32_NOREXrm8, 0 },
    { X86::MOVZX32rr8,      X86::MOVZX32rm8, 0 },
    { X86::MOVZX64rr16,     X86::MOVZX64rm16, 0 },
    { X86::MOVZX64rr32,     X86::MOVZX64rm32, 0 },
    { X86::MOVZX64rr8,      X86::MOVZX64rm8, 0 },
    { X86::PSHUFDri,        X86::PSHUFDmi, 16 },
    { X86::PSHUFHWri,       X86::PSHUFHWmi, 16 },
    { X86::PSHUFLWri,       X86::PSHUFLWmi, 16 },
    { X86::RCPPSr,          X86::RCPPSm, 16 },
    { X86::RCPPSr_Int,      X86::RCPPSm_Int, 16 },
    { X86::RSQRTPSr,        X86::RSQRTPSm, 16 },
    { X86::RSQRTPSr_Int,    X86::RSQRTPSm_Int, 16 },
    { X86::RSQRTSSr,        X86::RSQRTSSm, 0 },
    { X86::RSQRTSSr_Int,    X86::RSQRTSSm_Int, 0 },
    { X86::SQRTPDr,         X86::SQRTPDm, 16 },
    { X86::SQRTPDr_Int,     X86::SQRTPDm_Int, 16 },
    { X86::SQRTPSr,         X86::SQRTPSm, 16 },
    { X86::SQRTPSr_Int,     X86::SQRTPSm_Int, 16 },
    { X86::SQRTSDr,         X86::SQRTSDm, 0 },
    { X86::SQRTSDr_Int,     X86::SQRTSDm_Int, 0 },
    { X86::SQRTSSr,         X86::SQRTSSm, 0 },
    { X86::SQRTSSr_Int,     X86::SQRTSSm_Int, 0 },
    { X86::TEST16rr,        X86::TEST16rm, 0 },
    { X86::TEST32rr,        X86::TEST32rm, 0 },
    { X86::TEST64rr,        X86::TEST64rm, 0 },
    { X86::TEST8rr,         X86::TEST8rm, 0 },
    // FIXME: TEST*rr EAX,EAX ---> CMP [mem], 0
    { X86::UCOMISDrr,       X86::UCOMISDrm, 0 },
    { X86::UCOMISSrr,       X86::UCOMISSrm, 0 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl1); i != e; ++i) {
    unsigned RegOp = OpTbl1[i][0];
    unsigned MemOp = OpTbl1[i][1];
    unsigned Align = OpTbl1[i][2];
    if (!RegOp2MemOpTable1.insert(std::make_pair((unsigned*)RegOp,
                                           std::make_pair(MemOp,Align))).second)
      assert(false && "Duplicated entries?");
    // Index 1, folded load
    unsigned AuxInfo = 1 | (1 << 4);
    if (RegOp != X86::FsMOVAPDrr && RegOp != X86::FsMOVAPSrr)
      if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                     std::make_pair(RegOp, AuxInfo))).second)
        AmbEntries.push_back(MemOp);
  }

  static const unsigned OpTbl2[][3] = {
    { X86::ADC32rr,         X86::ADC32rm, 0 },
    { X86::ADC64rr,         X86::ADC64rm, 0 },
    { X86::ADD16rr,         X86::ADD16rm, 0 },
    { X86::ADD32rr,         X86::ADD32rm, 0 },
    { X86::ADD64rr,         X86::ADD64rm, 0 },
    { X86::ADD8rr,          X86::ADD8rm, 0 },
    { X86::ADDPDrr,         X86::ADDPDrm, 16 },
    { X86::ADDPSrr,         X86::ADDPSrm, 16 },
    { X86::ADDSDrr,         X86::ADDSDrm, 0 },
    { X86::ADDSSrr,         X86::ADDSSrm, 0 },
    { X86::ADDSUBPDrr,      X86::ADDSUBPDrm, 16 },
    { X86::ADDSUBPSrr,      X86::ADDSUBPSrm, 16 },
    { X86::AND16rr,         X86::AND16rm, 0 },
    { X86::AND32rr,         X86::AND32rm, 0 },
    { X86::AND64rr,         X86::AND64rm, 0 },
    { X86::AND8rr,          X86::AND8rm, 0 },
    { X86::ANDNPDrr,        X86::ANDNPDrm, 16 },
    { X86::ANDNPSrr,        X86::ANDNPSrm, 16 },
    { X86::ANDPDrr,         X86::ANDPDrm, 16 },
    { X86::ANDPSrr,         X86::ANDPSrm, 16 },
    { X86::CMOVA16rr,       X86::CMOVA16rm, 0 },
    { X86::CMOVA32rr,       X86::CMOVA32rm, 0 },
    { X86::CMOVA64rr,       X86::CMOVA64rm, 0 },
    { X86::CMOVAE16rr,      X86::CMOVAE16rm, 0 },
    { X86::CMOVAE32rr,      X86::CMOVAE32rm, 0 },
    { X86::CMOVAE64rr,      X86::CMOVAE64rm, 0 },
    { X86::CMOVB16rr,       X86::CMOVB16rm, 0 },
    { X86::CMOVB32rr,       X86::CMOVB32rm, 0 },
    { X86::CMOVB64rr,       X86::CMOVB64rm, 0 },
    { X86::CMOVBE16rr,      X86::CMOVBE16rm, 0 },
    { X86::CMOVBE32rr,      X86::CMOVBE32rm, 0 },
    { X86::CMOVBE64rr,      X86::CMOVBE64rm, 0 },
    { X86::CMOVE16rr,       X86::CMOVE16rm, 0 },
    { X86::CMOVE32rr,       X86::CMOVE32rm, 0 },
    { X86::CMOVE64rr,       X86::CMOVE64rm, 0 },
    { X86::CMOVG16rr,       X86::CMOVG16rm, 0 },
    { X86::CMOVG32rr,       X86::CMOVG32rm, 0 },
    { X86::CMOVG64rr,       X86::CMOVG64rm, 0 },
    { X86::CMOVGE16rr,      X86::CMOVGE16rm, 0 },
    { X86::CMOVGE32rr,      X86::CMOVGE32rm, 0 },
    { X86::CMOVGE64rr,      X86::CMOVGE64rm, 0 },
    { X86::CMOVL16rr,       X86::CMOVL16rm, 0 },
    { X86::CMOVL32rr,       X86::CMOVL32rm, 0 },
    { X86::CMOVL64rr,       X86::CMOVL64rm, 0 },
    { X86::CMOVLE16rr,      X86::CMOVLE16rm, 0 },
    { X86::CMOVLE32rr,      X86::CMOVLE32rm, 0 },
    { X86::CMOVLE64rr,      X86::CMOVLE64rm, 0 },
    { X86::CMOVNE16rr,      X86::CMOVNE16rm, 0 },
    { X86::CMOVNE32rr,      X86::CMOVNE32rm, 0 },
    { X86::CMOVNE64rr,      X86::CMOVNE64rm, 0 },
    { X86::CMOVNO16rr,      X86::CMOVNO16rm, 0 },
    { X86::CMOVNO32rr,      X86::CMOVNO32rm, 0 },
    { X86::CMOVNO64rr,      X86::CMOVNO64rm, 0 },
    { X86::CMOVNP16rr,      X86::CMOVNP16rm, 0 },
    { X86::CMOVNP32rr,      X86::CMOVNP32rm, 0 },
    { X86::CMOVNP64rr,      X86::CMOVNP64rm, 0 },
    { X86::CMOVNS16rr,      X86::CMOVNS16rm, 0 },
    { X86::CMOVNS32rr,      X86::CMOVNS32rm, 0 },
    { X86::CMOVNS64rr,      X86::CMOVNS64rm, 0 },
    { X86::CMOVO16rr,       X86::CMOVO16rm, 0 },
    { X86::CMOVO32rr,       X86::CMOVO32rm, 0 },
    { X86::CMOVO64rr,       X86::CMOVO64rm, 0 },
    { X86::CMOVP16rr,       X86::CMOVP16rm, 0 },
    { X86::CMOVP32rr,       X86::CMOVP32rm, 0 },
    { X86::CMOVP64rr,       X86::CMOVP64rm, 0 },
    { X86::CMOVS16rr,       X86::CMOVS16rm, 0 },
    { X86::CMOVS32rr,       X86::CMOVS32rm, 0 },
    { X86::CMOVS64rr,       X86::CMOVS64rm, 0 },
    { X86::CMPPDrri,        X86::CMPPDrmi, 16 },
    { X86::CMPPSrri,        X86::CMPPSrmi, 16 },
    { X86::CMPSDrr,         X86::CMPSDrm, 0 },
    { X86::CMPSSrr,         X86::CMPSSrm, 0 },
    { X86::DIVPDrr,         X86::DIVPDrm, 16 },
    { X86::DIVPSrr,         X86::DIVPSrm, 16 },
    { X86::DIVSDrr,         X86::DIVSDrm, 0 },
    { X86::DIVSSrr,         X86::DIVSSrm, 0 },
    { X86::FsANDNPDrr,      X86::FsANDNPDrm, 16 },
    { X86::FsANDNPSrr,      X86::FsANDNPSrm, 16 },
    { X86::FsANDPDrr,       X86::FsANDPDrm, 16 },
    { X86::FsANDPSrr,       X86::FsANDPSrm, 16 },
    { X86::FsORPDrr,        X86::FsORPDrm, 16 },
    { X86::FsORPSrr,        X86::FsORPSrm, 16 },
    { X86::FsXORPDrr,       X86::FsXORPDrm, 16 },
    { X86::FsXORPSrr,       X86::FsXORPSrm, 16 },
    { X86::HADDPDrr,        X86::HADDPDrm, 16 },
    { X86::HADDPSrr,        X86::HADDPSrm, 16 },
    { X86::HSUBPDrr,        X86::HSUBPDrm, 16 },
    { X86::HSUBPSrr,        X86::HSUBPSrm, 16 },
    { X86::IMUL16rr,        X86::IMUL16rm, 0 },
    { X86::IMUL32rr,        X86::IMUL32rm, 0 },
    { X86::IMUL64rr,        X86::IMUL64rm, 0 },
    { X86::MAXPDrr,         X86::MAXPDrm, 16 },
    { X86::MAXPDrr_Int,     X86::MAXPDrm_Int, 16 },
    { X86::MAXPSrr,         X86::MAXPSrm, 16 },
    { X86::MAXPSrr_Int,     X86::MAXPSrm_Int, 16 },
    { X86::MAXSDrr,         X86::MAXSDrm, 0 },
    { X86::MAXSDrr_Int,     X86::MAXSDrm_Int, 0 },
    { X86::MAXSSrr,         X86::MAXSSrm, 0 },
    { X86::MAXSSrr_Int,     X86::MAXSSrm_Int, 0 },
    { X86::MINPDrr,         X86::MINPDrm, 16 },
    { X86::MINPDrr_Int,     X86::MINPDrm_Int, 16 },
    { X86::MINPSrr,         X86::MINPSrm, 16 },
    { X86::MINPSrr_Int,     X86::MINPSrm_Int, 16 },
    { X86::MINSDrr,         X86::MINSDrm, 0 },
    { X86::MINSDrr_Int,     X86::MINSDrm_Int, 0 },
    { X86::MINSSrr,         X86::MINSSrm, 0 },
    { X86::MINSSrr_Int,     X86::MINSSrm_Int, 0 },
    { X86::MULPDrr,         X86::MULPDrm, 16 },
    { X86::MULPSrr,         X86::MULPSrm, 16 },
    { X86::MULSDrr,         X86::MULSDrm, 0 },
    { X86::MULSSrr,         X86::MULSSrm, 0 },
    { X86::OR16rr,          X86::OR16rm, 0 },
    { X86::OR32rr,          X86::OR32rm, 0 },
    { X86::OR64rr,          X86::OR64rm, 0 },
    { X86::OR8rr,           X86::OR8rm, 0 },
    { X86::ORPDrr,          X86::ORPDrm, 16 },
    { X86::ORPSrr,          X86::ORPSrm, 16 },
    { X86::PACKSSDWrr,      X86::PACKSSDWrm, 16 },
    { X86::PACKSSWBrr,      X86::PACKSSWBrm, 16 },
    { X86::PACKUSWBrr,      X86::PACKUSWBrm, 16 },
    { X86::PADDBrr,         X86::PADDBrm, 16 },
    { X86::PADDDrr,         X86::PADDDrm, 16 },
    { X86::PADDQrr,         X86::PADDQrm, 16 },
    { X86::PADDSBrr,        X86::PADDSBrm, 16 },
    { X86::PADDSWrr,        X86::PADDSWrm, 16 },
    { X86::PADDWrr,         X86::PADDWrm, 16 },
    { X86::PANDNrr,         X86::PANDNrm, 16 },
    { X86::PANDrr,          X86::PANDrm, 16 },
    { X86::PAVGBrr,         X86::PAVGBrm, 16 },
    { X86::PAVGWrr,         X86::PAVGWrm, 16 },
    { X86::PCMPEQBrr,       X86::PCMPEQBrm, 16 },
    { X86::PCMPEQDrr,       X86::PCMPEQDrm, 16 },
    { X86::PCMPEQWrr,       X86::PCMPEQWrm, 16 },
    { X86::PCMPGTBrr,       X86::PCMPGTBrm, 16 },
    { X86::PCMPGTDrr,       X86::PCMPGTDrm, 16 },
    { X86::PCMPGTWrr,       X86::PCMPGTWrm, 16 },
    { X86::PINSRWrri,       X86::PINSRWrmi, 16 },
    { X86::PMADDWDrr,       X86::PMADDWDrm, 16 },
    { X86::PMAXSWrr,        X86::PMAXSWrm, 16 },
    { X86::PMAXUBrr,        X86::PMAXUBrm, 16 },
    { X86::PMINSWrr,        X86::PMINSWrm, 16 },
    { X86::PMINUBrr,        X86::PMINUBrm, 16 },
    { X86::PMULDQrr,        X86::PMULDQrm, 16 },
    { X86::PMULHUWrr,       X86::PMULHUWrm, 16 },
    { X86::PMULHWrr,        X86::PMULHWrm, 16 },
    { X86::PMULLDrr,        X86::PMULLDrm, 16 },
    { X86::PMULLDrr_int,    X86::PMULLDrm_int, 16 },
    { X86::PMULLWrr,        X86::PMULLWrm, 16 },
    { X86::PMULUDQrr,       X86::PMULUDQrm, 16 },
    { X86::PORrr,           X86::PORrm, 16 },
    { X86::PSADBWrr,        X86::PSADBWrm, 16 },
    { X86::PSLLDrr,         X86::PSLLDrm, 16 },
    { X86::PSLLQrr,         X86::PSLLQrm, 16 },
    { X86::PSLLWrr,         X86::PSLLWrm, 16 },
    { X86::PSRADrr,         X86::PSRADrm, 16 },
    { X86::PSRAWrr,         X86::PSRAWrm, 16 },
    { X86::PSRLDrr,         X86::PSRLDrm, 16 },
    { X86::PSRLQrr,         X86::PSRLQrm, 16 },
    { X86::PSRLWrr,         X86::PSRLWrm, 16 },
    { X86::PSUBBrr,         X86::PSUBBrm, 16 },
    { X86::PSUBDrr,         X86::PSUBDrm, 16 },
    { X86::PSUBSBrr,        X86::PSUBSBrm, 16 },
    { X86::PSUBSWrr,        X86::PSUBSWrm, 16 },
    { X86::PSUBWrr,         X86::PSUBWrm, 16 },
    { X86::PUNPCKHBWrr,     X86::PUNPCKHBWrm, 16 },
    { X86::PUNPCKHDQrr,     X86::PUNPCKHDQrm, 16 },
    { X86::PUNPCKHQDQrr,    X86::PUNPCKHQDQrm, 16 },
    { X86::PUNPCKHWDrr,     X86::PUNPCKHWDrm, 16 },
    { X86::PUNPCKLBWrr,     X86::PUNPCKLBWrm, 16 },
    { X86::PUNPCKLDQrr,     X86::PUNPCKLDQrm, 16 },
    { X86::PUNPCKLQDQrr,    X86::PUNPCKLQDQrm, 16 },
    { X86::PUNPCKLWDrr,     X86::PUNPCKLWDrm, 16 },
    { X86::PXORrr,          X86::PXORrm, 16 },
    { X86::SBB32rr,         X86::SBB32rm, 0 },
    { X86::SBB64rr,         X86::SBB64rm, 0 },
    { X86::SHUFPDrri,       X86::SHUFPDrmi, 16 },
    { X86::SHUFPSrri,       X86::SHUFPSrmi, 16 },
    { X86::SUB16rr,         X86::SUB16rm, 0 },
    { X86::SUB32rr,         X86::SUB32rm, 0 },
    { X86::SUB64rr,         X86::SUB64rm, 0 },
    { X86::SUB8rr,          X86::SUB8rm, 0 },
    { X86::SUBPDrr,         X86::SUBPDrm, 16 },
    { X86::SUBPSrr,         X86::SUBPSrm, 16 },
    { X86::SUBSDrr,         X86::SUBSDrm, 0 },
    { X86::SUBSSrr,         X86::SUBSSrm, 0 },
    // FIXME: TEST*rr -> swapped operand of TEST*mr.
    { X86::UNPCKHPDrr,      X86::UNPCKHPDrm, 16 },
    { X86::UNPCKHPSrr,      X86::UNPCKHPSrm, 16 },
    { X86::UNPCKLPDrr,      X86::UNPCKLPDrm, 16 },
    { X86::UNPCKLPSrr,      X86::UNPCKLPSrm, 16 },
    { X86::XOR16rr,         X86::XOR16rm, 0 },
    { X86::XOR32rr,         X86::XOR32rm, 0 },
    { X86::XOR64rr,         X86::XOR64rm, 0 },
    { X86::XOR8rr,          X86::XOR8rm, 0 },
    { X86::XORPDrr,         X86::XORPDrm, 16 },
    { X86::XORPSrr,         X86::XORPSrm, 16 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2); i != e; ++i) {
    unsigned RegOp = OpTbl2[i][0];
    unsigned MemOp = OpTbl2[i][1];
    unsigned Align = OpTbl2[i][2];
    if (!RegOp2MemOpTable2.insert(std::make_pair((unsigned*)RegOp,
                                           std::make_pair(MemOp,Align))).second)
      assert(false && "Duplicated entries?");
    // Index 2, folded load
    unsigned AuxInfo = 2 | (1 << 4);
    if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                   std::make_pair(RegOp, AuxInfo))).second)
      AmbEntries.push_back(MemOp);
  }

  // Remove ambiguous entries.
  assert(AmbEntries.empty() && "Duplicated entries in unfolding maps?");
}

bool X86InstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned &SrcReg, unsigned &DstReg,
                               unsigned &SrcSubIdx, unsigned &DstSubIdx) const {
  switch (MI.getOpcode()) {
  default:
    return false;
  case X86::MOV8rr:
  case X86::MOV8rr_NOREX:
  case X86::MOV16rr:
  case X86::MOV32rr: 
  case X86::MOV64rr:
  case X86::MOVSSrr:
  case X86::MOVSDrr:

  // FP Stack register class copies
  case X86::MOV_Fp3232: case X86::MOV_Fp6464: case X86::MOV_Fp8080:
  case X86::MOV_Fp3264: case X86::MOV_Fp3280:
  case X86::MOV_Fp6432: case X86::MOV_Fp8032:
      
  case X86::FsMOVAPSrr:
  case X86::FsMOVAPDrr:
  case X86::MOVAPSrr:
  case X86::MOVAPDrr:
  case X86::MOVDQArr:
  case X86::MOVSS2PSrr:
  case X86::MOVSD2PDrr:
  case X86::MOVPS2SSrr:
  case X86::MOVPD2SDrr:
  case X86::MMX_MOVQ64rr:
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "invalid register-register move instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SrcSubIdx = MI.getOperand(1).getSubReg();
    DstSubIdx = MI.getOperand(0).getSubReg();
    return true;
  }
}

unsigned X86InstrInfo::isLoadFromStackSlot(const MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    if (MI->getOperand(1).isFI() && MI->getOperand(2).isImm() &&
        MI->getOperand(3).isReg() && MI->getOperand(4).isImm() &&
        MI->getOperand(2).getImm() == 1 &&
        MI->getOperand(3).getReg() == 0 &&
        MI->getOperand(4).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8mr:
  case X86::MOV16mr:
  case X86::MOV32mr:
  case X86::MOV64mr:
  case X86::ST_FpP64m:
  case X86::MOVSSmr:
  case X86::MOVSDmr:
  case X86::MOVAPSmr:
  case X86::MOVAPDmr:
  case X86::MOVDQAmr:
  case X86::MMX_MOVD64mr:
  case X86::MMX_MOVQ64mr:
  case X86::MMX_MOVNTQmr:
    if (MI->getOperand(0).isFI() && MI->getOperand(1).isImm() &&
        MI->getOperand(2).isReg() && MI->getOperand(3).isImm() &&
        MI->getOperand(1).getImm() == 1 &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(X86AddrNumOperands).getReg();
    }
    break;
  }
  return 0;
}

/// regIsPICBase - Return true if register is PIC base (i.e.g defined by
/// X86::MOVPC32r.
static bool regIsPICBase(unsigned BaseReg, const MachineRegisterInfo &MRI) {
  bool isPICBase = false;
  for (MachineRegisterInfo::def_iterator I = MRI.def_begin(BaseReg),
         E = MRI.def_end(); I != E; ++I) {
    MachineInstr *DefMI = I.getOperand().getParent();
    if (DefMI->getOpcode() != X86::MOVPC32r)
      return false;
    assert(!isPICBase && "More than one PIC base?");
    isPICBase = true;
  }
  return isPICBase;
}

/// CanRematLoadWithDispOperand - Return true if a load with the specified
/// operand is a candidate for remat: for this to be true we need to know that
/// the load will always return the same value, even if moved.
static bool CanRematLoadWithDispOperand(const MachineOperand &MO,
                                        X86TargetMachine &TM) {
  // Loads from constant pool entries can be remat'd.
  if (MO.isCPI()) return true;
  
  // We can remat globals in some cases.
  if (MO.isGlobal()) {
    // If this is a load of a stub, not of the global, we can remat it.  This
    // access will always return the address of the global.
    if (isGlobalStubReference(MO.getTargetFlags()))
      return true;
    
    // If the global itself is constant, we can remat the load.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(MO.getGlobal()))
      if (GV->isConstant())
        return true;
  }
  return false;
}
 
bool
X86InstrInfo::isReallyTriviallyReMaterializable(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default: break;
    case X86::MOV8rm:
    case X86::MOV16rm:
    case X86::MOV32rm:
    case X86::MOV64rm:
    case X86::LD_Fp64m:
    case X86::MOVSSrm:
    case X86::MOVSDrm:
    case X86::MOVAPSrm:
    case X86::MOVAPDrm:
    case X86::MOVDQArm:
    case X86::MMX_MOVD64rm:
    case X86::MMX_MOVQ64rm: {
      // Loads from constant pools are trivially rematerializable.
      if (MI->getOperand(1).isReg() &&
          MI->getOperand(2).isImm() &&
          MI->getOperand(3).isReg() && MI->getOperand(3).getReg() == 0 &&
          CanRematLoadWithDispOperand(MI->getOperand(4), TM)) {
        unsigned BaseReg = MI->getOperand(1).getReg();
        if (BaseReg == 0 || BaseReg == X86::RIP)
          return true;
        // Allow re-materialization of PIC load.
        if (!ReMatPICStubLoad && MI->getOperand(4).isGlobal())
          return false;
        const MachineFunction &MF = *MI->getParent()->getParent();
        const MachineRegisterInfo &MRI = MF.getRegInfo();
        bool isPICBase = false;
        for (MachineRegisterInfo::def_iterator I = MRI.def_begin(BaseReg),
               E = MRI.def_end(); I != E; ++I) {
          MachineInstr *DefMI = I.getOperand().getParent();
          if (DefMI->getOpcode() != X86::MOVPC32r)
            return false;
          assert(!isPICBase && "More than one PIC base?");
          isPICBase = true;
        }
        return isPICBase;
      } 
      return false;
    }
 
     case X86::LEA32r:
     case X86::LEA64r: {
       if (MI->getOperand(2).isImm() &&
           MI->getOperand(3).isReg() && MI->getOperand(3).getReg() == 0 &&
           !MI->getOperand(4).isReg()) {
         // lea fi#, lea GV, etc. are all rematerializable.
         if (!MI->getOperand(1).isReg())
           return true;
         unsigned BaseReg = MI->getOperand(1).getReg();
         if (BaseReg == 0)
           return true;
         // Allow re-materialization of lea PICBase + x.
         const MachineFunction &MF = *MI->getParent()->getParent();
         const MachineRegisterInfo &MRI = MF.getRegInfo();
         return regIsPICBase(BaseReg, MRI);
       }
       return false;
     }
  }

  // All other instructions marked M_REMATERIALIZABLE are always trivially
  // rematerializable.
  return true;
}

/// isSafeToClobberEFLAGS - Return true if it's safe insert an instruction that
/// would clobber the EFLAGS condition register. Note the result may be
/// conservative. If it cannot definitely determine the safety after visiting
/// two instructions it assumes it's not safe.
static bool isSafeToClobberEFLAGS(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) {
  // It's always safe to clobber EFLAGS at the end of a block.
  if (I == MBB.end())
    return true;

  // For compile time consideration, if we are not able to determine the
  // safety after visiting 2 instructions, we will assume it's not safe.
  for (unsigned i = 0; i < 2; ++i) {
    bool SeenDef = false;
    for (unsigned j = 0, e = I->getNumOperands(); j != e; ++j) {
      MachineOperand &MO = I->getOperand(j);
      if (!MO.isReg())
        continue;
      if (MO.getReg() == X86::EFLAGS) {
        if (MO.isUse())
          return false;
        SeenDef = true;
      }
    }

    if (SeenDef)
      // This instruction defines EFLAGS, no need to look any further.
      return true;
    ++I;

    // If we make it to the end of the block, it's safe to clobber EFLAGS.
    if (I == MBB.end())
      return true;
  }

  // Conservative answer.
  return false;
}

void X86InstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I,
                                 unsigned DestReg, unsigned SubIdx,
                                 const MachineInstr *Orig) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (SubIdx && TargetRegisterInfo::isPhysicalRegister(DestReg)) {
    DestReg = RI.getSubReg(DestReg, SubIdx);
    SubIdx = 0;
  }

  // MOV32r0 etc. are implemented with xor which clobbers condition code.
  // Re-materialize them as movri instructions to avoid side effects.
  bool Clone = true;
  unsigned Opc = Orig->getOpcode();
  switch (Opc) {
  default: break;
  case X86::MOV8r0:
  case X86::MOV16r0:
  case X86::MOV32r0: {
    if (!isSafeToClobberEFLAGS(MBB, I)) {
      switch (Opc) {
      default: break;
      case X86::MOV8r0:  Opc = X86::MOV8ri;  break;
      case X86::MOV16r0: Opc = X86::MOV16ri; break;
      case X86::MOV32r0: Opc = X86::MOV32ri; break;
      }
      Clone = false;
    }
    break;
  }
  }

  if (Clone) {
    MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
    MI->getOperand(0).setReg(DestReg);
    MBB.insert(I, MI);
  } else {
    BuildMI(MBB, I, DL, get(Opc), DestReg).addImm(0);
  }

  MachineInstr *NewMI = prior(I);
  NewMI->getOperand(0).setSubReg(SubIdx);
}

/// isInvariantLoad - Return true if the specified instruction (which is marked
/// mayLoad) is loading from a location whose value is invariant across the
/// function.  For example, loading a value from the constant pool or from
/// from the argument area of a function if it does not change.  This should
/// only return true of *all* loads the instruction does are invariant (if it
/// does multiple loads).
bool X86InstrInfo::isInvariantLoad(const MachineInstr *MI) const {
  // This code cares about loads from three cases: constant pool entries,
  // invariant argument slots, and global stubs.  In order to handle these cases
  // for all of the myriad of X86 instructions, we just scan for a CP/FI/GV
  // operand and base our analysis on it.  This is safe because the address of
  // none of these three cases is ever used as anything other than a load base
  // and X86 doesn't have any instructions that load from multiple places.
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    // Loads from constant pools are trivially invariant.
    if (MO.isCPI())
      return true;

    if (MO.isGlobal())
      return isGlobalStubReference(MO.getTargetFlags());

    // If this is a load from an invariant stack slot, the load is a constant.
    if (MO.isFI()) {
      const MachineFrameInfo &MFI =
        *MI->getParent()->getParent()->getFrameInfo();
      int Idx = MO.getIndex();
      return MFI.isFixedObjectIndex(Idx) && MFI.isImmutableObjectIndex(Idx);
    }
  }
  
  // All other instances of these instructions are presumed to have other
  // issues.
  return false;
}

/// hasLiveCondCodeDef - True if MI has a condition code def, e.g. EFLAGS, that
/// is not marked dead.
static bool hasLiveCondCodeDef(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        MO.getReg() == X86::EFLAGS && !MO.isDead()) {
      return true;
    }
  }
  return false;
}

/// convertToThreeAddress - This method must be implemented by targets that
/// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
/// may be able to convert a two-address instruction into a true
/// three-address instruction on demand.  This allows the X86 target (for
/// example) to convert ADD and SHL instructions into LEA instructions if they
/// would require register copies due to two-addressness.
///
/// This method returns a null pointer if the transformation cannot be
/// performed, otherwise it returns the new instruction.
///
MachineInstr *
X86InstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                    MachineBasicBlock::iterator &MBBI,
                                    LiveVariables *LV) const {
  MachineInstr *MI = MBBI;
  MachineFunction &MF = *MI->getParent()->getParent();
  // All instructions input are two-addr instructions.  Get the known operands.
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();
  bool isDead = MI->getOperand(0).isDead();
  bool isKill = MI->getOperand(1).isKill();

  MachineInstr *NewMI = NULL;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  bool DisableLEA16 = true;

  unsigned MIOpc = MI->getOpcode();
  switch (MIOpc) {
  case X86::SHUFPSrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufps instruction!");
    if (!TM.getSubtarget<X86Subtarget>().hasSSE2()) return 0;
    
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    if (B != C) return 0;
    unsigned A = MI->getOperand(0).getReg();
    unsigned M = MI->getOperand(3).getImm();
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::PSHUFDri))
      .addReg(A, RegState::Define | getDeadRegState(isDead))
      .addReg(B, getKillRegState(isKill)).addImm(M);
    break;
  }
  case X86::SHL64ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;

    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
      .addReg(Dest, RegState::Define | getDeadRegState(isDead))
      .addReg(0).addImm(1 << ShAmt)
      .addReg(Src, getKillRegState(isKill))
      .addImm(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;

    unsigned Opc = TM.getSubtarget<X86Subtarget>().is64Bit() ?
      X86::LEA64_32r : X86::LEA32r;
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(Opc))
      .addReg(Dest, RegState::Define | getDeadRegState(isDead))
      .addReg(0).addImm(1 << ShAmt)
      .addReg(Src, getKillRegState(isKill)).addImm(0);
    break;
  }
  case X86::SHL16ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;

    if (DisableLEA16) {
      // If 16-bit LEA is disabled, use 32-bit LEA via subregisters.
      MachineRegisterInfo &RegInfo = MFI->getParent()->getRegInfo();
      unsigned Opc = TM.getSubtarget<X86Subtarget>().is64Bit()
        ? X86::LEA64_32r : X86::LEA32r;
      unsigned leaInReg = RegInfo.createVirtualRegister(&X86::GR32RegClass);
      unsigned leaOutReg = RegInfo.createVirtualRegister(&X86::GR32RegClass);
            
      // Build and insert into an implicit UNDEF value. This is OK because
      // well be shifting and then extracting the lower 16-bits. 
      BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(X86::IMPLICIT_DEF), leaInReg);
      MachineInstr *InsMI =
        BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(X86::INSERT_SUBREG),leaInReg)
        .addReg(leaInReg)
        .addReg(Src, getKillRegState(isKill))
        .addImm(X86::SUBREG_16BIT);
      
      NewMI = BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(Opc), leaOutReg)
        .addReg(0).addImm(1 << ShAmt)
        .addReg(leaInReg, RegState::Kill)
        .addImm(0);
      
      MachineInstr *ExtMI =
        BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(X86::EXTRACT_SUBREG))
        .addReg(Dest, RegState::Define | getDeadRegState(isDead))
        .addReg(leaOutReg, RegState::Kill)
        .addImm(X86::SUBREG_16BIT);

      if (LV) {
        // Update live variables
        LV->getVarInfo(leaInReg).Kills.push_back(NewMI);
        LV->getVarInfo(leaOutReg).Kills.push_back(ExtMI);
        if (isKill)
          LV->replaceKillInstruction(Src, MI, InsMI);
        if (isDead)
          LV->replaceKillInstruction(Dest, MI, ExtMI);
      }
      return ExtMI;
    } else {
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
        .addReg(Dest, RegState::Define | getDeadRegState(isDead))
        .addReg(0).addImm(1 << ShAmt)
        .addReg(Src, getKillRegState(isKill))
        .addImm(0);
    }
    break;
  }
  default: {
    // The following opcodes also sets the condition code register(s). Only
    // convert them to equivalent lea if the condition code register def's
    // are dead!
    if (hasLiveCondCodeDef(MI))
      return 0;

    bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();
    switch (MIOpc) {
    default: return 0;
    case X86::INC64r:
    case X86::INC32r:
    case X86::INC64_32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      unsigned Opc = MIOpc == X86::INC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      NewMI = addLeaRegOffset(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                              Src, isKill, 1);
      break;
    }
    case X86::INC16r:
    case X86::INC64_16r:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                           .addReg(Dest, RegState::Define |
                                   getDeadRegState(isDead)),
                           Src, isKill, 1);
      break;
    case X86::DEC64r:
    case X86::DEC32r:
    case X86::DEC64_32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      unsigned Opc = MIOpc == X86::DEC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      NewMI = addLeaRegOffset(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                              Src, isKill, -1);
      break;
    }
    case X86::DEC16r:
    case X86::DEC64_16r:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                           .addReg(Dest, RegState::Define |
                                   getDeadRegState(isDead)),
                           Src, isKill, -1);
      break;
    case X86::ADD64rr:
    case X86::ADD32rr: {
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      unsigned Opc = MIOpc == X86::ADD64rr ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      unsigned Src2 = MI->getOperand(2).getReg();
      bool isKill2 = MI->getOperand(2).isKill();
      NewMI = addRegReg(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                        .addReg(Dest, RegState::Define |
                                getDeadRegState(isDead)),
                        Src, isKill, Src2, isKill2);
      if (LV && isKill2)
        LV->replaceKillInstruction(Src2, MI, NewMI);
      break;
    }
    case X86::ADD16rr: {
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      unsigned Src2 = MI->getOperand(2).getReg();
      bool isKill2 = MI->getOperand(2).isKill();
      NewMI = addRegReg(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                        .addReg(Dest, RegState::Define |
                                getDeadRegState(isDead)),
                        Src, isKill, Src2, isKill2);
      if (LV && isKill2)
        LV->replaceKillInstruction(Src2, MI, NewMI);
      break;
    }
    case X86::ADD64ri32:
    case X86::ADD64ri8:
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      if (MI->getOperand(2).isImm())
        NewMI = addLeaRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
                                .addReg(Dest, RegState::Define |
                                        getDeadRegState(isDead)),
                                Src, isKill, MI->getOperand(2).getImm());
      break;
    case X86::ADD32ri:
    case X86::ADD32ri8:
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      if (MI->getOperand(2).isImm()) {
        unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;
        NewMI = addLeaRegOffset(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                                .addReg(Dest, RegState::Define |
                                        getDeadRegState(isDead)),
                                Src, isKill, MI->getOperand(2).getImm());
      }
      break;
    case X86::ADD16ri:
    case X86::ADD16ri8:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      if (MI->getOperand(2).isImm())
        NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                             .addReg(Dest, RegState::Define |
                                     getDeadRegState(isDead)),
                             Src, isKill, MI->getOperand(2).getImm());
      break;
    case X86::SHL16ri:
      if (DisableLEA16) return 0;
    case X86::SHL32ri:
    case X86::SHL64ri: {
      assert(MI->getNumOperands() >= 3 && MI->getOperand(2).isImm() &&
             "Unknown shl instruction!");
      unsigned ShAmt = MI->getOperand(2).getImm();
      if (ShAmt == 1 || ShAmt == 2 || ShAmt == 3) {
        X86AddressMode AM;
        AM.Scale = 1 << ShAmt;
        AM.IndexReg = Src;
        unsigned Opc = MIOpc == X86::SHL64ri ? X86::LEA64r
          : (MIOpc == X86::SHL32ri
             ? (is64Bit ? X86::LEA64_32r : X86::LEA32r) : X86::LEA16r);
        NewMI = addFullAddress(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                               .addReg(Dest, RegState::Define |
                                       getDeadRegState(isDead)), AM);
        if (isKill)
          NewMI->getOperand(3).setIsKill(true);
      }
      break;
    }
    }
  }
  }

  if (!NewMI) return 0;

  if (LV) {  // Update live variables
    if (isKill)
      LV->replaceKillInstruction(Src, MI, NewMI);
    if (isDead)
      LV->replaceKillInstruction(Dest, MI, NewMI);
  }

  MFI->insert(MBBI, NewMI);          // Insert the new inst    
  return NewMI;
}

/// commuteInstruction - We have a few instructions that must be hacked on to
/// commute them.
///
MachineInstr *
X86InstrInfo::commuteInstruction(MachineInstr *MI, bool NewMI) const {
  switch (MI->getOpcode()) {
  case X86::SHRD16rri8: // A = SHRD16rri8 B, C, I -> A = SHLD16rri8 C, B, (16-I)
  case X86::SHLD16rri8: // A = SHLD16rri8 B, C, I -> A = SHRD16rri8 C, B, (16-I)
  case X86::SHRD32rri8: // A = SHRD32rri8 B, C, I -> A = SHLD32rri8 C, B, (32-I)
  case X86::SHLD32rri8: // A = SHLD32rri8 B, C, I -> A = SHRD32rri8 C, B, (32-I)
  case X86::SHRD64rri8: // A = SHRD64rri8 B, C, I -> A = SHLD64rri8 C, B, (64-I)
  case X86::SHLD64rri8:{// A = SHLD64rri8 B, C, I -> A = SHRD64rri8 C, B, (64-I)
    unsigned Opc;
    unsigned Size;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Unreachable!");
    case X86::SHRD16rri8: Size = 16; Opc = X86::SHLD16rri8; break;
    case X86::SHLD16rri8: Size = 16; Opc = X86::SHRD16rri8; break;
    case X86::SHRD32rri8: Size = 32; Opc = X86::SHLD32rri8; break;
    case X86::SHLD32rri8: Size = 32; Opc = X86::SHRD32rri8; break;
    case X86::SHRD64rri8: Size = 64; Opc = X86::SHLD64rri8; break;
    case X86::SHLD64rri8: Size = 64; Opc = X86::SHRD64rri8; break;
    }
    unsigned Amt = MI->getOperand(3).getImm();
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->setDesc(get(Opc));
    MI->getOperand(3).setImm(Size-Amt);
    return TargetInstrInfoImpl::commuteInstruction(MI, NewMI);
  }
  case X86::CMOVB16rr:
  case X86::CMOVB32rr:
  case X86::CMOVB64rr:
  case X86::CMOVAE16rr:
  case X86::CMOVAE32rr:
  case X86::CMOVAE64rr:
  case X86::CMOVE16rr:
  case X86::CMOVE32rr:
  case X86::CMOVE64rr:
  case X86::CMOVNE16rr:
  case X86::CMOVNE32rr:
  case X86::CMOVNE64rr:
  case X86::CMOVBE16rr:
  case X86::CMOVBE32rr:
  case X86::CMOVBE64rr:
  case X86::CMOVA16rr:
  case X86::CMOVA32rr:
  case X86::CMOVA64rr:
  case X86::CMOVL16rr:
  case X86::CMOVL32rr:
  case X86::CMOVL64rr:
  case X86::CMOVGE16rr:
  case X86::CMOVGE32rr:
  case X86::CMOVGE64rr:
  case X86::CMOVLE16rr:
  case X86::CMOVLE32rr:
  case X86::CMOVLE64rr:
  case X86::CMOVG16rr:
  case X86::CMOVG32rr:
  case X86::CMOVG64rr:
  case X86::CMOVS16rr:
  case X86::CMOVS32rr:
  case X86::CMOVS64rr:
  case X86::CMOVNS16rr:
  case X86::CMOVNS32rr:
  case X86::CMOVNS64rr:
  case X86::CMOVP16rr:
  case X86::CMOVP32rr:
  case X86::CMOVP64rr:
  case X86::CMOVNP16rr:
  case X86::CMOVNP32rr:
  case X86::CMOVNP64rr:
  case X86::CMOVO16rr:
  case X86::CMOVO32rr:
  case X86::CMOVO64rr:
  case X86::CMOVNO16rr:
  case X86::CMOVNO32rr:
  case X86::CMOVNO64rr: {
    unsigned Opc = 0;
    switch (MI->getOpcode()) {
    default: break;
    case X86::CMOVB16rr:  Opc = X86::CMOVAE16rr; break;
    case X86::CMOVB32rr:  Opc = X86::CMOVAE32rr; break;
    case X86::CMOVB64rr:  Opc = X86::CMOVAE64rr; break;
    case X86::CMOVAE16rr: Opc = X86::CMOVB16rr; break;
    case X86::CMOVAE32rr: Opc = X86::CMOVB32rr; break;
    case X86::CMOVAE64rr: Opc = X86::CMOVB64rr; break;
    case X86::CMOVE16rr:  Opc = X86::CMOVNE16rr; break;
    case X86::CMOVE32rr:  Opc = X86::CMOVNE32rr; break;
    case X86::CMOVE64rr:  Opc = X86::CMOVNE64rr; break;
    case X86::CMOVNE16rr: Opc = X86::CMOVE16rr; break;
    case X86::CMOVNE32rr: Opc = X86::CMOVE32rr; break;
    case X86::CMOVNE64rr: Opc = X86::CMOVE64rr; break;
    case X86::CMOVBE16rr: Opc = X86::CMOVA16rr; break;
    case X86::CMOVBE32rr: Opc = X86::CMOVA32rr; break;
    case X86::CMOVBE64rr: Opc = X86::CMOVA64rr; break;
    case X86::CMOVA16rr:  Opc = X86::CMOVBE16rr; break;
    case X86::CMOVA32rr:  Opc = X86::CMOVBE32rr; break;
    case X86::CMOVA64rr:  Opc = X86::CMOVBE64rr; break;
    case X86::CMOVL16rr:  Opc = X86::CMOVGE16rr; break;
    case X86::CMOVL32rr:  Opc = X86::CMOVGE32rr; break;
    case X86::CMOVL64rr:  Opc = X86::CMOVGE64rr; break;
    case X86::CMOVGE16rr: Opc = X86::CMOVL16rr; break;
    case X86::CMOVGE32rr: Opc = X86::CMOVL32rr; break;
    case X86::CMOVGE64rr: Opc = X86::CMOVL64rr; break;
    case X86::CMOVLE16rr: Opc = X86::CMOVG16rr; break;
    case X86::CMOVLE32rr: Opc = X86::CMOVG32rr; break;
    case X86::CMOVLE64rr: Opc = X86::CMOVG64rr; break;
    case X86::CMOVG16rr:  Opc = X86::CMOVLE16rr; break;
    case X86::CMOVG32rr:  Opc = X86::CMOVLE32rr; break;
    case X86::CMOVG64rr:  Opc = X86::CMOVLE64rr; break;
    case X86::CMOVS16rr:  Opc = X86::CMOVNS16rr; break;
    case X86::CMOVS32rr:  Opc = X86::CMOVNS32rr; break;
    case X86::CMOVS64rr:  Opc = X86::CMOVNS64rr; break;
    case X86::CMOVNS16rr: Opc = X86::CMOVS16rr; break;
    case X86::CMOVNS32rr: Opc = X86::CMOVS32rr; break;
    case X86::CMOVNS64rr: Opc = X86::CMOVS64rr; break;
    case X86::CMOVP16rr:  Opc = X86::CMOVNP16rr; break;
    case X86::CMOVP32rr:  Opc = X86::CMOVNP32rr; break;
    case X86::CMOVP64rr:  Opc = X86::CMOVNP64rr; break;
    case X86::CMOVNP16rr: Opc = X86::CMOVP16rr; break;
    case X86::CMOVNP32rr: Opc = X86::CMOVP32rr; break;
    case X86::CMOVNP64rr: Opc = X86::CMOVP64rr; break;
    case X86::CMOVO16rr:  Opc = X86::CMOVNO16rr; break;
    case X86::CMOVO32rr:  Opc = X86::CMOVNO32rr; break;
    case X86::CMOVO64rr:  Opc = X86::CMOVNO64rr; break;
    case X86::CMOVNO16rr: Opc = X86::CMOVO16rr; break;
    case X86::CMOVNO32rr: Opc = X86::CMOVO32rr; break;
    case X86::CMOVNO64rr: Opc = X86::CMOVO64rr; break;
    }
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->setDesc(get(Opc));
    // Fallthrough intended.
  }
  default:
    return TargetInstrInfoImpl::commuteInstruction(MI, NewMI);
  }
}

static X86::CondCode GetCondFromBranchOpc(unsigned BrOpc) {
  switch (BrOpc) {
  default: return X86::COND_INVALID;
  case X86::JE:  return X86::COND_E;
  case X86::JNE: return X86::COND_NE;
  case X86::JL:  return X86::COND_L;
  case X86::JLE: return X86::COND_LE;
  case X86::JG:  return X86::COND_G;
  case X86::JGE: return X86::COND_GE;
  case X86::JB:  return X86::COND_B;
  case X86::JBE: return X86::COND_BE;
  case X86::JA:  return X86::COND_A;
  case X86::JAE: return X86::COND_AE;
  case X86::JS:  return X86::COND_S;
  case X86::JNS: return X86::COND_NS;
  case X86::JP:  return X86::COND_P;
  case X86::JNP: return X86::COND_NP;
  case X86::JO:  return X86::COND_O;
  case X86::JNO: return X86::COND_NO;
  }
}

unsigned X86::GetCondBranchFromCond(X86::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case X86::COND_E:  return X86::JE;
  case X86::COND_NE: return X86::JNE;
  case X86::COND_L:  return X86::JL;
  case X86::COND_LE: return X86::JLE;
  case X86::COND_G:  return X86::JG;
  case X86::COND_GE: return X86::JGE;
  case X86::COND_B:  return X86::JB;
  case X86::COND_BE: return X86::JBE;
  case X86::COND_A:  return X86::JA;
  case X86::COND_AE: return X86::JAE;
  case X86::COND_S:  return X86::JS;
  case X86::COND_NS: return X86::JNS;
  case X86::COND_P:  return X86::JP;
  case X86::COND_NP: return X86::JNP;
  case X86::COND_O:  return X86::JO;
  case X86::COND_NO: return X86::JNO;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified condition,
/// e.g. turning COND_E to COND_NE.
X86::CondCode X86::GetOppositeBranchCondition(X86::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case X86::COND_E:  return X86::COND_NE;
  case X86::COND_NE: return X86::COND_E;
  case X86::COND_L:  return X86::COND_GE;
  case X86::COND_LE: return X86::COND_G;
  case X86::COND_G:  return X86::COND_LE;
  case X86::COND_GE: return X86::COND_L;
  case X86::COND_B:  return X86::COND_AE;
  case X86::COND_BE: return X86::COND_A;
  case X86::COND_A:  return X86::COND_BE;
  case X86::COND_AE: return X86::COND_B;
  case X86::COND_S:  return X86::COND_NS;
  case X86::COND_NS: return X86::COND_S;
  case X86::COND_P:  return X86::COND_NP;
  case X86::COND_NP: return X86::COND_P;
  case X86::COND_O:  return X86::COND_NO;
  case X86::COND_NO: return X86::COND_O;
  }
}

bool X86InstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isTerminator()) return false;
  
  // Conditional branch is a special case.
  if (TID.isBranch() && !TID.isBarrier())
    return true;
  if (!TID.isPredicable())
    return true;
  return !isPredicated(MI);
}

// For purposes of branch analysis do not count FP_REG_KILL as a terminator.
static bool isBrAnalysisUnpredicatedTerminator(const MachineInstr *MI,
                                               const X86InstrInfo &TII) {
  if (MI->getOpcode() == X86::FP_REG_KILL)
    return false;
  return TII.isUnpredicatedTerminator(MI);
}

bool X86InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, 
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isBrAnalysisUnpredicatedTerminator(I, *this))
      break;
    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!I->getDesc().isBranch())
      return true;
    // Handle unconditional branches.
    if (I->getOpcode() == X86::JMP) {
      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      while (next(I) != MBB.end())
        next(I)->eraseFromParent();
      Cond.clear();
      FBB = 0;
      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = 0;
        I->eraseFromParent();
        I = MBB.end();
        continue;
      }
      // TBB is used to indicate the unconditinal destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }
    // Handle conditional branches.
    X86::CondCode BranchCode = GetCondFromBranchOpc(I->getOpcode());
    if (BranchCode == X86::COND_INVALID)
      return true;  // Can't handle indirect branch.
    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      FBB = TBB;
      TBB = I->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));
      continue;
    }
    // Handle subsequent conditional branches. Only handle the case
    // where all conditional branches branch to the same destination
    // and their condition opcodes fit one of the special
    // multi-branch idioms.
    assert(Cond.size() == 1);
    assert(TBB);
    // Only handle the case where all conditional branches branch to
    // the same destination.
    if (TBB != I->getOperand(0).getMBB())
      return true;
    X86::CondCode OldBranchCode = (X86::CondCode)Cond[0].getImm();
    // If the conditions are the same, we can leave them alone.
    if (OldBranchCode == BranchCode)
      continue;
    // If they differ, see if they fit one of the known patterns.
    // Theoretically we could handle more patterns here, but
    // we shouldn't expect to see them if instruction selection
    // has done a reasonable job.
    if ((OldBranchCode == X86::COND_NP &&
         BranchCode == X86::COND_E) ||
        (OldBranchCode == X86::COND_E &&
         BranchCode == X86::COND_NP))
      BranchCode = X86::COND_NP_OR_E;
    else if ((OldBranchCode == X86::COND_P &&
              BranchCode == X86::COND_NE) ||
             (OldBranchCode == X86::COND_NE &&
              BranchCode == X86::COND_P))
      BranchCode = X86::COND_NE_OR_P;
    else
      return true;
    // Update the MachineOperand.
    Cond[0].setImm(BranchCode);
  }

  return false;
}

unsigned X86InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->getOpcode() != X86::JMP &&
        GetCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }
  
  return Count;
}

unsigned
X86InstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME this should probably have a DebugLoc operand
  DebugLoc dl = DebugLoc::getUnknownLoc();
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, dl, get(X86::JMP)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  X86::CondCode CC = (X86::CondCode)Cond[0].getImm();
  switch (CC) {
  case X86::COND_NP_OR_E:
    // Synthesize NP_OR_E with two branches.
    BuildMI(&MBB, dl, get(X86::JNP)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, dl, get(X86::JE)).addMBB(TBB);
    ++Count;
    break;
  case X86::COND_NE_OR_P:
    // Synthesize NE_OR_P with two branches.
    BuildMI(&MBB, dl, get(X86::JNE)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, dl, get(X86::JP)).addMBB(TBB);
    ++Count;
    break;
  default: {
    unsigned Opc = GetCondBranchFromCond(CC);
    BuildMI(&MBB, dl, get(Opc)).addMBB(TBB);
    ++Count;
  }
  }
  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    BuildMI(&MBB, dl, get(X86::JMP)).addMBB(FBB);
    ++Count;
  }
  return Count;
}

/// isHReg - Test if the given register is a physical h register.
static bool isHReg(unsigned Reg) {
  return X86::GR8_ABCD_HRegClass.contains(Reg);
}

bool X86InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI,
                                unsigned DestReg, unsigned SrcReg,
                                const TargetRegisterClass *DestRC,
                                const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  // Determine if DstRC and SrcRC have a common superclass in common.
  const TargetRegisterClass *CommonRC = DestRC;
  if (DestRC == SrcRC)
    /* Source and destination have the same register class. */;
  else if (CommonRC->hasSuperClass(SrcRC))
    CommonRC = SrcRC;
  else if (!DestRC->hasSubClass(SrcRC)) {
    // Neither of GR64_NOREX or GR64_NOSP is a superclass of the other,
    // but we want to copy then as GR64. Similarly, for GR32_NOREX and
    // GR32_NOSP, copy as GR32.
    if (SrcRC->hasSuperClass(&X86::GR64RegClass) &&
        DestRC->hasSuperClass(&X86::GR64RegClass))
      CommonRC = &X86::GR64RegClass;
    else if (SrcRC->hasSuperClass(&X86::GR32RegClass) &&
             DestRC->hasSuperClass(&X86::GR32RegClass))
      CommonRC = &X86::GR32RegClass;
    else
      CommonRC = 0;
  }

  if (CommonRC) {
    unsigned Opc;
    if (CommonRC == &X86::GR64RegClass || CommonRC == &X86::GR64_NOSPRegClass) {
      Opc = X86::MOV64rr;
    } else if (CommonRC == &X86::GR32RegClass ||
               CommonRC == &X86::GR32_NOSPRegClass) {
      Opc = X86::MOV32rr;
    } else if (CommonRC == &X86::GR16RegClass) {
      Opc = X86::MOV16rr;
    } else if (CommonRC == &X86::GR8RegClass) {
      // Copying to or from a physical H register on x86-64 requires a NOREX
      // move.  Otherwise use a normal move.
      if ((isHReg(DestReg) || isHReg(SrcReg)) &&
          TM.getSubtarget<X86Subtarget>().is64Bit())
        Opc = X86::MOV8rr_NOREX;
      else
        Opc = X86::MOV8rr;
    } else if (CommonRC == &X86::GR64_ABCDRegClass) {
      Opc = X86::MOV64rr;
    } else if (CommonRC == &X86::GR32_ABCDRegClass) {
      Opc = X86::MOV32rr;
    } else if (CommonRC == &X86::GR16_ABCDRegClass) {
      Opc = X86::MOV16rr;
    } else if (CommonRC == &X86::GR8_ABCD_LRegClass) {
      Opc = X86::MOV8rr;
    } else if (CommonRC == &X86::GR8_ABCD_HRegClass) {
      if (TM.getSubtarget<X86Subtarget>().is64Bit())
        Opc = X86::MOV8rr_NOREX;
      else
        Opc = X86::MOV8rr;
    } else if (CommonRC == &X86::GR64_NOREXRegClass ||
               CommonRC == &X86::GR64_NOREX_NOSPRegClass) {
      Opc = X86::MOV64rr;
    } else if (CommonRC == &X86::GR32_NOREXRegClass) {
      Opc = X86::MOV32rr;
    } else if (CommonRC == &X86::GR16_NOREXRegClass) {
      Opc = X86::MOV16rr;
    } else if (CommonRC == &X86::GR8_NOREXRegClass) {
      Opc = X86::MOV8rr;
    } else if (CommonRC == &X86::RFP32RegClass) {
      Opc = X86::MOV_Fp3232;
    } else if (CommonRC == &X86::RFP64RegClass || CommonRC == &X86::RSTRegClass) {
      Opc = X86::MOV_Fp6464;
    } else if (CommonRC == &X86::RFP80RegClass) {
      Opc = X86::MOV_Fp8080;
    } else if (CommonRC == &X86::FR32RegClass) {
      Opc = X86::FsMOVAPSrr;
    } else if (CommonRC == &X86::FR64RegClass) {
      Opc = X86::FsMOVAPDrr;
    } else if (CommonRC == &X86::VR128RegClass) {
      Opc = X86::MOVAPSrr;
    } else if (CommonRC == &X86::VR64RegClass) {
      Opc = X86::MMX_MOVQ64rr;
    } else {
      return false;
    }
    BuildMI(MBB, MI, DL, get(Opc), DestReg).addReg(SrcReg);
    return true;
  }

  // Moving EFLAGS to / from another register requires a push and a pop.
  if (SrcRC == &X86::CCRRegClass) {
    if (SrcReg != X86::EFLAGS)
      return false;
    if (DestRC == &X86::GR64RegClass || DestRC == &X86::GR64_NOSPRegClass) {
      BuildMI(MBB, MI, DL, get(X86::PUSHFQ));
      BuildMI(MBB, MI, DL, get(X86::POP64r), DestReg);
      return true;
    } else if (DestRC == &X86::GR32RegClass ||
               DestRC == &X86::GR32_NOSPRegClass) {
      BuildMI(MBB, MI, DL, get(X86::PUSHFD));
      BuildMI(MBB, MI, DL, get(X86::POP32r), DestReg);
      return true;
    }
  } else if (DestRC == &X86::CCRRegClass) {
    if (DestReg != X86::EFLAGS)
      return false;
    if (SrcRC == &X86::GR64RegClass || DestRC == &X86::GR64_NOSPRegClass) {
      BuildMI(MBB, MI, DL, get(X86::PUSH64r)).addReg(SrcReg);
      BuildMI(MBB, MI, DL, get(X86::POPFQ));
      return true;
    } else if (SrcRC == &X86::GR32RegClass ||
               DestRC == &X86::GR32_NOSPRegClass) {
      BuildMI(MBB, MI, DL, get(X86::PUSH32r)).addReg(SrcReg);
      BuildMI(MBB, MI, DL, get(X86::POPFD));
      return true;
    }
  }

  // Moving from ST(0) turns into FpGET_ST0_32 etc.
  if (SrcRC == &X86::RSTRegClass) {
    // Copying from ST(0)/ST(1).
    if (SrcReg != X86::ST0 && SrcReg != X86::ST1)
      // Can only copy from ST(0)/ST(1) right now
      return false;
    bool isST0 = SrcReg == X86::ST0;
    unsigned Opc;
    if (DestRC == &X86::RFP32RegClass)
      Opc = isST0 ? X86::FpGET_ST0_32 : X86::FpGET_ST1_32;
    else if (DestRC == &X86::RFP64RegClass)
      Opc = isST0 ? X86::FpGET_ST0_64 : X86::FpGET_ST1_64;
    else {
      if (DestRC != &X86::RFP80RegClass)
        return false;
      Opc = isST0 ? X86::FpGET_ST0_80 : X86::FpGET_ST1_80;
    }
    BuildMI(MBB, MI, DL, get(Opc), DestReg);
    return true;
  }

  // Moving to ST(0) turns into FpSET_ST0_32 etc.
  if (DestRC == &X86::RSTRegClass) {
    // Copying to ST(0) / ST(1).
    if (DestReg != X86::ST0 && DestReg != X86::ST1)
      // Can only copy to TOS right now
      return false;
    bool isST0 = DestReg == X86::ST0;
    unsigned Opc;
    if (SrcRC == &X86::RFP32RegClass)
      Opc = isST0 ? X86::FpSET_ST0_32 : X86::FpSET_ST1_32;
    else if (SrcRC == &X86::RFP64RegClass)
      Opc = isST0 ? X86::FpSET_ST0_64 : X86::FpSET_ST1_64;
    else {
      if (SrcRC != &X86::RFP80RegClass)
        return false;
      Opc = isST0 ? X86::FpSET_ST0_80 : X86::FpSET_ST1_80;
    }
    BuildMI(MBB, MI, DL, get(Opc)).addReg(SrcReg);
    return true;
  }
  
  // Not yet supported!
  return false;
}

static unsigned getStoreRegOpcode(unsigned SrcReg,
                                  const TargetRegisterClass *RC,
                                  bool isStackAligned,
                                  TargetMachine &TM) {
  unsigned Opc = 0;
  if (RC == &X86::GR64RegClass || RC == &X86::GR64_NOSPRegClass) {
    Opc = X86::MOV64mr;
  } else if (RC == &X86::GR32RegClass || RC == &X86::GR32_NOSPRegClass) {
    Opc = X86::MOV32mr;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16mr;
  } else if (RC == &X86::GR8RegClass) {
    // Copying to or from a physical H register on x86-64 requires a NOREX
    // move.  Otherwise use a normal move.
    if (isHReg(SrcReg) &&
        TM.getSubtarget<X86Subtarget>().is64Bit())
      Opc = X86::MOV8mr_NOREX;
    else
      Opc = X86::MOV8mr;
  } else if (RC == &X86::GR64_ABCDRegClass) {
    Opc = X86::MOV64mr;
  } else if (RC == &X86::GR32_ABCDRegClass) {
    Opc = X86::MOV32mr;
  } else if (RC == &X86::GR16_ABCDRegClass) {
    Opc = X86::MOV16mr;
  } else if (RC == &X86::GR8_ABCD_LRegClass) {
    Opc = X86::MOV8mr;
  } else if (RC == &X86::GR8_ABCD_HRegClass) {
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      Opc = X86::MOV8mr_NOREX;
    else
      Opc = X86::MOV8mr;
  } else if (RC == &X86::GR64_NOREXRegClass ||
             RC == &X86::GR64_NOREX_NOSPRegClass) {
    Opc = X86::MOV64mr;
  } else if (RC == &X86::GR32_NOREXRegClass) {
    Opc = X86::MOV32mr;
  } else if (RC == &X86::GR16_NOREXRegClass) {
    Opc = X86::MOV16mr;
  } else if (RC == &X86::GR8_NOREXRegClass) {
    Opc = X86::MOV8mr;
  } else if (RC == &X86::RFP80RegClass) {
    Opc = X86::ST_FpP80m;   // pops
  } else if (RC == &X86::RFP64RegClass) {
    Opc = X86::ST_Fp64m;
  } else if (RC == &X86::RFP32RegClass) {
    Opc = X86::ST_Fp32m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSmr;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDmr;
  } else if (RC == &X86::VR128RegClass) {
    // If stack is realigned we can use aligned stores.
    Opc = isStackAligned ? X86::MOVAPSmr : X86::MOVUPSmr;
  } else if (RC == &X86::VR64RegClass) {
    Opc = X86::MMX_MOVQ64mr;
  } else {
    llvm_unreachable("Unknown regclass");
  }

  return Opc;
}

void X86InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned SrcReg, bool isKill, int FrameIdx,
                                       const TargetRegisterClass *RC) const {
  const MachineFunction &MF = *MBB.getParent();
  bool isAligned = (RI.getStackAlignment() >= 16) ||
    RI.needsStackRealignment(MF);
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, TM);
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  addFrameReference(BuildMI(MBB, MI, DL, get(Opc)), FrameIdx)
    .addReg(SrcReg, getKillRegState(isKill));
}

void X86InstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                  bool isKill,
                                  SmallVectorImpl<MachineOperand> &Addr,
                                  const TargetRegisterClass *RC,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const {
  bool isAligned = (RI.getStackAlignment() >= 16) ||
    RI.needsStackRealignment(MF);
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, TM);
  DebugLoc DL = DebugLoc::getUnknownLoc();
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  MIB.addReg(SrcReg, getKillRegState(isKill));
  NewMIs.push_back(MIB);
}

static unsigned getLoadRegOpcode(unsigned DestReg,
                                 const TargetRegisterClass *RC,
                                 bool isStackAligned,
                                 const TargetMachine &TM) {
  unsigned Opc = 0;
  if (RC == &X86::GR64RegClass || RC == &X86::GR64_NOSPRegClass) {
    Opc = X86::MOV64rm;
  } else if (RC == &X86::GR32RegClass || RC == &X86::GR32_NOSPRegClass) {
    Opc = X86::MOV32rm;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16rm;
  } else if (RC == &X86::GR8RegClass) {
    // Copying to or from a physical H register on x86-64 requires a NOREX
    // move.  Otherwise use a normal move.
    if (isHReg(DestReg) &&
        TM.getSubtarget<X86Subtarget>().is64Bit())
      Opc = X86::MOV8rm_NOREX;
    else
      Opc = X86::MOV8rm;
  } else if (RC == &X86::GR64_ABCDRegClass) {
    Opc = X86::MOV64rm;
  } else if (RC == &X86::GR32_ABCDRegClass) {
    Opc = X86::MOV32rm;
  } else if (RC == &X86::GR16_ABCDRegClass) {
    Opc = X86::MOV16rm;
  } else if (RC == &X86::GR8_ABCD_LRegClass) {
    Opc = X86::MOV8rm;
  } else if (RC == &X86::GR8_ABCD_HRegClass) {
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      Opc = X86::MOV8rm_NOREX;
    else
      Opc = X86::MOV8rm;
  } else if (RC == &X86::GR64_NOREXRegClass ||
             RC == &X86::GR64_NOREX_NOSPRegClass) {
    Opc = X86::MOV64rm;
  } else if (RC == &X86::GR32_NOREXRegClass) {
    Opc = X86::MOV32rm;
  } else if (RC == &X86::GR16_NOREXRegClass) {
    Opc = X86::MOV16rm;
  } else if (RC == &X86::GR8_NOREXRegClass) {
    Opc = X86::MOV8rm;
  } else if (RC == &X86::RFP80RegClass) {
    Opc = X86::LD_Fp80m;
  } else if (RC == &X86::RFP64RegClass) {
    Opc = X86::LD_Fp64m;
  } else if (RC == &X86::RFP32RegClass) {
    Opc = X86::LD_Fp32m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSrm;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDrm;
  } else if (RC == &X86::VR128RegClass) {
    // If stack is realigned we can use aligned loads.
    Opc = isStackAligned ? X86::MOVAPSrm : X86::MOVUPSrm;
  } else if (RC == &X86::VR64RegClass) {
    Opc = X86::MMX_MOVQ64rm;
  } else {
    llvm_unreachable("Unknown regclass");
  }

  return Opc;
}

void X86InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC) const{
  const MachineFunction &MF = *MBB.getParent();
  bool isAligned = (RI.getStackAlignment() >= 16) ||
    RI.needsStackRealignment(MF);
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, TM);
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  addFrameReference(BuildMI(MBB, MI, DL, get(Opc), DestReg), FrameIdx);
}

void X86InstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  bool isAligned = (RI.getStackAlignment() >= 16) ||
    RI.needsStackRealignment(MF);
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, TM);
  DebugLoc DL = DebugLoc::getUnknownLoc();
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  NewMIs.push_back(MIB);
}

bool X86InstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();
  bool isWin64 = TM.getSubtarget<X86Subtarget>().isTargetWin64();
  unsigned SlotSize = is64Bit ? 8 : 4;

  MachineFunction &MF = *MBB.getParent();
  unsigned FPReg = RI.getFrameRegister(MF);
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  unsigned CalleeFrameSize = 0;
  
  unsigned Opc = is64Bit ? X86::PUSH64r : X86::PUSH32r;
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    const TargetRegisterClass *RegClass = CSI[i-1].getRegClass();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    if (Reg == FPReg)
      // X86RegisterInfo::emitPrologue will handle spilling of frame register.
      continue;
    if (RegClass != &X86::VR128RegClass && !isWin64) {
      CalleeFrameSize += SlotSize;
      BuildMI(MBB, MI, DL, get(Opc)).addReg(Reg, RegState::Kill);
    } else {
      storeRegToStackSlot(MBB, MI, Reg, true, CSI[i-1].getFrameIdx(), RegClass);
    }
  }

  X86FI->setCalleeSavedFrameSize(CalleeFrameSize);
  return true;
}

bool X86InstrInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  unsigned FPReg = RI.getFrameRegister(MF);
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();
  bool isWin64 = TM.getSubtarget<X86Subtarget>().isTargetWin64();
  unsigned Opc = is64Bit ? X86::POP64r : X86::POP32r;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (Reg == FPReg)
      // X86RegisterInfo::emitEpilogue will handle restoring of frame register.
      continue;
    const TargetRegisterClass *RegClass = CSI[i].getRegClass();
    if (RegClass != &X86::VR128RegClass && !isWin64) {
      BuildMI(MBB, MI, DL, get(Opc), Reg);
    } else {
      loadRegFromStackSlot(MBB, MI, Reg, CSI[i].getFrameIdx(), RegClass);
    }
  }
  return true;
}

static MachineInstr *FuseTwoAddrInst(MachineFunction &MF, unsigned Opcode,
                                     const SmallVectorImpl<MachineOperand> &MOs,
                                     MachineInstr *MI,
                                     const TargetInstrInfo &TII) {
  // Create the base instruction with the memory operand as the first part.
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(NewMI);
  unsigned NumAddrOps = MOs.size();
  for (unsigned i = 0; i != NumAddrOps; ++i)
    MIB.addOperand(MOs[i]);
  if (NumAddrOps < 4)  // FrameIndex only
    addOffset(MIB, 0);
  
  // Loop over the rest of the ri operands, converting them over.
  unsigned NumOps = MI->getDesc().getNumOperands()-2;
  for (unsigned i = 0; i != NumOps; ++i) {
    MachineOperand &MO = MI->getOperand(i+2);
    MIB.addOperand(MO);
  }
  for (unsigned i = NumOps+2, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    MIB.addOperand(MO);
  }
  return MIB;
}

static MachineInstr *FuseInst(MachineFunction &MF,
                              unsigned Opcode, unsigned OpNo,
                              const SmallVectorImpl<MachineOperand> &MOs,
                              MachineInstr *MI, const TargetInstrInfo &TII) {
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(NewMI);
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (i == OpNo) {
      assert(MO.isReg() && "Expected to fold into reg operand!");
      unsigned NumAddrOps = MOs.size();
      for (unsigned i = 0; i != NumAddrOps; ++i)
        MIB.addOperand(MOs[i]);
      if (NumAddrOps < 4)  // FrameIndex only
        addOffset(MIB, 0);
    } else {
      MIB.addOperand(MO);
    }
  }
  return MIB;
}

static MachineInstr *MakeM0Inst(const TargetInstrInfo &TII, unsigned Opcode,
                                const SmallVectorImpl<MachineOperand> &MOs,
                                MachineInstr *MI) {
  MachineFunction &MF = *MI->getParent()->getParent();
  MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), TII.get(Opcode));

  unsigned NumAddrOps = MOs.size();
  for (unsigned i = 0; i != NumAddrOps; ++i)
    MIB.addOperand(MOs[i]);
  if (NumAddrOps < 4)  // FrameIndex only
    addOffset(MIB, 0);
  return MIB.addImm(0);
}

MachineInstr*
X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                    MachineInstr *MI, unsigned i,
                                    const SmallVectorImpl<MachineOperand> &MOs,
                                    unsigned Align) const {
  const DenseMap<unsigned*, std::pair<unsigned,unsigned> > *OpcodeTablePtr=NULL;
  bool isTwoAddrFold = false;
  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, TOI::TIED_TO) != -1;

  MachineInstr *NewMI = NULL;
  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (isTwoAddr && NumOps >= 2 && i < 2 &&
      MI->getOperand(0).isReg() &&
      MI->getOperand(1).isReg() &&
      MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) { 
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
    isTwoAddrFold = true;
  } else if (i == 0) { // If operand 0
    if (MI->getOpcode() == X86::MOV16r0)
      NewMI = MakeM0Inst(*this, X86::MOV16mi, MOs, MI);
    else if (MI->getOpcode() == X86::MOV32r0)
      NewMI = MakeM0Inst(*this, X86::MOV32mi, MOs, MI);
    else if (MI->getOpcode() == X86::MOV8r0)
      NewMI = MakeM0Inst(*this, X86::MOV8mi, MOs, MI);
    if (NewMI)
      return NewMI;
    
    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (i == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (i == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  }
  
  // If table selected...
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
      OpcodeTablePtr->find((unsigned*)MI->getOpcode());
    if (I != OpcodeTablePtr->end()) {
      unsigned MinAlign = I->second.second;
      if (Align < MinAlign)
        return NULL;
      if (isTwoAddrFold)
        NewMI = FuseTwoAddrInst(MF, I->second.first, MOs, MI, *this);
      else
        NewMI = FuseInst(MF, I->second.first, i, MOs, MI, *this);
      return NewMI;
    }
  }
  
  // No fusion 
  if (PrintFailedFusing)
    cerr << "We failed to fuse operand " << i << " in " << *MI;
  return NULL;
}


MachineInstr* X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                                  int FrameIndex) const {
  // Check switch flag 
  if (NoFusing) return NULL;

  const MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned Alignment = MFI->getObjectAlignment(FrameIndex);
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri32; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  SmallVector<MachineOperand,4> MOs;
  MOs.push_back(MachineOperand::CreateFI(FrameIndex));
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, Alignment);
}

MachineInstr* X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                                  MachineInstr *LoadMI) const {
  // Check switch flag 
  if (NoFusing) return NULL;

  // Determine the alignment of the load.
  unsigned Alignment = 0;
  if (LoadMI->hasOneMemOperand())
    Alignment = LoadMI->memoperands_begin()->getAlignment();
  else if (LoadMI->getOpcode() == X86::V_SET0 ||
           LoadMI->getOpcode() == X86::V_SETALLONES)
    Alignment = 16;
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri32; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  SmallVector<MachineOperand,X86AddrNumOperands> MOs;
  if (LoadMI->getOpcode() == X86::V_SET0 ||
      LoadMI->getOpcode() == X86::V_SETALLONES) {
    // Folding a V_SET0 or V_SETALLONES as a load, to ease register pressure.
    // Create a constant-pool entry and operands to load from it.

    // x86-32 PIC requires a PIC base register for constant pools.
    unsigned PICBase = 0;
    if (TM.getRelocationModel() == Reloc::PIC_) {
      if (TM.getSubtarget<X86Subtarget>().is64Bit())
        PICBase = X86::RIP;
      else
        // FIXME: PICBase = TM.getInstrInfo()->getGlobalBaseReg(&MF);
        // This doesn't work for several reasons.
        // 1. GlobalBaseReg may have been spilled.
        // 2. It may not be live at MI.
        return false;
    }

    // Create a v4i32 constant-pool entry.
    MachineConstantPool &MCP = *MF.getConstantPool();
    const VectorType *Ty =
          VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()), 4);
    Constant *C = LoadMI->getOpcode() == X86::V_SET0 ?
                    Constant::getNullValue(Ty) :
                    Constant::getAllOnesValue(Ty);
    unsigned CPI = MCP.getConstantPoolIndex(C, 16);

    // Create operands to load from the constant pool entry.
    MOs.push_back(MachineOperand::CreateReg(PICBase, false));
    MOs.push_back(MachineOperand::CreateImm(1));
    MOs.push_back(MachineOperand::CreateReg(0, false));
    MOs.push_back(MachineOperand::CreateCPI(CPI, 0));
    MOs.push_back(MachineOperand::CreateReg(0, false));
  } else {
    // Folding a normal load. Just copy the load's address operands.
    unsigned NumOps = LoadMI->getDesc().getNumOperands();
    for (unsigned i = NumOps - X86AddrNumOperands; i != NumOps; ++i)
      MOs.push_back(LoadMI->getOperand(i));
  }
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, Alignment);
}


bool X86InstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                  const SmallVectorImpl<unsigned> &Ops) const {
  // Check switch flag 
  if (NoFusing) return 0;

  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    switch (MI->getOpcode()) {
    default: return false;
    case X86::TEST8rr: 
    case X86::TEST16rr:
    case X86::TEST32rr:
    case X86::TEST64rr:
      return true;
    }
  }

  if (Ops.size() != 1)
    return false;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, TOI::TIED_TO) != -1;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  const DenseMap<unsigned*, std::pair<unsigned,unsigned> > *OpcodeTablePtr=NULL;
  if (isTwoAddr && NumOps >= 2 && OpNum < 2) { 
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
  } else if (OpNum == 0) { // If operand 0
    switch (Opc) {
    case X86::MOV8r0:
    case X86::MOV16r0:
    case X86::MOV32r0:
      return true;
    default: break;
    }
    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (OpNum == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (OpNum == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  }
  
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
      OpcodeTablePtr->find((unsigned*)Opc);
    if (I != OpcodeTablePtr->end())
      return true;
  }
  return false;
}

bool X86InstrInfo::unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
    MemOp2RegOpTable.find((unsigned*)MI->getOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  DebugLoc dl = MI->getDebugLoc();
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & 0xf;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  if (UnfoldLoad && !FoldedLoad)
    return false;
  UnfoldLoad &= FoldedLoad;
  if (UnfoldStore && !FoldedStore)
    return false;
  UnfoldStore &= FoldedStore;

  const TargetInstrDesc &TID = get(Opc);
  const TargetOperandInfo &TOI = TID.OpInfo[Index];
  const TargetRegisterClass *RC = TOI.getRegClass(&RI);
  SmallVector<MachineOperand, X86AddrNumOperands> AddrOps;
  SmallVector<MachineOperand,2> BeforeOps;
  SmallVector<MachineOperand,2> AfterOps;
  SmallVector<MachineOperand,4> ImpOps;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &Op = MI->getOperand(i);
    if (i >= Index && i < Index + X86AddrNumOperands)
      AddrOps.push_back(Op);
    else if (Op.isReg() && Op.isImplicit())
      ImpOps.push_back(Op);
    else if (i < Index)
      BeforeOps.push_back(Op);
    else if (i > Index)
      AfterOps.push_back(Op);
  }

  // Emit the load instruction.
  if (UnfoldLoad) {
    loadRegFromAddr(MF, Reg, AddrOps, RC, NewMIs);
    if (UnfoldStore) {
      // Address operands cannot be marked isKill.
      for (unsigned i = 1; i != 1 + X86AddrNumOperands; ++i) {
        MachineOperand &MO = NewMIs[0]->getOperand(i);
        if (MO.isReg())
          MO.setIsKill(false);
      }
    }
  }

  // Emit the data processing instruction.
  MachineInstr *DataMI = MF.CreateMachineInstr(TID, MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(DataMI);
  
  if (FoldedStore)
    MIB.addReg(Reg, RegState::Define);
  for (unsigned i = 0, e = BeforeOps.size(); i != e; ++i)
    MIB.addOperand(BeforeOps[i]);
  if (FoldedLoad)
    MIB.addReg(Reg);
  for (unsigned i = 0, e = AfterOps.size(); i != e; ++i)
    MIB.addOperand(AfterOps[i]);
  for (unsigned i = 0, e = ImpOps.size(); i != e; ++i) {
    MachineOperand &MO = ImpOps[i];
    MIB.addReg(MO.getReg(),
               getDefRegState(MO.isDef()) |
               RegState::Implicit |
               getKillRegState(MO.isKill()) |
               getDeadRegState(MO.isDead()) |
               getUndefRegState(MO.isUndef()));
  }
  // Change CMP32ri r, 0 back to TEST32rr r, r, etc.
  unsigned NewOpc = 0;
  switch (DataMI->getOpcode()) {
  default: break;
  case X86::CMP64ri32:
  case X86::CMP32ri:
  case X86::CMP16ri:
  case X86::CMP8ri: {
    MachineOperand &MO0 = DataMI->getOperand(0);
    MachineOperand &MO1 = DataMI->getOperand(1);
    if (MO1.getImm() == 0) {
      switch (DataMI->getOpcode()) {
      default: break;
      case X86::CMP64ri32: NewOpc = X86::TEST64rr; break;
      case X86::CMP32ri:   NewOpc = X86::TEST32rr; break;
      case X86::CMP16ri:   NewOpc = X86::TEST16rr; break;
      case X86::CMP8ri:    NewOpc = X86::TEST8rr; break;
      }
      DataMI->setDesc(get(NewOpc));
      MO1.ChangeToRegister(MO0.getReg(), false);
    }
  }
  }
  NewMIs.push_back(DataMI);

  // Emit the store instruction.
  if (UnfoldStore) {
    const TargetRegisterClass *DstRC = TID.OpInfo[0].getRegClass(&RI);
    storeRegToAddr(MF, Reg, true, AddrOps, DstRC, NewMIs);
  }

  return true;
}

bool
X86InstrInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                  SmallVectorImpl<SDNode*> &NewNodes) const {
  if (!N->isMachineOpcode())
    return false;

  DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
    MemOp2RegOpTable.find((unsigned*)N->getMachineOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & 0xf;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  const TargetInstrDesc &TID = get(Opc);
  const TargetRegisterClass *RC = TID.OpInfo[Index].getRegClass(&RI);
  unsigned NumDefs = TID.NumDefs;
  std::vector<SDValue> AddrOps;
  std::vector<SDValue> BeforeOps;
  std::vector<SDValue> AfterOps;
  DebugLoc dl = N->getDebugLoc();
  unsigned NumOps = N->getNumOperands();
  for (unsigned i = 0; i != NumOps-1; ++i) {
    SDValue Op = N->getOperand(i);
    if (i >= Index-NumDefs && i < Index-NumDefs + X86AddrNumOperands)
      AddrOps.push_back(Op);
    else if (i < Index-NumDefs)
      BeforeOps.push_back(Op);
    else if (i > Index-NumDefs)
      AfterOps.push_back(Op);
  }
  SDValue Chain = N->getOperand(NumOps-1);
  AddrOps.push_back(Chain);

  // Emit the load instruction.
  SDNode *Load = 0;
  const MachineFunction &MF = DAG.getMachineFunction();
  if (FoldedLoad) {
    EVT VT = *RC->vt_begin();
    bool isAligned = (RI.getStackAlignment() >= 16) ||
      RI.needsStackRealignment(MF);
    Load = DAG.getTargetNode(getLoadRegOpcode(0, RC, isAligned, TM), dl,
                             VT, MVT::Other, &AddrOps[0], AddrOps.size());
    NewNodes.push_back(Load);
  }

  // Emit the data processing instruction.
  std::vector<EVT> VTs;
  const TargetRegisterClass *DstRC = 0;
  if (TID.getNumDefs() > 0) {
    DstRC = TID.OpInfo[0].getRegClass(&RI);
    VTs.push_back(*DstRC->vt_begin());
  }
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    EVT VT = N->getValueType(i);
    if (VT != MVT::Other && i >= (unsigned)TID.getNumDefs())
      VTs.push_back(VT);
  }
  if (Load)
    BeforeOps.push_back(SDValue(Load, 0));
  std::copy(AfterOps.begin(), AfterOps.end(), std::back_inserter(BeforeOps));
  SDNode *NewNode= DAG.getTargetNode(Opc, dl, VTs, &BeforeOps[0],
                                     BeforeOps.size());
  NewNodes.push_back(NewNode);

  // Emit the store instruction.
  if (FoldedStore) {
    AddrOps.pop_back();
    AddrOps.push_back(SDValue(NewNode, 0));
    AddrOps.push_back(Chain);
    bool isAligned = (RI.getStackAlignment() >= 16) ||
      RI.needsStackRealignment(MF);
    SDNode *Store = DAG.getTargetNode(getStoreRegOpcode(0, DstRC,
                                                        isAligned, TM),
                                      dl, MVT::Other,
                                      &AddrOps[0], AddrOps.size());
    NewNodes.push_back(Store);
  }

  return true;
}

unsigned X86InstrInfo::getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore) const {
  DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
    MemOp2RegOpTable.find((unsigned*)Opc);
  if (I == MemOp2RegOpTable.end())
    return 0;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  if (UnfoldLoad && !FoldedLoad)
    return 0;
  if (UnfoldStore && !FoldedStore)
    return 0;
  return I->second.first;
}

bool X86InstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case X86::TCRETURNri:
  case X86::TCRETURNdi:
  case X86::RET:     // Return.
  case X86::RETI:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm:
  case X86::JMP:     // Uncond branch.
  case X86::JMP32r:  // Indirect branch.
  case X86::JMP64r:  // Indirect branch (64-bit).
  case X86::JMP32m:  // Indirect branch through mem.
  case X86::JMP64m:  // Indirect branch through mem (64-bit).
    return true;
  default: return false;
  }
}

bool X86InstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid X86 branch condition!");
  X86::CondCode CC = static_cast<X86::CondCode>(Cond[0].getImm());
  if (CC == X86::COND_NE_OR_P || CC == X86::COND_NP_OR_E)
    return true;
  Cond[0].setImm(GetOppositeBranchCondition(CC));
  return false;
}

bool X86InstrInfo::
isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
  // FIXME: Return false for x87 stack register classes for now. We can't
  // allow any loads of these registers before FpGet_ST0_80.
  return !(RC == &X86::CCRRegClass || RC == &X86::RFP32RegClass ||
           RC == &X86::RFP64RegClass || RC == &X86::RFP80RegClass);
}

unsigned X86InstrInfo::sizeOfImm(const TargetInstrDesc *Desc) {
  switch (Desc->TSFlags & X86II::ImmMask) {
  case X86II::Imm8:   return 1;
  case X86II::Imm16:  return 2;
  case X86II::Imm32:  return 4;
  case X86II::Imm64:  return 8;
  default: llvm_unreachable("Immediate size not set!");
    return 0;
  }
}

/// isX86_64ExtendedReg - Is the MachineOperand a x86-64 extended register?
/// e.g. r8, xmm8, etc.
bool X86InstrInfo::isX86_64ExtendedReg(const MachineOperand &MO) {
  if (!MO.isReg()) return false;
  switch (MO.getReg()) {
  default: break;
  case X86::R8:    case X86::R9:    case X86::R10:   case X86::R11:
  case X86::R12:   case X86::R13:   case X86::R14:   case X86::R15:
  case X86::R8D:   case X86::R9D:   case X86::R10D:  case X86::R11D:
  case X86::R12D:  case X86::R13D:  case X86::R14D:  case X86::R15D:
  case X86::R8W:   case X86::R9W:   case X86::R10W:  case X86::R11W:
  case X86::R12W:  case X86::R13W:  case X86::R14W:  case X86::R15W:
  case X86::R8B:   case X86::R9B:   case X86::R10B:  case X86::R11B:
  case X86::R12B:  case X86::R13B:  case X86::R14B:  case X86::R15B:
  case X86::XMM8:  case X86::XMM9:  case X86::XMM10: case X86::XMM11:
  case X86::XMM12: case X86::XMM13: case X86::XMM14: case X86::XMM15:
    return true;
  }
  return false;
}


/// determineREX - Determine if the MachineInstr has to be encoded with a X86-64
/// REX prefix which specifies 1) 64-bit instructions, 2) non-default operand
/// size, and 3) use of X86-64 extended registers.
unsigned X86InstrInfo::determineREX(const MachineInstr &MI) {
  unsigned REX = 0;
  const TargetInstrDesc &Desc = MI.getDesc();

  // Pseudo instructions do not need REX prefix byte.
  if ((Desc.TSFlags & X86II::FormMask) == X86II::Pseudo)
    return 0;
  if (Desc.TSFlags & X86II::REX_W)
    REX |= 1 << 3;

  unsigned NumOps = Desc.getNumOperands();
  if (NumOps) {
    bool isTwoAddr = NumOps > 1 &&
      Desc.getOperandConstraint(1, TOI::TIED_TO) != -1;

    // If it accesses SPL, BPL, SIL, or DIL, then it requires a 0x40 REX prefix.
    unsigned i = isTwoAddr ? 1 : 0;
    for (unsigned e = NumOps; i != e; ++i) {
      const MachineOperand& MO = MI.getOperand(i);
      if (MO.isReg()) {
        unsigned Reg = MO.getReg();
        if (isX86_64NonExtLowByteReg(Reg))
          REX |= 0x40;
      }
    }

    switch (Desc.TSFlags & X86II::FormMask) {
    case X86II::MRMInitReg:
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= (1 << 0) | (1 << 2);
      break;
    case X86II::MRMSrcReg: {
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= 1 << 2;
      i = isTwoAddr ? 2 : 1;
      for (unsigned e = NumOps; i != e; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (isX86_64ExtendedReg(MO))
          REX |= 1 << 0;
      }
      break;
    }
    case X86II::MRMSrcMem: {
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= 1 << 2;
      unsigned Bit = 0;
      i = isTwoAddr ? 2 : 1;
      for (; i != NumOps; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (MO.isReg()) {
          if (isX86_64ExtendedReg(MO))
            REX |= 1 << Bit;
          Bit++;
        }
      }
      break;
    }
    case X86II::MRM0m: case X86II::MRM1m:
    case X86II::MRM2m: case X86II::MRM3m:
    case X86II::MRM4m: case X86II::MRM5m:
    case X86II::MRM6m: case X86II::MRM7m:
    case X86II::MRMDestMem: {
      unsigned e = (isTwoAddr ? X86AddrNumOperands+1 : X86AddrNumOperands);
      i = isTwoAddr ? 1 : 0;
      if (NumOps > e && isX86_64ExtendedReg(MI.getOperand(e)))
        REX |= 1 << 2;
      unsigned Bit = 0;
      for (; i != e; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (MO.isReg()) {
          if (isX86_64ExtendedReg(MO))
            REX |= 1 << Bit;
          Bit++;
        }
      }
      break;
    }
    default: {
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= 1 << 0;
      i = isTwoAddr ? 2 : 1;
      for (unsigned e = NumOps; i != e; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (isX86_64ExtendedReg(MO))
          REX |= 1 << 2;
      }
      break;
    }
    }
  }
  return REX;
}

/// sizePCRelativeBlockAddress - This method returns the size of a PC
/// relative block address instruction
///
static unsigned sizePCRelativeBlockAddress() {
  return 4;
}

/// sizeGlobalAddress - Give the size of the emission of this global address
///
static unsigned sizeGlobalAddress(bool dword) {
  return dword ? 8 : 4;
}

/// sizeConstPoolAddress - Give the size of the emission of this constant
/// pool address
///
static unsigned sizeConstPoolAddress(bool dword) {
  return dword ? 8 : 4;
}

/// sizeExternalSymbolAddress - Give the size of the emission of this external
/// symbol
///
static unsigned sizeExternalSymbolAddress(bool dword) {
  return dword ? 8 : 4;
}

/// sizeJumpTableAddress - Give the size of the emission of this jump
/// table address
///
static unsigned sizeJumpTableAddress(bool dword) {
  return dword ? 8 : 4;
}

static unsigned sizeConstant(unsigned Size) {
  return Size;
}

static unsigned sizeRegModRMByte(){
  return 1;
}

static unsigned sizeSIBByte(){
  return 1;
}

static unsigned getDisplacementFieldSize(const MachineOperand *RelocOp) {
  unsigned FinalSize = 0;
  // If this is a simple integer displacement that doesn't require a relocation.
  if (!RelocOp) {
    FinalSize += sizeConstant(4);
    return FinalSize;
  }
  
  // Otherwise, this is something that requires a relocation.
  if (RelocOp->isGlobal()) {
    FinalSize += sizeGlobalAddress(false);
  } else if (RelocOp->isCPI()) {
    FinalSize += sizeConstPoolAddress(false);
  } else if (RelocOp->isJTI()) {
    FinalSize += sizeJumpTableAddress(false);
  } else {
    llvm_unreachable("Unknown value to relocate!");
  }
  return FinalSize;
}

static unsigned getMemModRMByteSize(const MachineInstr &MI, unsigned Op,
                                    bool IsPIC, bool Is64BitMode) {
  const MachineOperand &Op3 = MI.getOperand(Op+3);
  int DispVal = 0;
  const MachineOperand *DispForReloc = 0;
  unsigned FinalSize = 0;
  
  // Figure out what sort of displacement we have to handle here.
  if (Op3.isGlobal()) {
    DispForReloc = &Op3;
  } else if (Op3.isCPI()) {
    if (Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal = 1;
    }
  } else if (Op3.isJTI()) {
    if (Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal = 1; 
    }
  } else {
    DispVal = 1;
  }

  const MachineOperand &Base     = MI.getOperand(Op);
  const MachineOperand &IndexReg = MI.getOperand(Op+2);

  unsigned BaseReg = Base.getReg();

  // Is a SIB byte needed?
  if ((!Is64BitMode || DispForReloc || BaseReg != 0) &&
      IndexReg.getReg() == 0 &&
      (BaseReg == 0 || X86RegisterInfo::getX86RegNum(BaseReg) != N86::ESP)) {      
    if (BaseReg == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      ++FinalSize; 
      FinalSize += getDisplacementFieldSize(DispForReloc);
    } else {
      unsigned BaseRegNo = X86RegisterInfo::getX86RegNum(BaseReg);
      if (!DispForReloc && DispVal == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        ++FinalSize;
      // Be pessimistic and assume it's a disp32, not a disp8
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        ++FinalSize;
        FinalSize += getDisplacementFieldSize(DispForReloc);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP &&
           IndexReg.getReg() != X86::RSP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    if (BaseReg == 0 || DispForReloc) {
      // Emit the normal disp32 encoding.
      ++FinalSize;
      ForceDisp32 = true;
    } else {
      ++FinalSize;
    }

    FinalSize += sizeSIBByte();

    // Do we need to output a displacement?
    if (DispVal != 0 || ForceDisp32) {
      FinalSize += getDisplacementFieldSize(DispForReloc);
    }
  }
  return FinalSize;
}


static unsigned GetInstSizeWithDesc(const MachineInstr &MI,
                                    const TargetInstrDesc *Desc,
                                    bool IsPIC, bool Is64BitMode) {
  
  unsigned Opcode = Desc->Opcode;
  unsigned FinalSize = 0;

  // Emit the lock opcode prefix as needed.
  if (Desc->TSFlags & X86II::LOCK) ++FinalSize;

  // Emit segment override opcode prefix as needed.
  switch (Desc->TSFlags & X86II::SegOvrMask) {
  case X86II::FS:
  case X86II::GS:
   ++FinalSize;
   break;
  default: llvm_unreachable("Invalid segment!");
  case 0: break;  // No segment override!
  }

  // Emit the repeat opcode prefix as needed.
  if ((Desc->TSFlags & X86II::Op0Mask) == X86II::REP) ++FinalSize;

  // Emit the operand size opcode prefix as needed.
  if (Desc->TSFlags & X86II::OpSize) ++FinalSize;

  // Emit the address size opcode prefix as needed.
  if (Desc->TSFlags & X86II::AdSize) ++FinalSize;

  bool Need0FPrefix = false;
  switch (Desc->TSFlags & X86II::Op0Mask) {
  case X86II::TB:  // Two-byte opcode prefix
  case X86II::T8:  // 0F 38
  case X86II::TA:  // 0F 3A
    Need0FPrefix = true;
    break;
  case X86II::TF: // F2 0F 38
    ++FinalSize;
    Need0FPrefix = true;
    break;
  case X86II::REP: break; // already handled.
  case X86II::XS:   // F3 0F
    ++FinalSize;
    Need0FPrefix = true;
    break;
  case X86II::XD:   // F2 0F
    ++FinalSize;
    Need0FPrefix = true;
    break;
  case X86II::D8: case X86II::D9: case X86II::DA: case X86II::DB:
  case X86II::DC: case X86II::DD: case X86II::DE: case X86II::DF:
    ++FinalSize;
    break; // Two-byte opcode prefix
  default: llvm_unreachable("Invalid prefix!");
  case 0: break;  // No prefix!
  }

  if (Is64BitMode) {
    // REX prefix
    unsigned REX = X86InstrInfo::determineREX(MI);
    if (REX)
      ++FinalSize;
  }

  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    ++FinalSize;

  switch (Desc->TSFlags & X86II::Op0Mask) {
  case X86II::T8:  // 0F 38
    ++FinalSize;
    break;
  case X86II::TA:  // 0F 3A
    ++FinalSize;
    break;
  case X86II::TF: // F2 0F 38
    ++FinalSize;
    break;
  }

  // If this is a two-address instruction, skip one of the register operands.
  unsigned NumOps = Desc->getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc->getOperandConstraint(1, TOI::TIED_TO) != -1)
    CurOp++;
  else if (NumOps > 2 && Desc->getOperandConstraint(NumOps-1, TOI::TIED_TO)== 0)
    // Skip the last source operand that is tied_to the dest reg. e.g. LXADD32
    --NumOps;

  switch (Desc->TSFlags & X86II::FormMask) {
  default: llvm_unreachable("Unknown FormMask value in X86 MachineCodeEmitter!");
  case X86II::Pseudo:
    // Remember the current PC offset, this is the PIC relocation
    // base address.
    switch (Opcode) {
    default: 
      break;
    case TargetInstrInfo::INLINEASM: {
      const MachineFunction *MF = MI.getParent()->getParent();
      const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();
      FinalSize += TII.getInlineAsmLength(MI.getOperand(0).getSymbolName(),
                                          *MF->getTarget().getTargetAsmInfo());
      break;
    }
    case TargetInstrInfo::DBG_LABEL:
    case TargetInstrInfo::EH_LABEL:
      break;
    case TargetInstrInfo::IMPLICIT_DEF:
    case TargetInstrInfo::DECLARE:
    case X86::DWARF_LOC:
    case X86::FP_REG_KILL:
      break;
    case X86::MOVPC32r: {
      // This emits the "call" portion of this pseudo instruction.
      ++FinalSize;
      FinalSize += sizeConstant(X86InstrInfo::sizeOfImm(Desc));
      break;
    }
    }
    CurOp = NumOps;
    break;
  case X86II::RawFrm:
    ++FinalSize;

    if (CurOp != NumOps) {
      const MachineOperand &MO = MI.getOperand(CurOp++);
      if (MO.isMBB()) {
        FinalSize += sizePCRelativeBlockAddress();
      } else if (MO.isGlobal()) {
        FinalSize += sizeGlobalAddress(false);
      } else if (MO.isSymbol()) {
        FinalSize += sizeExternalSymbolAddress(false);
      } else if (MO.isImm()) {
        FinalSize += sizeConstant(X86InstrInfo::sizeOfImm(Desc));
      } else {
        llvm_unreachable("Unknown RawFrm operand!");
      }
    }
    break;

  case X86II::AddRegFrm:
    ++FinalSize;
    ++CurOp;
    
    if (CurOp != NumOps) {
      const MachineOperand &MO1 = MI.getOperand(CurOp++);
      unsigned Size = X86InstrInfo::sizeOfImm(Desc);
      if (MO1.isImm())
        FinalSize += sizeConstant(Size);
      else {
        bool dword = false;
        if (Opcode == X86::MOV64ri)
          dword = true; 
        if (MO1.isGlobal()) {
          FinalSize += sizeGlobalAddress(dword);
        } else if (MO1.isSymbol())
          FinalSize += sizeExternalSymbolAddress(dword);
        else if (MO1.isCPI())
          FinalSize += sizeConstPoolAddress(dword);
        else if (MO1.isJTI())
          FinalSize += sizeJumpTableAddress(dword);
      }
    }
    break;

  case X86II::MRMDestReg: {
    ++FinalSize; 
    FinalSize += sizeRegModRMByte();
    CurOp += 2;
    if (CurOp != NumOps) {
      ++CurOp;
      FinalSize += sizeConstant(X86InstrInfo::sizeOfImm(Desc));
    }
    break;
  }
  case X86II::MRMDestMem: {
    ++FinalSize;
    FinalSize += getMemModRMByteSize(MI, CurOp, IsPIC, Is64BitMode);
    CurOp +=  X86AddrNumOperands + 1;
    if (CurOp != NumOps) {
      ++CurOp;
      FinalSize += sizeConstant(X86InstrInfo::sizeOfImm(Desc));
    }
    break;
  }

  case X86II::MRMSrcReg:
    ++FinalSize;
    FinalSize += sizeRegModRMByte();
    CurOp += 2;
    if (CurOp != NumOps) {
      ++CurOp;
      FinalSize += sizeConstant(X86InstrInfo::sizeOfImm(Desc));
    }
    break;

  case X86II::MRMSrcMem: {
    int AddrOperands;
    if (Opcode == X86::LEA64r || Opcode == X86::LEA64_32r ||
        Opcode == X86::LEA16r || Opcode == X86::LEA32r)
      AddrOperands = X86AddrNumOperands - 1; // No segment register
    else
      AddrOperands = X86AddrNumOperands;

    ++FinalSize;
    FinalSize += getMemModRMByteSize(MI, CurOp+1, IsPIC, Is64BitMode);
    CurOp += AddrOperands + 1;
    if (CurOp != NumOps) {
      ++CurOp;
      FinalSize += sizeConstant(X86InstrInfo::sizeOfImm(Desc));
    }
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r:
    ++FinalSize;
    if (Desc->getOpcode() == X86::LFENCE ||
        Desc->getOpcode() == X86::MFENCE) {
      // Special handling of lfence and mfence;
      FinalSize += sizeRegModRMByte();
    } else if (Desc->getOpcode() == X86::MONITOR ||
               Desc->getOpcode() == X86::MWAIT) {
      // Special handling of monitor and mwait.
      FinalSize += sizeRegModRMByte() + 1; // +1 for the opcode.
    } else {
      ++CurOp;
      FinalSize += sizeRegModRMByte();
    }

    if (CurOp != NumOps) {
      const MachineOperand &MO1 = MI.getOperand(CurOp++);
      unsigned Size = X86InstrInfo::sizeOfImm(Desc);
      if (MO1.isImm())
        FinalSize += sizeConstant(Size);
      else {
        bool dword = false;
        if (Opcode == X86::MOV64ri32)
          dword = true;
        if (MO1.isGlobal()) {
          FinalSize += sizeGlobalAddress(dword);
        } else if (MO1.isSymbol())
          FinalSize += sizeExternalSymbolAddress(dword);
        else if (MO1.isCPI())
          FinalSize += sizeConstPoolAddress(dword);
        else if (MO1.isJTI())
          FinalSize += sizeJumpTableAddress(dword);
      }
    }
    break;

  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m: {
    
    ++FinalSize;
    FinalSize += getMemModRMByteSize(MI, CurOp, IsPIC, Is64BitMode);
    CurOp += X86AddrNumOperands;

    if (CurOp != NumOps) {
      const MachineOperand &MO = MI.getOperand(CurOp++);
      unsigned Size = X86InstrInfo::sizeOfImm(Desc);
      if (MO.isImm())
        FinalSize += sizeConstant(Size);
      else {
        bool dword = false;
        if (Opcode == X86::MOV64mi32)
          dword = true;
        if (MO.isGlobal()) {
          FinalSize += sizeGlobalAddress(dword);
        } else if (MO.isSymbol())
          FinalSize += sizeExternalSymbolAddress(dword);
        else if (MO.isCPI())
          FinalSize += sizeConstPoolAddress(dword);
        else if (MO.isJTI())
          FinalSize += sizeJumpTableAddress(dword);
      }
    }
    break;
  }

  case X86II::MRMInitReg:
    ++FinalSize;
    // Duplicate register, used by things like MOV8r0 (aka xor reg,reg).
    FinalSize += sizeRegModRMByte();
    ++CurOp;
    break;
  }

  if (!Desc->isVariadic() && CurOp != NumOps) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Cannot determine size: " << MI;
    llvm_report_error(Msg.str());
  }
  

  return FinalSize;
}


unsigned X86InstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  const TargetInstrDesc &Desc = MI->getDesc();
  bool IsPIC = TM.getRelocationModel() == Reloc::PIC_;
  bool Is64BitMode = TM.getSubtargetImpl()->is64Bit();
  unsigned Size = GetInstSizeWithDesc(*MI, &Desc, IsPIC, Is64BitMode);
  if (Desc.getOpcode() == X86::MOVPC32r)
    Size += GetInstSizeWithDesc(*MI, &get(X86::POP32r), IsPIC, Is64BitMode);
  return Size;
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned X86InstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  assert(!TM.getSubtarget<X86Subtarget>().is64Bit() &&
         "X86-64 PIC uses RIP relative addressing");

  X86MachineFunctionInfo *X86FI = MF->getInfo<X86MachineFunctionInfo>();
  unsigned GlobalBaseReg = X86FI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MBBI != FirstMBB.end()) DL = MBBI->getDebugLoc();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  unsigned PC = RegInfo.createVirtualRegister(X86::GR32RegisterClass);
  
  const TargetInstrInfo *TII = TM.getInstrInfo();
  // Operand of MovePCtoStack is completely ignored by asm printer. It's
  // only used in JIT code emission as displacement to pc.
  BuildMI(FirstMBB, MBBI, DL, TII->get(X86::MOVPC32r), PC).addImm(0);
  
  // If we're using vanilla 'GOT' PIC style, we should use relative addressing
  // not to pc, but to _GLOBAL_OFFSET_TABLE_ external.
  if (TM.getSubtarget<X86Subtarget>().isPICStyleGOT()) {
    GlobalBaseReg = RegInfo.createVirtualRegister(X86::GR32RegisterClass);
    // Generate addl $__GLOBAL_OFFSET_TABLE_ + [.-piclabel], %some_register
    BuildMI(FirstMBB, MBBI, DL, TII->get(X86::ADD32ri), GlobalBaseReg)
      .addReg(PC).addExternalSymbol("_GLOBAL_OFFSET_TABLE_", 0,
                                    X86II::MO_GOT_ABSOLUTE_ADDRESS);
  } else {
    GlobalBaseReg = PC;
  }

  X86FI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}
