//===-- X86InstrInfo.cpp - X86 Instruction Information --------------------===//
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
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "x86-instr-info"

#define GET_INSTRINFO_CTOR_DTOR
#include "X86GenInstrInfo.inc"

static cl::opt<bool>
NoFusing("disable-spill-fusing",
         cl::desc("Disable fusing of spill code into instructions"));
static cl::opt<bool>
PrintFailedFusing("print-failed-fuse-candidates",
                  cl::desc("Print instructions that the allocator wants to"
                           " fuse, but the X86 backend currently can't"),
                  cl::Hidden);
static cl::opt<bool>
ReMatPICStubLoad("remat-pic-stub-load",
                 cl::desc("Re-materialize load from stub in PIC mode"),
                 cl::init(false), cl::Hidden);

enum {
  // Select which memory operand is being unfolded.
  // (stored in bits 0 - 3)
  TB_INDEX_0    = 0,
  TB_INDEX_1    = 1,
  TB_INDEX_2    = 2,
  TB_INDEX_3    = 3,
  TB_INDEX_4    = 4,
  TB_INDEX_MASK = 0xf,

  // Do not insert the reverse map (MemOp -> RegOp) into the table.
  // This may be needed because there is a many -> one mapping.
  TB_NO_REVERSE   = 1 << 4,

  // Do not insert the forward map (RegOp -> MemOp) into the table.
  // This is needed for Native Client, which prohibits branch
  // instructions from using a memory operand.
  TB_NO_FORWARD   = 1 << 5,

  TB_FOLDED_LOAD  = 1 << 6,
  TB_FOLDED_STORE = 1 << 7,

  // Minimum alignment required for load/store.
  // Used for RegOp->MemOp conversion.
  // (stored in bits 8 - 15)
  TB_ALIGN_SHIFT = 8,
  TB_ALIGN_NONE  =    0 << TB_ALIGN_SHIFT,
  TB_ALIGN_16    =   16 << TB_ALIGN_SHIFT,
  TB_ALIGN_32    =   32 << TB_ALIGN_SHIFT,
  TB_ALIGN_64    =   64 << TB_ALIGN_SHIFT,
  TB_ALIGN_MASK  = 0xff << TB_ALIGN_SHIFT
};

struct X86MemoryFoldTableEntry {
  uint16_t RegOp;
  uint16_t MemOp;
  uint16_t Flags;
};

// Pin the vtable to this file.
void X86InstrInfo::anchor() {}

X86InstrInfo::X86InstrInfo(X86Subtarget &STI)
    : X86GenInstrInfo((STI.isTarget64BitLP64() ? X86::ADJCALLSTACKDOWN64
                                               : X86::ADJCALLSTACKDOWN32),
                      (STI.isTarget64BitLP64() ? X86::ADJCALLSTACKUP64
                                               : X86::ADJCALLSTACKUP32),
                      X86::CATCHRET),
      Subtarget(STI), RI(STI.getTargetTriple()) {

  static const X86MemoryFoldTableEntry MemoryFoldTable2Addr[] = {
    { X86::ADC32ri,     X86::ADC32mi,    0 },
    { X86::ADC32ri8,    X86::ADC32mi8,   0 },
    { X86::ADC32rr,     X86::ADC32mr,    0 },
    { X86::ADC64ri32,   X86::ADC64mi32,  0 },
    { X86::ADC64ri8,    X86::ADC64mi8,   0 },
    { X86::ADC64rr,     X86::ADC64mr,    0 },
    { X86::ADD16ri,     X86::ADD16mi,    0 },
    { X86::ADD16ri8,    X86::ADD16mi8,   0 },
    { X86::ADD16ri_DB,  X86::ADD16mi,    TB_NO_REVERSE },
    { X86::ADD16ri8_DB, X86::ADD16mi8,   TB_NO_REVERSE },
    { X86::ADD16rr,     X86::ADD16mr,    0 },
    { X86::ADD16rr_DB,  X86::ADD16mr,    TB_NO_REVERSE },
    { X86::ADD32ri,     X86::ADD32mi,    0 },
    { X86::ADD32ri8,    X86::ADD32mi8,   0 },
    { X86::ADD32ri_DB,  X86::ADD32mi,    TB_NO_REVERSE },
    { X86::ADD32ri8_DB, X86::ADD32mi8,   TB_NO_REVERSE },
    { X86::ADD32rr,     X86::ADD32mr,    0 },
    { X86::ADD32rr_DB,  X86::ADD32mr,    TB_NO_REVERSE },
    { X86::ADD64ri32,   X86::ADD64mi32,  0 },
    { X86::ADD64ri8,    X86::ADD64mi8,   0 },
    { X86::ADD64ri32_DB,X86::ADD64mi32,  TB_NO_REVERSE },
    { X86::ADD64ri8_DB, X86::ADD64mi8,   TB_NO_REVERSE },
    { X86::ADD64rr,     X86::ADD64mr,    0 },
    { X86::ADD64rr_DB,  X86::ADD64mr,    TB_NO_REVERSE },
    { X86::ADD8ri,      X86::ADD8mi,     0 },
    { X86::ADD8rr,      X86::ADD8mr,     0 },
    { X86::AND16ri,     X86::AND16mi,    0 },
    { X86::AND16ri8,    X86::AND16mi8,   0 },
    { X86::AND16rr,     X86::AND16mr,    0 },
    { X86::AND32ri,     X86::AND32mi,    0 },
    { X86::AND32ri8,    X86::AND32mi8,   0 },
    { X86::AND32rr,     X86::AND32mr,    0 },
    { X86::AND64ri32,   X86::AND64mi32,  0 },
    { X86::AND64ri8,    X86::AND64mi8,   0 },
    { X86::AND64rr,     X86::AND64mr,    0 },
    { X86::AND8ri,      X86::AND8mi,     0 },
    { X86::AND8rr,      X86::AND8mr,     0 },
    { X86::DEC16r,      X86::DEC16m,     0 },
    { X86::DEC32r,      X86::DEC32m,     0 },
    { X86::DEC64r,      X86::DEC64m,     0 },
    { X86::DEC8r,       X86::DEC8m,      0 },
    { X86::INC16r,      X86::INC16m,     0 },
    { X86::INC32r,      X86::INC32m,     0 },
    { X86::INC64r,      X86::INC64m,     0 },
    { X86::INC8r,       X86::INC8m,      0 },
    { X86::NEG16r,      X86::NEG16m,     0 },
    { X86::NEG32r,      X86::NEG32m,     0 },
    { X86::NEG64r,      X86::NEG64m,     0 },
    { X86::NEG8r,       X86::NEG8m,      0 },
    { X86::NOT16r,      X86::NOT16m,     0 },
    { X86::NOT32r,      X86::NOT32m,     0 },
    { X86::NOT64r,      X86::NOT64m,     0 },
    { X86::NOT8r,       X86::NOT8m,      0 },
    { X86::OR16ri,      X86::OR16mi,     0 },
    { X86::OR16ri8,     X86::OR16mi8,    0 },
    { X86::OR16rr,      X86::OR16mr,     0 },
    { X86::OR32ri,      X86::OR32mi,     0 },
    { X86::OR32ri8,     X86::OR32mi8,    0 },
    { X86::OR32rr,      X86::OR32mr,     0 },
    { X86::OR64ri32,    X86::OR64mi32,   0 },
    { X86::OR64ri8,     X86::OR64mi8,    0 },
    { X86::OR64rr,      X86::OR64mr,     0 },
    { X86::OR8ri,       X86::OR8mi,      0 },
    { X86::OR8rr,       X86::OR8mr,      0 },
    { X86::ROL16r1,     X86::ROL16m1,    0 },
    { X86::ROL16rCL,    X86::ROL16mCL,   0 },
    { X86::ROL16ri,     X86::ROL16mi,    0 },
    { X86::ROL32r1,     X86::ROL32m1,    0 },
    { X86::ROL32rCL,    X86::ROL32mCL,   0 },
    { X86::ROL32ri,     X86::ROL32mi,    0 },
    { X86::ROL64r1,     X86::ROL64m1,    0 },
    { X86::ROL64rCL,    X86::ROL64mCL,   0 },
    { X86::ROL64ri,     X86::ROL64mi,    0 },
    { X86::ROL8r1,      X86::ROL8m1,     0 },
    { X86::ROL8rCL,     X86::ROL8mCL,    0 },
    { X86::ROL8ri,      X86::ROL8mi,     0 },
    { X86::ROR16r1,     X86::ROR16m1,    0 },
    { X86::ROR16rCL,    X86::ROR16mCL,   0 },
    { X86::ROR16ri,     X86::ROR16mi,    0 },
    { X86::ROR32r1,     X86::ROR32m1,    0 },
    { X86::ROR32rCL,    X86::ROR32mCL,   0 },
    { X86::ROR32ri,     X86::ROR32mi,    0 },
    { X86::ROR64r1,     X86::ROR64m1,    0 },
    { X86::ROR64rCL,    X86::ROR64mCL,   0 },
    { X86::ROR64ri,     X86::ROR64mi,    0 },
    { X86::ROR8r1,      X86::ROR8m1,     0 },
    { X86::ROR8rCL,     X86::ROR8mCL,    0 },
    { X86::ROR8ri,      X86::ROR8mi,     0 },
    { X86::SAR16r1,     X86::SAR16m1,    0 },
    { X86::SAR16rCL,    X86::SAR16mCL,   0 },
    { X86::SAR16ri,     X86::SAR16mi,    0 },
    { X86::SAR32r1,     X86::SAR32m1,    0 },
    { X86::SAR32rCL,    X86::SAR32mCL,   0 },
    { X86::SAR32ri,     X86::SAR32mi,    0 },
    { X86::SAR64r1,     X86::SAR64m1,    0 },
    { X86::SAR64rCL,    X86::SAR64mCL,   0 },
    { X86::SAR64ri,     X86::SAR64mi,    0 },
    { X86::SAR8r1,      X86::SAR8m1,     0 },
    { X86::SAR8rCL,     X86::SAR8mCL,    0 },
    { X86::SAR8ri,      X86::SAR8mi,     0 },
    { X86::SBB32ri,     X86::SBB32mi,    0 },
    { X86::SBB32ri8,    X86::SBB32mi8,   0 },
    { X86::SBB32rr,     X86::SBB32mr,    0 },
    { X86::SBB64ri32,   X86::SBB64mi32,  0 },
    { X86::SBB64ri8,    X86::SBB64mi8,   0 },
    { X86::SBB64rr,     X86::SBB64mr,    0 },
    { X86::SHL16rCL,    X86::SHL16mCL,   0 },
    { X86::SHL16ri,     X86::SHL16mi,    0 },
    { X86::SHL32rCL,    X86::SHL32mCL,   0 },
    { X86::SHL32ri,     X86::SHL32mi,    0 },
    { X86::SHL64rCL,    X86::SHL64mCL,   0 },
    { X86::SHL64ri,     X86::SHL64mi,    0 },
    { X86::SHL8rCL,     X86::SHL8mCL,    0 },
    { X86::SHL8ri,      X86::SHL8mi,     0 },
    { X86::SHLD16rrCL,  X86::SHLD16mrCL, 0 },
    { X86::SHLD16rri8,  X86::SHLD16mri8, 0 },
    { X86::SHLD32rrCL,  X86::SHLD32mrCL, 0 },
    { X86::SHLD32rri8,  X86::SHLD32mri8, 0 },
    { X86::SHLD64rrCL,  X86::SHLD64mrCL, 0 },
    { X86::SHLD64rri8,  X86::SHLD64mri8, 0 },
    { X86::SHR16r1,     X86::SHR16m1,    0 },
    { X86::SHR16rCL,    X86::SHR16mCL,   0 },
    { X86::SHR16ri,     X86::SHR16mi,    0 },
    { X86::SHR32r1,     X86::SHR32m1,    0 },
    { X86::SHR32rCL,    X86::SHR32mCL,   0 },
    { X86::SHR32ri,     X86::SHR32mi,    0 },
    { X86::SHR64r1,     X86::SHR64m1,    0 },
    { X86::SHR64rCL,    X86::SHR64mCL,   0 },
    { X86::SHR64ri,     X86::SHR64mi,    0 },
    { X86::SHR8r1,      X86::SHR8m1,     0 },
    { X86::SHR8rCL,     X86::SHR8mCL,    0 },
    { X86::SHR8ri,      X86::SHR8mi,     0 },
    { X86::SHRD16rrCL,  X86::SHRD16mrCL, 0 },
    { X86::SHRD16rri8,  X86::SHRD16mri8, 0 },
    { X86::SHRD32rrCL,  X86::SHRD32mrCL, 0 },
    { X86::SHRD32rri8,  X86::SHRD32mri8, 0 },
    { X86::SHRD64rrCL,  X86::SHRD64mrCL, 0 },
    { X86::SHRD64rri8,  X86::SHRD64mri8, 0 },
    { X86::SUB16ri,     X86::SUB16mi,    0 },
    { X86::SUB16ri8,    X86::SUB16mi8,   0 },
    { X86::SUB16rr,     X86::SUB16mr,    0 },
    { X86::SUB32ri,     X86::SUB32mi,    0 },
    { X86::SUB32ri8,    X86::SUB32mi8,   0 },
    { X86::SUB32rr,     X86::SUB32mr,    0 },
    { X86::SUB64ri32,   X86::SUB64mi32,  0 },
    { X86::SUB64ri8,    X86::SUB64mi8,   0 },
    { X86::SUB64rr,     X86::SUB64mr,    0 },
    { X86::SUB8ri,      X86::SUB8mi,     0 },
    { X86::SUB8rr,      X86::SUB8mr,     0 },
    { X86::XOR16ri,     X86::XOR16mi,    0 },
    { X86::XOR16ri8,    X86::XOR16mi8,   0 },
    { X86::XOR16rr,     X86::XOR16mr,    0 },
    { X86::XOR32ri,     X86::XOR32mi,    0 },
    { X86::XOR32ri8,    X86::XOR32mi8,   0 },
    { X86::XOR32rr,     X86::XOR32mr,    0 },
    { X86::XOR64ri32,   X86::XOR64mi32,  0 },
    { X86::XOR64ri8,    X86::XOR64mi8,   0 },
    { X86::XOR64rr,     X86::XOR64mr,    0 },
    { X86::XOR8ri,      X86::XOR8mi,     0 },
    { X86::XOR8rr,      X86::XOR8mr,     0 }
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable2Addr) {
    AddTableEntry(RegOp2MemOpTable2Addr, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 0, folded load and store, no alignment requirement.
                  Entry.Flags | TB_INDEX_0 | TB_FOLDED_LOAD | TB_FOLDED_STORE);
  }

  static const X86MemoryFoldTableEntry MemoryFoldTable0[] = {
    { X86::BT16ri8,     X86::BT16mi8,       TB_FOLDED_LOAD },
    { X86::BT32ri8,     X86::BT32mi8,       TB_FOLDED_LOAD },
    { X86::BT64ri8,     X86::BT64mi8,       TB_FOLDED_LOAD },
    { X86::CALL32r,     X86::CALL32m,       TB_FOLDED_LOAD },
    { X86::CALL64r,     X86::CALL64m,       TB_FOLDED_LOAD },
    { X86::CMP16ri,     X86::CMP16mi,       TB_FOLDED_LOAD },
    { X86::CMP16ri8,    X86::CMP16mi8,      TB_FOLDED_LOAD },
    { X86::CMP16rr,     X86::CMP16mr,       TB_FOLDED_LOAD },
    { X86::CMP32ri,     X86::CMP32mi,       TB_FOLDED_LOAD },
    { X86::CMP32ri8,    X86::CMP32mi8,      TB_FOLDED_LOAD },
    { X86::CMP32rr,     X86::CMP32mr,       TB_FOLDED_LOAD },
    { X86::CMP64ri32,   X86::CMP64mi32,     TB_FOLDED_LOAD },
    { X86::CMP64ri8,    X86::CMP64mi8,      TB_FOLDED_LOAD },
    { X86::CMP64rr,     X86::CMP64mr,       TB_FOLDED_LOAD },
    { X86::CMP8ri,      X86::CMP8mi,        TB_FOLDED_LOAD },
    { X86::CMP8rr,      X86::CMP8mr,        TB_FOLDED_LOAD },
    { X86::DIV16r,      X86::DIV16m,        TB_FOLDED_LOAD },
    { X86::DIV32r,      X86::DIV32m,        TB_FOLDED_LOAD },
    { X86::DIV64r,      X86::DIV64m,        TB_FOLDED_LOAD },
    { X86::DIV8r,       X86::DIV8m,         TB_FOLDED_LOAD },
    { X86::EXTRACTPSrr, X86::EXTRACTPSmr,   TB_FOLDED_STORE },
    { X86::IDIV16r,     X86::IDIV16m,       TB_FOLDED_LOAD },
    { X86::IDIV32r,     X86::IDIV32m,       TB_FOLDED_LOAD },
    { X86::IDIV64r,     X86::IDIV64m,       TB_FOLDED_LOAD },
    { X86::IDIV8r,      X86::IDIV8m,        TB_FOLDED_LOAD },
    { X86::IMUL16r,     X86::IMUL16m,       TB_FOLDED_LOAD },
    { X86::IMUL32r,     X86::IMUL32m,       TB_FOLDED_LOAD },
    { X86::IMUL64r,     X86::IMUL64m,       TB_FOLDED_LOAD },
    { X86::IMUL8r,      X86::IMUL8m,        TB_FOLDED_LOAD },
    { X86::JMP32r,      X86::JMP32m,        TB_FOLDED_LOAD },
    { X86::JMP64r,      X86::JMP64m,        TB_FOLDED_LOAD },
    { X86::MOV16ri,     X86::MOV16mi,       TB_FOLDED_STORE },
    { X86::MOV16rr,     X86::MOV16mr,       TB_FOLDED_STORE },
    { X86::MOV32ri,     X86::MOV32mi,       TB_FOLDED_STORE },
    { X86::MOV32rr,     X86::MOV32mr,       TB_FOLDED_STORE },
    { X86::MOV64ri32,   X86::MOV64mi32,     TB_FOLDED_STORE },
    { X86::MOV64rr,     X86::MOV64mr,       TB_FOLDED_STORE },
    { X86::MOV8ri,      X86::MOV8mi,        TB_FOLDED_STORE },
    { X86::MOV8rr,      X86::MOV8mr,        TB_FOLDED_STORE },
    { X86::MOV8rr_NOREX, X86::MOV8mr_NOREX, TB_FOLDED_STORE },
    { X86::MOVAPDrr,    X86::MOVAPDmr,      TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::MOVAPSrr,    X86::MOVAPSmr,      TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::MOVDQArr,    X86::MOVDQAmr,      TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::MOVPDI2DIrr, X86::MOVPDI2DImr,   TB_FOLDED_STORE },
    { X86::MOVPQIto64rr,X86::MOVPQI2QImr,   TB_FOLDED_STORE },
    { X86::MOVSDto64rr, X86::MOVSDto64mr,   TB_FOLDED_STORE },
    { X86::MOVSS2DIrr,  X86::MOVSS2DImr,    TB_FOLDED_STORE },
    { X86::MOVUPDrr,    X86::MOVUPDmr,      TB_FOLDED_STORE },
    { X86::MOVUPSrr,    X86::MOVUPSmr,      TB_FOLDED_STORE },
    { X86::MUL16r,      X86::MUL16m,        TB_FOLDED_LOAD },
    { X86::MUL32r,      X86::MUL32m,        TB_FOLDED_LOAD },
    { X86::MUL64r,      X86::MUL64m,        TB_FOLDED_LOAD },
    { X86::MUL8r,       X86::MUL8m,         TB_FOLDED_LOAD },
    { X86::PEXTRDrr,    X86::PEXTRDmr,      TB_FOLDED_STORE },
    { X86::PEXTRQrr,    X86::PEXTRQmr,      TB_FOLDED_STORE },
    { X86::PUSH16r,     X86::PUSH16rmm,     TB_FOLDED_LOAD },
    { X86::PUSH32r,     X86::PUSH32rmm,     TB_FOLDED_LOAD },
    { X86::PUSH64r,     X86::PUSH64rmm,     TB_FOLDED_LOAD },
    { X86::SETAEr,      X86::SETAEm,        TB_FOLDED_STORE },
    { X86::SETAr,       X86::SETAm,         TB_FOLDED_STORE },
    { X86::SETBEr,      X86::SETBEm,        TB_FOLDED_STORE },
    { X86::SETBr,       X86::SETBm,         TB_FOLDED_STORE },
    { X86::SETEr,       X86::SETEm,         TB_FOLDED_STORE },
    { X86::SETGEr,      X86::SETGEm,        TB_FOLDED_STORE },
    { X86::SETGr,       X86::SETGm,         TB_FOLDED_STORE },
    { X86::SETLEr,      X86::SETLEm,        TB_FOLDED_STORE },
    { X86::SETLr,       X86::SETLm,         TB_FOLDED_STORE },
    { X86::SETNEr,      X86::SETNEm,        TB_FOLDED_STORE },
    { X86::SETNOr,      X86::SETNOm,        TB_FOLDED_STORE },
    { X86::SETNPr,      X86::SETNPm,        TB_FOLDED_STORE },
    { X86::SETNSr,      X86::SETNSm,        TB_FOLDED_STORE },
    { X86::SETOr,       X86::SETOm,         TB_FOLDED_STORE },
    { X86::SETPr,       X86::SETPm,         TB_FOLDED_STORE },
    { X86::SETSr,       X86::SETSm,         TB_FOLDED_STORE },
    { X86::TAILJMPr,    X86::TAILJMPm,      TB_FOLDED_LOAD },
    { X86::TAILJMPr64,  X86::TAILJMPm64,    TB_FOLDED_LOAD },
    { X86::TAILJMPr64_REX, X86::TAILJMPm64_REX, TB_FOLDED_LOAD },
    { X86::TEST16ri,    X86::TEST16mi,      TB_FOLDED_LOAD },
    { X86::TEST32ri,    X86::TEST32mi,      TB_FOLDED_LOAD },
    { X86::TEST64ri32,  X86::TEST64mi32,    TB_FOLDED_LOAD },
    { X86::TEST8ri,     X86::TEST8mi,       TB_FOLDED_LOAD },

    // AVX 128-bit versions of foldable instructions
    { X86::VEXTRACTPSrr,X86::VEXTRACTPSmr,  TB_FOLDED_STORE  },
    { X86::VEXTRACTF128rr, X86::VEXTRACTF128mr, TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVAPDrr,   X86::VMOVAPDmr,     TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVAPSrr,   X86::VMOVAPSmr,     TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVDQArr,   X86::VMOVDQAmr,     TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVPDI2DIrr,X86::VMOVPDI2DImr,  TB_FOLDED_STORE },
    { X86::VMOVPQIto64rr, X86::VMOVPQI2QImr,TB_FOLDED_STORE },
    { X86::VMOVSDto64rr,X86::VMOVSDto64mr,  TB_FOLDED_STORE },
    { X86::VMOVSS2DIrr, X86::VMOVSS2DImr,   TB_FOLDED_STORE },
    { X86::VMOVUPDrr,   X86::VMOVUPDmr,     TB_FOLDED_STORE },
    { X86::VMOVUPSrr,   X86::VMOVUPSmr,     TB_FOLDED_STORE },
    { X86::VPEXTRDrr,   X86::VPEXTRDmr,     TB_FOLDED_STORE },
    { X86::VPEXTRQrr,   X86::VPEXTRQmr,     TB_FOLDED_STORE },

    // AVX 256-bit foldable instructions
    { X86::VEXTRACTI128rr, X86::VEXTRACTI128mr, TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVAPDYrr,  X86::VMOVAPDYmr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVAPSYrr,  X86::VMOVAPSYmr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVDQAYrr,  X86::VMOVDQAYmr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVUPDYrr,  X86::VMOVUPDYmr,    TB_FOLDED_STORE },
    { X86::VMOVUPSYrr,  X86::VMOVUPSYmr,    TB_FOLDED_STORE },

    // AVX-512 foldable instructions
    { X86::VMOVPDI2DIZrr,   X86::VMOVPDI2DIZmr, TB_FOLDED_STORE },
    { X86::VMOVAPDZrr,      X86::VMOVAPDZmr,    TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVAPSZrr,      X86::VMOVAPSZmr,    TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVDQA32Zrr,    X86::VMOVDQA32Zmr,  TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVDQA64Zrr,    X86::VMOVDQA64Zmr,  TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVUPDZrr,      X86::VMOVUPDZmr,    TB_FOLDED_STORE },
    { X86::VMOVUPSZrr,      X86::VMOVUPSZmr,    TB_FOLDED_STORE },
    { X86::VMOVDQU8Zrr,     X86::VMOVDQU8Zmr,   TB_FOLDED_STORE },
    { X86::VMOVDQU16Zrr,    X86::VMOVDQU16Zmr,  TB_FOLDED_STORE },
    { X86::VMOVDQU32Zrr,    X86::VMOVDQU32Zmr,  TB_FOLDED_STORE },
    { X86::VMOVDQU64Zrr,    X86::VMOVDQU64Zmr,  TB_FOLDED_STORE },

    // AVX-512 foldable instructions (256-bit versions)
    { X86::VMOVAPDZ256rr,      X86::VMOVAPDZ256mr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVAPSZ256rr,      X86::VMOVAPSZ256mr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVDQA32Z256rr,    X86::VMOVDQA32Z256mr,  TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVDQA64Z256rr,    X86::VMOVDQA64Z256mr,  TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVUPDZ256rr,      X86::VMOVUPDZ256mr,    TB_FOLDED_STORE },
    { X86::VMOVUPSZ256rr,      X86::VMOVUPSZ256mr,    TB_FOLDED_STORE },
    { X86::VMOVDQU8Z256rr,     X86::VMOVDQU8Z256mr,   TB_FOLDED_STORE },
    { X86::VMOVDQU16Z256rr,    X86::VMOVDQU16Z256mr,  TB_FOLDED_STORE },
    { X86::VMOVDQU32Z256rr,    X86::VMOVDQU32Z256mr,  TB_FOLDED_STORE },
    { X86::VMOVDQU64Z256rr,    X86::VMOVDQU64Z256mr,  TB_FOLDED_STORE },

    // AVX-512 foldable instructions (128-bit versions)
    { X86::VMOVAPDZ128rr,      X86::VMOVAPDZ128mr,    TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVAPSZ128rr,      X86::VMOVAPSZ128mr,    TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVDQA32Z128rr,    X86::VMOVDQA32Z128mr,  TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVDQA64Z128rr,    X86::VMOVDQA64Z128mr,  TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVUPDZ128rr,      X86::VMOVUPDZ128mr,    TB_FOLDED_STORE },
    { X86::VMOVUPSZ128rr,      X86::VMOVUPSZ128mr,    TB_FOLDED_STORE },
    { X86::VMOVDQU8Z128rr,     X86::VMOVDQU8Z128mr,   TB_FOLDED_STORE },
    { X86::VMOVDQU16Z128rr,    X86::VMOVDQU16Z128mr,  TB_FOLDED_STORE },
    { X86::VMOVDQU32Z128rr,    X86::VMOVDQU32Z128mr,  TB_FOLDED_STORE },
    { X86::VMOVDQU64Z128rr,    X86::VMOVDQU64Z128mr,  TB_FOLDED_STORE },

    // F16C foldable instructions
    { X86::VCVTPS2PHrr,        X86::VCVTPS2PHmr,      TB_FOLDED_STORE },
    { X86::VCVTPS2PHYrr,       X86::VCVTPS2PHYmr,     TB_FOLDED_STORE }
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable0) {
    AddTableEntry(RegOp2MemOpTable0, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp, TB_INDEX_0 | Entry.Flags);
  }

  static const X86MemoryFoldTableEntry MemoryFoldTable1[] = {
    { X86::BSF16rr,         X86::BSF16rm,             0 },
    { X86::BSF32rr,         X86::BSF32rm,             0 },
    { X86::BSF64rr,         X86::BSF64rm,             0 },
    { X86::BSR16rr,         X86::BSR16rm,             0 },
    { X86::BSR32rr,         X86::BSR32rm,             0 },
    { X86::BSR64rr,         X86::BSR64rm,             0 },
    { X86::CMP16rr,         X86::CMP16rm,             0 },
    { X86::CMP32rr,         X86::CMP32rm,             0 },
    { X86::CMP64rr,         X86::CMP64rm,             0 },
    { X86::CMP8rr,          X86::CMP8rm,              0 },
    { X86::CVTSD2SSrr,      X86::CVTSD2SSrm,          0 },
    { X86::CVTSI2SD64rr,    X86::CVTSI2SD64rm,        0 },
    { X86::CVTSI2SDrr,      X86::CVTSI2SDrm,          0 },
    { X86::CVTSI2SS64rr,    X86::CVTSI2SS64rm,        0 },
    { X86::CVTSI2SSrr,      X86::CVTSI2SSrm,          0 },
    { X86::CVTSS2SDrr,      X86::CVTSS2SDrm,          0 },
    { X86::CVTTSD2SI64rr,   X86::CVTTSD2SI64rm,       0 },
    { X86::CVTTSD2SIrr,     X86::CVTTSD2SIrm,         0 },
    { X86::CVTTSS2SI64rr,   X86::CVTTSS2SI64rm,       0 },
    { X86::CVTTSS2SIrr,     X86::CVTTSS2SIrm,         0 },
    { X86::IMUL16rri,       X86::IMUL16rmi,           0 },
    { X86::IMUL16rri8,      X86::IMUL16rmi8,          0 },
    { X86::IMUL32rri,       X86::IMUL32rmi,           0 },
    { X86::IMUL32rri8,      X86::IMUL32rmi8,          0 },
    { X86::IMUL64rri32,     X86::IMUL64rmi32,         0 },
    { X86::IMUL64rri8,      X86::IMUL64rmi8,          0 },
    { X86::Int_COMISDrr,    X86::Int_COMISDrm,        0 },
    { X86::Int_COMISSrr,    X86::Int_COMISSrm,        0 },
    { X86::CVTSD2SI64rr,    X86::CVTSD2SI64rm,        0 },
    { X86::CVTSD2SIrr,      X86::CVTSD2SIrm,          0 },
    { X86::CVTSS2SI64rr,    X86::CVTSS2SI64rm,        0 },
    { X86::CVTSS2SIrr,      X86::CVTSS2SIrm,          0 },
    { X86::CVTDQ2PDrr,      X86::CVTDQ2PDrm,          TB_ALIGN_16 },
    { X86::CVTDQ2PSrr,      X86::CVTDQ2PSrm,          TB_ALIGN_16 },
    { X86::CVTPD2DQrr,      X86::CVTPD2DQrm,          TB_ALIGN_16 },
    { X86::CVTPD2PSrr,      X86::CVTPD2PSrm,          TB_ALIGN_16 },
    { X86::CVTPS2DQrr,      X86::CVTPS2DQrm,          TB_ALIGN_16 },
    { X86::CVTPS2PDrr,      X86::CVTPS2PDrm,          TB_ALIGN_16 },
    { X86::CVTTPD2DQrr,     X86::CVTTPD2DQrm,         TB_ALIGN_16 },
    { X86::CVTTPS2DQrr,     X86::CVTTPS2DQrm,         TB_ALIGN_16 },
    { X86::Int_CVTTSD2SI64rr,X86::Int_CVTTSD2SI64rm,  0 },
    { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm,     0 },
    { X86::Int_CVTTSS2SI64rr,X86::Int_CVTTSS2SI64rm,  0 },
    { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm,     0 },
    { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm,       0 },
    { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm,       0 },
    { X86::MOV16rr,         X86::MOV16rm,             0 },
    { X86::MOV32rr,         X86::MOV32rm,             0 },
    { X86::MOV64rr,         X86::MOV64rm,             0 },
    { X86::MOV64toPQIrr,    X86::MOVQI2PQIrm,         0 },
    { X86::MOV64toSDrr,     X86::MOV64toSDrm,         0 },
    { X86::MOV8rr,          X86::MOV8rm,              0 },
    { X86::MOVAPDrr,        X86::MOVAPDrm,            TB_ALIGN_16 },
    { X86::MOVAPSrr,        X86::MOVAPSrm,            TB_ALIGN_16 },
    { X86::MOVDDUPrr,       X86::MOVDDUPrm,           0 },
    { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm,         0 },
    { X86::MOVDI2SSrr,      X86::MOVDI2SSrm,          0 },
    { X86::MOVDQArr,        X86::MOVDQArm,            TB_ALIGN_16 },
    { X86::MOVSHDUPrr,      X86::MOVSHDUPrm,          TB_ALIGN_16 },
    { X86::MOVSLDUPrr,      X86::MOVSLDUPrm,          TB_ALIGN_16 },
    { X86::MOVSX16rr8,      X86::MOVSX16rm8,          0 },
    { X86::MOVSX32rr16,     X86::MOVSX32rm16,         0 },
    { X86::MOVSX32rr8,      X86::MOVSX32rm8,          0 },
    { X86::MOVSX64rr16,     X86::MOVSX64rm16,         0 },
    { X86::MOVSX64rr32,     X86::MOVSX64rm32,         0 },
    { X86::MOVSX64rr8,      X86::MOVSX64rm8,          0 },
    { X86::MOVUPDrr,        X86::MOVUPDrm,            TB_ALIGN_16 },
    { X86::MOVUPSrr,        X86::MOVUPSrm,            0 },
    { X86::MOVZPQILo2PQIrr, X86::MOVZPQILo2PQIrm,     TB_ALIGN_16 },
    { X86::MOVZX16rr8,      X86::MOVZX16rm8,          0 },
    { X86::MOVZX32rr16,     X86::MOVZX32rm16,         0 },
    { X86::MOVZX32_NOREXrr8, X86::MOVZX32_NOREXrm8,   0 },
    { X86::MOVZX32rr8,      X86::MOVZX32rm8,          0 },
    { X86::PABSBrr128,      X86::PABSBrm128,          TB_ALIGN_16 },
    { X86::PABSDrr128,      X86::PABSDrm128,          TB_ALIGN_16 },
    { X86::PABSWrr128,      X86::PABSWrm128,          TB_ALIGN_16 },
    { X86::PCMPESTRIrr,     X86::PCMPESTRIrm,         TB_ALIGN_16 },
    { X86::PCMPESTRM128rr,  X86::PCMPESTRM128rm,      TB_ALIGN_16 },
    { X86::PCMPISTRIrr,     X86::PCMPISTRIrm,         TB_ALIGN_16 },
    { X86::PCMPISTRM128rr,  X86::PCMPISTRM128rm,      TB_ALIGN_16 },
    { X86::PHMINPOSUWrr128, X86::PHMINPOSUWrm128,     TB_ALIGN_16 },
    { X86::PMOVSXBDrr,      X86::PMOVSXBDrm,          TB_ALIGN_16 },
    { X86::PMOVSXBQrr,      X86::PMOVSXBQrm,          TB_ALIGN_16 },
    { X86::PMOVSXBWrr,      X86::PMOVSXBWrm,          TB_ALIGN_16 },
    { X86::PMOVSXDQrr,      X86::PMOVSXDQrm,          TB_ALIGN_16 },
    { X86::PMOVSXWDrr,      X86::PMOVSXWDrm,          TB_ALIGN_16 },
    { X86::PMOVSXWQrr,      X86::PMOVSXWQrm,          TB_ALIGN_16 },
    { X86::PMOVZXBDrr,      X86::PMOVZXBDrm,          TB_ALIGN_16 },
    { X86::PMOVZXBQrr,      X86::PMOVZXBQrm,          TB_ALIGN_16 },
    { X86::PMOVZXBWrr,      X86::PMOVZXBWrm,          TB_ALIGN_16 },
    { X86::PMOVZXDQrr,      X86::PMOVZXDQrm,          TB_ALIGN_16 },
    { X86::PMOVZXWDrr,      X86::PMOVZXWDrm,          TB_ALIGN_16 },
    { X86::PMOVZXWQrr,      X86::PMOVZXWQrm,          TB_ALIGN_16 },
    { X86::PSHUFDri,        X86::PSHUFDmi,            TB_ALIGN_16 },
    { X86::PSHUFHWri,       X86::PSHUFHWmi,           TB_ALIGN_16 },
    { X86::PSHUFLWri,       X86::PSHUFLWmi,           TB_ALIGN_16 },
    { X86::PTESTrr,         X86::PTESTrm,             TB_ALIGN_16 },
    { X86::RCPPSr,          X86::RCPPSm,              TB_ALIGN_16 },
    { X86::RCPSSr,          X86::RCPSSm,              0 },
    { X86::RCPSSr_Int,      X86::RCPSSm_Int,          0 },
    { X86::ROUNDPDr,        X86::ROUNDPDm,            TB_ALIGN_16 },
    { X86::ROUNDPSr,        X86::ROUNDPSm,            TB_ALIGN_16 },
    { X86::RSQRTPSr,        X86::RSQRTPSm,            TB_ALIGN_16 },
    { X86::RSQRTSSr,        X86::RSQRTSSm,            0 },
    { X86::RSQRTSSr_Int,    X86::RSQRTSSm_Int,        0 },
    { X86::SQRTPDr,         X86::SQRTPDm,             TB_ALIGN_16 },
    { X86::SQRTPSr,         X86::SQRTPSm,             TB_ALIGN_16 },
    { X86::SQRTSDr,         X86::SQRTSDm,             0 },
    { X86::SQRTSDr_Int,     X86::SQRTSDm_Int,         0 },
    { X86::SQRTSSr,         X86::SQRTSSm,             0 },
    { X86::SQRTSSr_Int,     X86::SQRTSSm_Int,         0 },
    { X86::TEST16rr,        X86::TEST16rm,            0 },
    { X86::TEST32rr,        X86::TEST32rm,            0 },
    { X86::TEST64rr,        X86::TEST64rm,            0 },
    { X86::TEST8rr,         X86::TEST8rm,             0 },
    // FIXME: TEST*rr EAX,EAX ---> CMP [mem], 0
    { X86::UCOMISDrr,       X86::UCOMISDrm,           0 },
    { X86::UCOMISSrr,       X86::UCOMISSrm,           0 },

    // MMX version of foldable instructions
    { X86::MMX_CVTPD2PIirr,   X86::MMX_CVTPD2PIirm,   0 },
    { X86::MMX_CVTPI2PDirr,   X86::MMX_CVTPI2PDirm,   0 },
    { X86::MMX_CVTPS2PIirr,   X86::MMX_CVTPS2PIirm,   0 },
    { X86::MMX_CVTTPD2PIirr,  X86::MMX_CVTTPD2PIirm,  0 },
    { X86::MMX_CVTTPS2PIirr,  X86::MMX_CVTTPS2PIirm,  0 },
    { X86::MMX_MOVD64to64rr,  X86::MMX_MOVQ64rm,      0 },
    { X86::MMX_PABSBrr64,     X86::MMX_PABSBrm64,     0 },
    { X86::MMX_PABSDrr64,     X86::MMX_PABSDrm64,     0 },
    { X86::MMX_PABSWrr64,     X86::MMX_PABSWrm64,     0 },
    { X86::MMX_PSHUFWri,      X86::MMX_PSHUFWmi,      0 },

    // 3DNow! version of foldable instructions
    { X86::PF2IDrr,         X86::PF2IDrm,             0 },
    { X86::PF2IWrr,         X86::PF2IWrm,             0 },
    { X86::PFRCPrr,         X86::PFRCPrm,             0 },
    { X86::PFRSQRTrr,       X86::PFRSQRTrm,           0 },
    { X86::PI2FDrr,         X86::PI2FDrm,             0 },
    { X86::PI2FWrr,         X86::PI2FWrm,             0 },
    { X86::PSWAPDrr,        X86::PSWAPDrm,            0 },

    // AVX 128-bit versions of foldable instructions
    { X86::Int_VCOMISDrr,   X86::Int_VCOMISDrm,       0 },
    { X86::Int_VCOMISSrr,   X86::Int_VCOMISSrm,       0 },
    { X86::Int_VUCOMISDrr,  X86::Int_VUCOMISDrm,      0 },
    { X86::Int_VUCOMISSrr,  X86::Int_VUCOMISSrm,      0 },
    { X86::VCVTTSD2SI64rr,  X86::VCVTTSD2SI64rm,      0 },
    { X86::Int_VCVTTSD2SI64rr,X86::Int_VCVTTSD2SI64rm,0 },
    { X86::VCVTTSD2SIrr,    X86::VCVTTSD2SIrm,        0 },
    { X86::Int_VCVTTSD2SIrr,X86::Int_VCVTTSD2SIrm,    0 },
    { X86::VCVTTSS2SI64rr,  X86::VCVTTSS2SI64rm,      0 },
    { X86::Int_VCVTTSS2SI64rr,X86::Int_VCVTTSS2SI64rm,0 },
    { X86::VCVTTSS2SIrr,    X86::VCVTTSS2SIrm,        0 },
    { X86::Int_VCVTTSS2SIrr,X86::Int_VCVTTSS2SIrm,    0 },
    { X86::VCVTSD2SI64rr,   X86::VCVTSD2SI64rm,       0 },
    { X86::VCVTSD2SIrr,     X86::VCVTSD2SIrm,         0 },
    { X86::VCVTSS2SI64rr,   X86::VCVTSS2SI64rm,       0 },
    { X86::VCVTSS2SIrr,     X86::VCVTSS2SIrm,         0 },
    { X86::VCVTDQ2PDrr,     X86::VCVTDQ2PDrm,         0 },
    { X86::VCVTDQ2PSrr,     X86::VCVTDQ2PSrm,         0 },
    { X86::VCVTPD2DQrr,     X86::VCVTPD2DQXrm,        0 },
    { X86::VCVTPD2PSrr,     X86::VCVTPD2PSXrm,        0 },
    { X86::VCVTPS2DQrr,     X86::VCVTPS2DQrm,         0 },
    { X86::VCVTPS2PDrr,     X86::VCVTPS2PDrm,         0 },
    { X86::VCVTTPD2DQrr,    X86::VCVTTPD2DQXrm,       0 },
    { X86::VCVTTPS2DQrr,    X86::VCVTTPS2DQrm,        0 },
    { X86::VMOV64toPQIrr,   X86::VMOVQI2PQIrm,        0 },
    { X86::VMOV64toSDrr,    X86::VMOV64toSDrm,        0 },
    { X86::VMOVAPDrr,       X86::VMOVAPDrm,           TB_ALIGN_16 },
    { X86::VMOVAPSrr,       X86::VMOVAPSrm,           TB_ALIGN_16 },
    { X86::VMOVDDUPrr,      X86::VMOVDDUPrm,          0 },
    { X86::VMOVDI2PDIrr,    X86::VMOVDI2PDIrm,        0 },
    { X86::VMOVDI2SSrr,     X86::VMOVDI2SSrm,         0 },
    { X86::VMOVDQArr,       X86::VMOVDQArm,           TB_ALIGN_16 },
    { X86::VMOVSLDUPrr,     X86::VMOVSLDUPrm,         0 },
    { X86::VMOVSHDUPrr,     X86::VMOVSHDUPrm,         0 },
    { X86::VMOVUPDrr,       X86::VMOVUPDrm,           0 },
    { X86::VMOVUPSrr,       X86::VMOVUPSrm,           0 },
    { X86::VMOVZPQILo2PQIrr,X86::VMOVZPQILo2PQIrm,    TB_ALIGN_16 },
    { X86::VPABSBrr128,     X86::VPABSBrm128,         0 },
    { X86::VPABSDrr128,     X86::VPABSDrm128,         0 },
    { X86::VPABSWrr128,     X86::VPABSWrm128,         0 },
    { X86::VPCMPESTRIrr,    X86::VPCMPESTRIrm,        0 },
    { X86::VPCMPESTRM128rr, X86::VPCMPESTRM128rm,     0 },
    { X86::VPCMPISTRIrr,    X86::VPCMPISTRIrm,        0 },
    { X86::VPCMPISTRM128rr, X86::VPCMPISTRM128rm,     0 },
    { X86::VPHMINPOSUWrr128, X86::VPHMINPOSUWrm128,   0 },
    { X86::VPERMILPDri,     X86::VPERMILPDmi,         0 },
    { X86::VPERMILPSri,     X86::VPERMILPSmi,         0 },
    { X86::VPMOVSXBDrr,     X86::VPMOVSXBDrm,         0 },
    { X86::VPMOVSXBQrr,     X86::VPMOVSXBQrm,         0 },
    { X86::VPMOVSXBWrr,     X86::VPMOVSXBWrm,         0 },
    { X86::VPMOVSXDQrr,     X86::VPMOVSXDQrm,         0 },
    { X86::VPMOVSXWDrr,     X86::VPMOVSXWDrm,         0 },
    { X86::VPMOVSXWQrr,     X86::VPMOVSXWQrm,         0 },
    { X86::VPMOVZXBDrr,     X86::VPMOVZXBDrm,         0 },
    { X86::VPMOVZXBQrr,     X86::VPMOVZXBQrm,         0 },
    { X86::VPMOVZXBWrr,     X86::VPMOVZXBWrm,         0 },
    { X86::VPMOVZXDQrr,     X86::VPMOVZXDQrm,         0 },
    { X86::VPMOVZXWDrr,     X86::VPMOVZXWDrm,         0 },
    { X86::VPMOVZXWQrr,     X86::VPMOVZXWQrm,         0 },
    { X86::VPSHUFDri,       X86::VPSHUFDmi,           0 },
    { X86::VPSHUFHWri,      X86::VPSHUFHWmi,          0 },
    { X86::VPSHUFLWri,      X86::VPSHUFLWmi,          0 },
    { X86::VPTESTrr,        X86::VPTESTrm,            0 },
    { X86::VRCPPSr,         X86::VRCPPSm,             0 },
    { X86::VROUNDPDr,       X86::VROUNDPDm,           0 },
    { X86::VROUNDPSr,       X86::VROUNDPSm,           0 },
    { X86::VRSQRTPSr,       X86::VRSQRTPSm,           0 },
    { X86::VSQRTPDr,        X86::VSQRTPDm,            0 },
    { X86::VSQRTPSr,        X86::VSQRTPSm,            0 },
    { X86::VTESTPDrr,       X86::VTESTPDrm,           0 },
    { X86::VTESTPSrr,       X86::VTESTPSrm,           0 },
    { X86::VUCOMISDrr,      X86::VUCOMISDrm,          0 },
    { X86::VUCOMISSrr,      X86::VUCOMISSrm,          0 },

    // AVX 256-bit foldable instructions
    { X86::VCVTDQ2PDYrr,    X86::VCVTDQ2PDYrm,        0 },
    { X86::VCVTDQ2PSYrr,    X86::VCVTDQ2PSYrm,        0 },
    { X86::VCVTPD2DQYrr,    X86::VCVTPD2DQYrm,        0 },
    { X86::VCVTPD2PSYrr,    X86::VCVTPD2PSYrm,        0 },
    { X86::VCVTPS2DQYrr,    X86::VCVTPS2DQYrm,        0 },
    { X86::VCVTPS2PDYrr,    X86::VCVTPS2PDYrm,        0 },
    { X86::VCVTTPD2DQYrr,   X86::VCVTTPD2DQYrm,       0 },
    { X86::VCVTTPS2DQYrr,   X86::VCVTTPS2DQYrm,       0 },
    { X86::VMOVAPDYrr,      X86::VMOVAPDYrm,          TB_ALIGN_32 },
    { X86::VMOVAPSYrr,      X86::VMOVAPSYrm,          TB_ALIGN_32 },
    { X86::VMOVDDUPYrr,     X86::VMOVDDUPYrm,         0 },
    { X86::VMOVDQAYrr,      X86::VMOVDQAYrm,          TB_ALIGN_32 },
    { X86::VMOVSLDUPYrr,    X86::VMOVSLDUPYrm,        0 },
    { X86::VMOVSHDUPYrr,    X86::VMOVSHDUPYrm,        0 },
    { X86::VMOVUPDYrr,      X86::VMOVUPDYrm,          0 },
    { X86::VMOVUPSYrr,      X86::VMOVUPSYrm,          0 },
    { X86::VPERMILPDYri,    X86::VPERMILPDYmi,        0 },
    { X86::VPERMILPSYri,    X86::VPERMILPSYmi,        0 },
    { X86::VPTESTYrr,       X86::VPTESTYrm,           0 },
    { X86::VRCPPSYr,        X86::VRCPPSYm,            0 },
    { X86::VROUNDYPDr,      X86::VROUNDYPDm,          0 },
    { X86::VROUNDYPSr,      X86::VROUNDYPSm,          0 },
    { X86::VRSQRTPSYr,      X86::VRSQRTPSYm,          0 },
    { X86::VSQRTPDYr,       X86::VSQRTPDYm,           0 },
    { X86::VSQRTPSYr,       X86::VSQRTPSYm,           0 },
    { X86::VTESTPDYrr,      X86::VTESTPDYrm,          0 },
    { X86::VTESTPSYrr,      X86::VTESTPSYrm,          0 },

    // AVX2 foldable instructions

    // VBROADCASTS{SD}rr register instructions were an AVX2 addition while the
    // VBROADCASTS{SD}rm memory instructions were available from AVX1.
    // TB_NO_REVERSE prevents unfolding from introducing an illegal instruction
    // on AVX1 targets. The VPBROADCAST instructions are all AVX2 instructions
    // so they don't need an equivalent limitation.
    { X86::VBROADCASTSSrr,  X86::VBROADCASTSSrm,      TB_NO_REVERSE },
    { X86::VBROADCASTSSYrr, X86::VBROADCASTSSYrm,     TB_NO_REVERSE },
    { X86::VBROADCASTSDYrr, X86::VBROADCASTSDYrm,     TB_NO_REVERSE },
    { X86::VPABSBrr256,     X86::VPABSBrm256,         0 },
    { X86::VPABSDrr256,     X86::VPABSDrm256,         0 },
    { X86::VPABSWrr256,     X86::VPABSWrm256,         0 },
    { X86::VPBROADCASTBrr,  X86::VPBROADCASTBrm,      0 },
    { X86::VPBROADCASTBYrr, X86::VPBROADCASTBYrm,     0 },
    { X86::VPBROADCASTDrr,  X86::VPBROADCASTDrm,      0 },
    { X86::VPBROADCASTDYrr, X86::VPBROADCASTDYrm,     0 },
    { X86::VPBROADCASTQrr,  X86::VPBROADCASTQrm,      0 },
    { X86::VPBROADCASTQYrr, X86::VPBROADCASTQYrm,     0 },
    { X86::VPBROADCASTWrr,  X86::VPBROADCASTWrm,      0 },
    { X86::VPBROADCASTWYrr, X86::VPBROADCASTWYrm,     0 },
    { X86::VPERMPDYri,      X86::VPERMPDYmi,          0 },
    { X86::VPERMQYri,       X86::VPERMQYmi,           0 },
    { X86::VPMOVSXBDYrr,    X86::VPMOVSXBDYrm,        0 },
    { X86::VPMOVSXBQYrr,    X86::VPMOVSXBQYrm,        0 },
    { X86::VPMOVSXBWYrr,    X86::VPMOVSXBWYrm,        0 },
    { X86::VPMOVSXDQYrr,    X86::VPMOVSXDQYrm,        0 },
    { X86::VPMOVSXWDYrr,    X86::VPMOVSXWDYrm,        0 },
    { X86::VPMOVSXWQYrr,    X86::VPMOVSXWQYrm,        0 },
    { X86::VPMOVZXBDYrr,    X86::VPMOVZXBDYrm,        0 },
    { X86::VPMOVZXBQYrr,    X86::VPMOVZXBQYrm,        0 },
    { X86::VPMOVZXBWYrr,    X86::VPMOVZXBWYrm,        0 },
    { X86::VPMOVZXDQYrr,    X86::VPMOVZXDQYrm,        0 },
    { X86::VPMOVZXWDYrr,    X86::VPMOVZXWDYrm,        0 },
    { X86::VPMOVZXWQYrr,    X86::VPMOVZXWQYrm,        0 },
    { X86::VPSHUFDYri,      X86::VPSHUFDYmi,          0 },
    { X86::VPSHUFHWYri,     X86::VPSHUFHWYmi,         0 },
    { X86::VPSHUFLWYri,     X86::VPSHUFLWYmi,         0 },

    // XOP foldable instructions
    { X86::VFRCZPDrr,          X86::VFRCZPDrm,        0 },
    { X86::VFRCZPDrrY,         X86::VFRCZPDrmY,       0 },
    { X86::VFRCZPSrr,          X86::VFRCZPSrm,        0 },
    { X86::VFRCZPSrrY,         X86::VFRCZPSrmY,       0 },
    { X86::VFRCZSDrr,          X86::VFRCZSDrm,        0 },
    { X86::VFRCZSSrr,          X86::VFRCZSSrm,        0 },
    { X86::VPHADDBDrr,         X86::VPHADDBDrm,       0 },
    { X86::VPHADDBQrr,         X86::VPHADDBQrm,       0 },
    { X86::VPHADDBWrr,         X86::VPHADDBWrm,       0 },
    { X86::VPHADDDQrr,         X86::VPHADDDQrm,       0 },
    { X86::VPHADDWDrr,         X86::VPHADDWDrm,       0 },
    { X86::VPHADDWQrr,         X86::VPHADDWQrm,       0 },
    { X86::VPHADDUBDrr,        X86::VPHADDUBDrm,      0 },
    { X86::VPHADDUBQrr,        X86::VPHADDUBQrm,      0 },
    { X86::VPHADDUBWrr,        X86::VPHADDUBWrm,      0 },
    { X86::VPHADDUDQrr,        X86::VPHADDUDQrm,      0 },
    { X86::VPHADDUWDrr,        X86::VPHADDUWDrm,      0 },
    { X86::VPHADDUWQrr,        X86::VPHADDUWQrm,      0 },
    { X86::VPHSUBBWrr,         X86::VPHSUBBWrm,       0 },
    { X86::VPHSUBDQrr,         X86::VPHSUBDQrm,       0 },
    { X86::VPHSUBWDrr,         X86::VPHSUBWDrm,       0 },
    { X86::VPROTBri,           X86::VPROTBmi,         0 },
    { X86::VPROTBrr,           X86::VPROTBmr,         0 },
    { X86::VPROTDri,           X86::VPROTDmi,         0 },
    { X86::VPROTDrr,           X86::VPROTDmr,         0 },
    { X86::VPROTQri,           X86::VPROTQmi,         0 },
    { X86::VPROTQrr,           X86::VPROTQmr,         0 },
    { X86::VPROTWri,           X86::VPROTWmi,         0 },
    { X86::VPROTWrr,           X86::VPROTWmr,         0 },
    { X86::VPSHABrr,           X86::VPSHABmr,         0 },
    { X86::VPSHADrr,           X86::VPSHADmr,         0 },
    { X86::VPSHAQrr,           X86::VPSHAQmr,         0 },
    { X86::VPSHAWrr,           X86::VPSHAWmr,         0 },
    { X86::VPSHLBrr,           X86::VPSHLBmr,         0 },
    { X86::VPSHLDrr,           X86::VPSHLDmr,         0 },
    { X86::VPSHLQrr,           X86::VPSHLQmr,         0 },
    { X86::VPSHLWrr,           X86::VPSHLWmr,         0 },

    // BMI/BMI2/LZCNT/POPCNT/TBM foldable instructions
    { X86::BEXTR32rr,       X86::BEXTR32rm,           0 },
    { X86::BEXTR64rr,       X86::BEXTR64rm,           0 },
    { X86::BEXTRI32ri,      X86::BEXTRI32mi,          0 },
    { X86::BEXTRI64ri,      X86::BEXTRI64mi,          0 },
    { X86::BLCFILL32rr,     X86::BLCFILL32rm,         0 },
    { X86::BLCFILL64rr,     X86::BLCFILL64rm,         0 },
    { X86::BLCI32rr,        X86::BLCI32rm,            0 },
    { X86::BLCI64rr,        X86::BLCI64rm,            0 },
    { X86::BLCIC32rr,       X86::BLCIC32rm,           0 },
    { X86::BLCIC64rr,       X86::BLCIC64rm,           0 },
    { X86::BLCMSK32rr,      X86::BLCMSK32rm,          0 },
    { X86::BLCMSK64rr,      X86::BLCMSK64rm,          0 },
    { X86::BLCS32rr,        X86::BLCS32rm,            0 },
    { X86::BLCS64rr,        X86::BLCS64rm,            0 },
    { X86::BLSFILL32rr,     X86::BLSFILL32rm,         0 },
    { X86::BLSFILL64rr,     X86::BLSFILL64rm,         0 },
    { X86::BLSI32rr,        X86::BLSI32rm,            0 },
    { X86::BLSI64rr,        X86::BLSI64rm,            0 },
    { X86::BLSIC32rr,       X86::BLSIC32rm,           0 },
    { X86::BLSIC64rr,       X86::BLSIC64rm,           0 },
    { X86::BLSMSK32rr,      X86::BLSMSK32rm,          0 },
    { X86::BLSMSK64rr,      X86::BLSMSK64rm,          0 },
    { X86::BLSR32rr,        X86::BLSR32rm,            0 },
    { X86::BLSR64rr,        X86::BLSR64rm,            0 },
    { X86::BZHI32rr,        X86::BZHI32rm,            0 },
    { X86::BZHI64rr,        X86::BZHI64rm,            0 },
    { X86::LZCNT16rr,       X86::LZCNT16rm,           0 },
    { X86::LZCNT32rr,       X86::LZCNT32rm,           0 },
    { X86::LZCNT64rr,       X86::LZCNT64rm,           0 },
    { X86::POPCNT16rr,      X86::POPCNT16rm,          0 },
    { X86::POPCNT32rr,      X86::POPCNT32rm,          0 },
    { X86::POPCNT64rr,      X86::POPCNT64rm,          0 },
    { X86::RORX32ri,        X86::RORX32mi,            0 },
    { X86::RORX64ri,        X86::RORX64mi,            0 },
    { X86::SARX32rr,        X86::SARX32rm,            0 },
    { X86::SARX64rr,        X86::SARX64rm,            0 },
    { X86::SHRX32rr,        X86::SHRX32rm,            0 },
    { X86::SHRX64rr,        X86::SHRX64rm,            0 },
    { X86::SHLX32rr,        X86::SHLX32rm,            0 },
    { X86::SHLX64rr,        X86::SHLX64rm,            0 },
    { X86::T1MSKC32rr,      X86::T1MSKC32rm,          0 },
    { X86::T1MSKC64rr,      X86::T1MSKC64rm,          0 },
    { X86::TZCNT16rr,       X86::TZCNT16rm,           0 },
    { X86::TZCNT32rr,       X86::TZCNT32rm,           0 },
    { X86::TZCNT64rr,       X86::TZCNT64rm,           0 },
    { X86::TZMSK32rr,       X86::TZMSK32rm,           0 },
    { X86::TZMSK64rr,       X86::TZMSK64rm,           0 },

    // AVX-512 foldable instructions
    { X86::VMOV64toPQIZrr,  X86::VMOVQI2PQIZrm,       0 },
    { X86::VMOVDI2SSZrr,    X86::VMOVDI2SSZrm,        0 },
    { X86::VMOVAPDZrr,      X86::VMOVAPDZrm,          TB_ALIGN_64 },
    { X86::VMOVAPSZrr,      X86::VMOVAPSZrm,          TB_ALIGN_64 },
    { X86::VMOVDQA32Zrr,    X86::VMOVDQA32Zrm,        TB_ALIGN_64 },
    { X86::VMOVDQA64Zrr,    X86::VMOVDQA64Zrm,        TB_ALIGN_64 },
    { X86::VMOVDQU8Zrr,     X86::VMOVDQU8Zrm,         0 },
    { X86::VMOVDQU16Zrr,    X86::VMOVDQU16Zrm,        0 },
    { X86::VMOVDQU32Zrr,    X86::VMOVDQU32Zrm,        0 },
    { X86::VMOVDQU64Zrr,    X86::VMOVDQU64Zrm,        0 },
    { X86::VMOVUPDZrr,      X86::VMOVUPDZrm,          0 },
    { X86::VMOVUPSZrr,      X86::VMOVUPSZrm,          0 },
    { X86::VPABSDZrr,       X86::VPABSDZrm,           0 },
    { X86::VPABSQZrr,       X86::VPABSQZrm,           0 },
    { X86::VBROADCASTSSZr,  X86::VBROADCASTSSZm,      TB_NO_REVERSE },
    { X86::VBROADCASTSDZr,  X86::VBROADCASTSDZm,      TB_NO_REVERSE },

    // AVX-512 foldable instructions (256-bit versions)
    { X86::VMOVAPDZ256rr,      X86::VMOVAPDZ256rm,          TB_ALIGN_32 },
    { X86::VMOVAPSZ256rr,      X86::VMOVAPSZ256rm,          TB_ALIGN_32 },
    { X86::VMOVDQA32Z256rr,    X86::VMOVDQA32Z256rm,        TB_ALIGN_32 },
    { X86::VMOVDQA64Z256rr,    X86::VMOVDQA64Z256rm,        TB_ALIGN_32 },
    { X86::VMOVDQU8Z256rr,     X86::VMOVDQU8Z256rm,         0 },
    { X86::VMOVDQU16Z256rr,    X86::VMOVDQU16Z256rm,        0 },
    { X86::VMOVDQU32Z256rr,    X86::VMOVDQU32Z256rm,        0 },
    { X86::VMOVDQU64Z256rr,    X86::VMOVDQU64Z256rm,        0 },
    { X86::VMOVUPDZ256rr,      X86::VMOVUPDZ256rm,          0 },
    { X86::VMOVUPSZ256rr,      X86::VMOVUPSZ256rm,          0 },
    { X86::VBROADCASTSSZ256r,  X86::VBROADCASTSSZ256m,      TB_NO_REVERSE },
    { X86::VBROADCASTSDZ256r,  X86::VBROADCASTSDZ256m,      TB_NO_REVERSE },

    // AVX-512 foldable instructions (256-bit versions)
    { X86::VMOVAPDZ128rr,      X86::VMOVAPDZ128rm,          TB_ALIGN_16 },
    { X86::VMOVAPSZ128rr,      X86::VMOVAPSZ128rm,          TB_ALIGN_16 },
    { X86::VMOVDQA32Z128rr,    X86::VMOVDQA32Z128rm,        TB_ALIGN_16 },
    { X86::VMOVDQA64Z128rr,    X86::VMOVDQA64Z128rm,        TB_ALIGN_16 },
    { X86::VMOVDQU8Z128rr,     X86::VMOVDQU8Z128rm,         0 },
    { X86::VMOVDQU16Z128rr,    X86::VMOVDQU16Z128rm,        0 },
    { X86::VMOVDQU32Z128rr,    X86::VMOVDQU32Z128rm,        0 },
    { X86::VMOVDQU64Z128rr,    X86::VMOVDQU64Z128rm,        0 },
    { X86::VMOVUPDZ128rr,      X86::VMOVUPDZ128rm,          0 },
    { X86::VMOVUPSZ128rr,      X86::VMOVUPSZ128rm,          0 },
    { X86::VBROADCASTSSZ128r,  X86::VBROADCASTSSZ128m,      TB_NO_REVERSE },

    // F16C foldable instructions
    { X86::VCVTPH2PSrr,        X86::VCVTPH2PSrm,            0 },
    { X86::VCVTPH2PSYrr,       X86::VCVTPH2PSYrm,           0 },

    // AES foldable instructions
    { X86::AESIMCrr,              X86::AESIMCrm,              TB_ALIGN_16 },
    { X86::AESKEYGENASSIST128rr,  X86::AESKEYGENASSIST128rm,  TB_ALIGN_16 },
    { X86::VAESIMCrr,             X86::VAESIMCrm,             0 },
    { X86::VAESKEYGENASSIST128rr, X86::VAESKEYGENASSIST128rm, 0 }
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable1) {
    AddTableEntry(RegOp2MemOpTable1, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 1, folded load
                  Entry.Flags | TB_INDEX_1 | TB_FOLDED_LOAD);
  }

  static const X86MemoryFoldTableEntry MemoryFoldTable2[] = {
    { X86::ADC32rr,         X86::ADC32rm,       0 },
    { X86::ADC64rr,         X86::ADC64rm,       0 },
    { X86::ADD16rr,         X86::ADD16rm,       0 },
    { X86::ADD16rr_DB,      X86::ADD16rm,       TB_NO_REVERSE },
    { X86::ADD32rr,         X86::ADD32rm,       0 },
    { X86::ADD32rr_DB,      X86::ADD32rm,       TB_NO_REVERSE },
    { X86::ADD64rr,         X86::ADD64rm,       0 },
    { X86::ADD64rr_DB,      X86::ADD64rm,       TB_NO_REVERSE },
    { X86::ADD8rr,          X86::ADD8rm,        0 },
    { X86::ADDPDrr,         X86::ADDPDrm,       TB_ALIGN_16 },
    { X86::ADDPSrr,         X86::ADDPSrm,       TB_ALIGN_16 },
    { X86::ADDSDrr,         X86::ADDSDrm,       0 },
    { X86::ADDSDrr_Int,     X86::ADDSDrm_Int,   0 },
    { X86::ADDSSrr,         X86::ADDSSrm,       0 },
    { X86::ADDSSrr_Int,     X86::ADDSSrm_Int,   0 },
    { X86::ADDSUBPDrr,      X86::ADDSUBPDrm,    TB_ALIGN_16 },
    { X86::ADDSUBPSrr,      X86::ADDSUBPSrm,    TB_ALIGN_16 },
    { X86::AND16rr,         X86::AND16rm,       0 },
    { X86::AND32rr,         X86::AND32rm,       0 },
    { X86::AND64rr,         X86::AND64rm,       0 },
    { X86::AND8rr,          X86::AND8rm,        0 },
    { X86::ANDNPDrr,        X86::ANDNPDrm,      TB_ALIGN_16 },
    { X86::ANDNPSrr,        X86::ANDNPSrm,      TB_ALIGN_16 },
    { X86::ANDPDrr,         X86::ANDPDrm,       TB_ALIGN_16 },
    { X86::ANDPSrr,         X86::ANDPSrm,       TB_ALIGN_16 },
    { X86::BLENDPDrri,      X86::BLENDPDrmi,    TB_ALIGN_16 },
    { X86::BLENDPSrri,      X86::BLENDPSrmi,    TB_ALIGN_16 },
    { X86::BLENDVPDrr0,     X86::BLENDVPDrm0,   TB_ALIGN_16 },
    { X86::BLENDVPSrr0,     X86::BLENDVPSrm0,   TB_ALIGN_16 },
    { X86::CMOVA16rr,       X86::CMOVA16rm,     0 },
    { X86::CMOVA32rr,       X86::CMOVA32rm,     0 },
    { X86::CMOVA64rr,       X86::CMOVA64rm,     0 },
    { X86::CMOVAE16rr,      X86::CMOVAE16rm,    0 },
    { X86::CMOVAE32rr,      X86::CMOVAE32rm,    0 },
    { X86::CMOVAE64rr,      X86::CMOVAE64rm,    0 },
    { X86::CMOVB16rr,       X86::CMOVB16rm,     0 },
    { X86::CMOVB32rr,       X86::CMOVB32rm,     0 },
    { X86::CMOVB64rr,       X86::CMOVB64rm,     0 },
    { X86::CMOVBE16rr,      X86::CMOVBE16rm,    0 },
    { X86::CMOVBE32rr,      X86::CMOVBE32rm,    0 },
    { X86::CMOVBE64rr,      X86::CMOVBE64rm,    0 },
    { X86::CMOVE16rr,       X86::CMOVE16rm,     0 },
    { X86::CMOVE32rr,       X86::CMOVE32rm,     0 },
    { X86::CMOVE64rr,       X86::CMOVE64rm,     0 },
    { X86::CMOVG16rr,       X86::CMOVG16rm,     0 },
    { X86::CMOVG32rr,       X86::CMOVG32rm,     0 },
    { X86::CMOVG64rr,       X86::CMOVG64rm,     0 },
    { X86::CMOVGE16rr,      X86::CMOVGE16rm,    0 },
    { X86::CMOVGE32rr,      X86::CMOVGE32rm,    0 },
    { X86::CMOVGE64rr,      X86::CMOVGE64rm,    0 },
    { X86::CMOVL16rr,       X86::CMOVL16rm,     0 },
    { X86::CMOVL32rr,       X86::CMOVL32rm,     0 },
    { X86::CMOVL64rr,       X86::CMOVL64rm,     0 },
    { X86::CMOVLE16rr,      X86::CMOVLE16rm,    0 },
    { X86::CMOVLE32rr,      X86::CMOVLE32rm,    0 },
    { X86::CMOVLE64rr,      X86::CMOVLE64rm,    0 },
    { X86::CMOVNE16rr,      X86::CMOVNE16rm,    0 },
    { X86::CMOVNE32rr,      X86::CMOVNE32rm,    0 },
    { X86::CMOVNE64rr,      X86::CMOVNE64rm,    0 },
    { X86::CMOVNO16rr,      X86::CMOVNO16rm,    0 },
    { X86::CMOVNO32rr,      X86::CMOVNO32rm,    0 },
    { X86::CMOVNO64rr,      X86::CMOVNO64rm,    0 },
    { X86::CMOVNP16rr,      X86::CMOVNP16rm,    0 },
    { X86::CMOVNP32rr,      X86::CMOVNP32rm,    0 },
    { X86::CMOVNP64rr,      X86::CMOVNP64rm,    0 },
    { X86::CMOVNS16rr,      X86::CMOVNS16rm,    0 },
    { X86::CMOVNS32rr,      X86::CMOVNS32rm,    0 },
    { X86::CMOVNS64rr,      X86::CMOVNS64rm,    0 },
    { X86::CMOVO16rr,       X86::CMOVO16rm,     0 },
    { X86::CMOVO32rr,       X86::CMOVO32rm,     0 },
    { X86::CMOVO64rr,       X86::CMOVO64rm,     0 },
    { X86::CMOVP16rr,       X86::CMOVP16rm,     0 },
    { X86::CMOVP32rr,       X86::CMOVP32rm,     0 },
    { X86::CMOVP64rr,       X86::CMOVP64rm,     0 },
    { X86::CMOVS16rr,       X86::CMOVS16rm,     0 },
    { X86::CMOVS32rr,       X86::CMOVS32rm,     0 },
    { X86::CMOVS64rr,       X86::CMOVS64rm,     0 },
    { X86::CMPPDrri,        X86::CMPPDrmi,      TB_ALIGN_16 },
    { X86::CMPPSrri,        X86::CMPPSrmi,      TB_ALIGN_16 },
    { X86::CMPSDrr,         X86::CMPSDrm,       0 },
    { X86::CMPSSrr,         X86::CMPSSrm,       0 },
    { X86::CRC32r32r32,     X86::CRC32r32m32,   0 },
    { X86::CRC32r64r64,     X86::CRC32r64m64,   0 },
    { X86::DIVPDrr,         X86::DIVPDrm,       TB_ALIGN_16 },
    { X86::DIVPSrr,         X86::DIVPSrm,       TB_ALIGN_16 },
    { X86::DIVSDrr,         X86::DIVSDrm,       0 },
    { X86::DIVSDrr_Int,     X86::DIVSDrm_Int,   0 },
    { X86::DIVSSrr,         X86::DIVSSrm,       0 },
    { X86::DIVSSrr_Int,     X86::DIVSSrm_Int,   0 },
    { X86::DPPDrri,         X86::DPPDrmi,       TB_ALIGN_16 },
    { X86::DPPSrri,         X86::DPPSrmi,       TB_ALIGN_16 },

    // Do not fold Fs* scalar logical op loads because there are no scalar
    // load variants for these instructions. When folded, the load is required
    // to be 128-bits, so the load size would not match.

    { X86::FvANDNPDrr,      X86::FvANDNPDrm,    TB_ALIGN_16 },
    { X86::FvANDNPSrr,      X86::FvANDNPSrm,    TB_ALIGN_16 },
    { X86::FvANDPDrr,       X86::FvANDPDrm,     TB_ALIGN_16 },
    { X86::FvANDPSrr,       X86::FvANDPSrm,     TB_ALIGN_16 },
    { X86::FvORPDrr,        X86::FvORPDrm,      TB_ALIGN_16 },
    { X86::FvORPSrr,        X86::FvORPSrm,      TB_ALIGN_16 },
    { X86::FvXORPDrr,       X86::FvXORPDrm,     TB_ALIGN_16 },
    { X86::FvXORPSrr,       X86::FvXORPSrm,     TB_ALIGN_16 },
    { X86::HADDPDrr,        X86::HADDPDrm,      TB_ALIGN_16 },
    { X86::HADDPSrr,        X86::HADDPSrm,      TB_ALIGN_16 },
    { X86::HSUBPDrr,        X86::HSUBPDrm,      TB_ALIGN_16 },
    { X86::HSUBPSrr,        X86::HSUBPSrm,      TB_ALIGN_16 },
    { X86::IMUL16rr,        X86::IMUL16rm,      0 },
    { X86::IMUL32rr,        X86::IMUL32rm,      0 },
    { X86::IMUL64rr,        X86::IMUL64rm,      0 },
    { X86::Int_CMPSDrr,     X86::Int_CMPSDrm,   0 },
    { X86::Int_CMPSSrr,     X86::Int_CMPSSrm,   0 },
    { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm,      0 },
    { X86::Int_CVTSI2SD64rr,X86::Int_CVTSI2SD64rm,    0 },
    { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm,      0 },
    { X86::Int_CVTSI2SS64rr,X86::Int_CVTSI2SS64rm,    0 },
    { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm,      0 },
    { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm,      0 },
    { X86::MAXPDrr,         X86::MAXPDrm,       TB_ALIGN_16 },
    { X86::MAXPSrr,         X86::MAXPSrm,       TB_ALIGN_16 },
    { X86::MAXSDrr,         X86::MAXSDrm,       0 },
    { X86::MAXSDrr_Int,     X86::MAXSDrm_Int,   0 },
    { X86::MAXSSrr,         X86::MAXSSrm,       0 },
    { X86::MAXSSrr_Int,     X86::MAXSSrm_Int,   0 },
    { X86::MINPDrr,         X86::MINPDrm,       TB_ALIGN_16 },
    { X86::MINPSrr,         X86::MINPSrm,       TB_ALIGN_16 },
    { X86::MINSDrr,         X86::MINSDrm,       0 },
    { X86::MINSDrr_Int,     X86::MINSDrm_Int,   0 },
    { X86::MINSSrr,         X86::MINSSrm,       0 },
    { X86::MINSSrr_Int,     X86::MINSSrm_Int,   0 },
    { X86::MPSADBWrri,      X86::MPSADBWrmi,    TB_ALIGN_16 },
    { X86::MULPDrr,         X86::MULPDrm,       TB_ALIGN_16 },
    { X86::MULPSrr,         X86::MULPSrm,       TB_ALIGN_16 },
    { X86::MULSDrr,         X86::MULSDrm,       0 },
    { X86::MULSDrr_Int,     X86::MULSDrm_Int,   0 },
    { X86::MULSSrr,         X86::MULSSrm,       0 },
    { X86::MULSSrr_Int,     X86::MULSSrm_Int,   0 },
    { X86::OR16rr,          X86::OR16rm,        0 },
    { X86::OR32rr,          X86::OR32rm,        0 },
    { X86::OR64rr,          X86::OR64rm,        0 },
    { X86::OR8rr,           X86::OR8rm,         0 },
    { X86::ORPDrr,          X86::ORPDrm,        TB_ALIGN_16 },
    { X86::ORPSrr,          X86::ORPSrm,        TB_ALIGN_16 },
    { X86::PACKSSDWrr,      X86::PACKSSDWrm,    TB_ALIGN_16 },
    { X86::PACKSSWBrr,      X86::PACKSSWBrm,    TB_ALIGN_16 },
    { X86::PACKUSDWrr,      X86::PACKUSDWrm,    TB_ALIGN_16 },
    { X86::PACKUSWBrr,      X86::PACKUSWBrm,    TB_ALIGN_16 },
    { X86::PADDBrr,         X86::PADDBrm,       TB_ALIGN_16 },
    { X86::PADDDrr,         X86::PADDDrm,       TB_ALIGN_16 },
    { X86::PADDQrr,         X86::PADDQrm,       TB_ALIGN_16 },
    { X86::PADDSBrr,        X86::PADDSBrm,      TB_ALIGN_16 },
    { X86::PADDSWrr,        X86::PADDSWrm,      TB_ALIGN_16 },
    { X86::PADDUSBrr,       X86::PADDUSBrm,     TB_ALIGN_16 },
    { X86::PADDUSWrr,       X86::PADDUSWrm,     TB_ALIGN_16 },
    { X86::PADDWrr,         X86::PADDWrm,       TB_ALIGN_16 },
    { X86::PALIGNR128rr,    X86::PALIGNR128rm,  TB_ALIGN_16 },
    { X86::PANDNrr,         X86::PANDNrm,       TB_ALIGN_16 },
    { X86::PANDrr,          X86::PANDrm,        TB_ALIGN_16 },
    { X86::PAVGBrr,         X86::PAVGBrm,       TB_ALIGN_16 },
    { X86::PAVGWrr,         X86::PAVGWrm,       TB_ALIGN_16 },
    { X86::PBLENDVBrr0,     X86::PBLENDVBrm0,   TB_ALIGN_16 },
    { X86::PBLENDWrri,      X86::PBLENDWrmi,    TB_ALIGN_16 },
    { X86::PCLMULQDQrr,     X86::PCLMULQDQrm,   TB_ALIGN_16 },
    { X86::PCMPEQBrr,       X86::PCMPEQBrm,     TB_ALIGN_16 },
    { X86::PCMPEQDrr,       X86::PCMPEQDrm,     TB_ALIGN_16 },
    { X86::PCMPEQQrr,       X86::PCMPEQQrm,     TB_ALIGN_16 },
    { X86::PCMPEQWrr,       X86::PCMPEQWrm,     TB_ALIGN_16 },
    { X86::PCMPGTBrr,       X86::PCMPGTBrm,     TB_ALIGN_16 },
    { X86::PCMPGTDrr,       X86::PCMPGTDrm,     TB_ALIGN_16 },
    { X86::PCMPGTQrr,       X86::PCMPGTQrm,     TB_ALIGN_16 },
    { X86::PCMPGTWrr,       X86::PCMPGTWrm,     TB_ALIGN_16 },
    { X86::PHADDDrr,        X86::PHADDDrm,      TB_ALIGN_16 },
    { X86::PHADDWrr,        X86::PHADDWrm,      TB_ALIGN_16 },
    { X86::PHADDSWrr128,    X86::PHADDSWrm128,  TB_ALIGN_16 },
    { X86::PHSUBDrr,        X86::PHSUBDrm,      TB_ALIGN_16 },
    { X86::PHSUBSWrr128,    X86::PHSUBSWrm128,  TB_ALIGN_16 },
    { X86::PHSUBWrr,        X86::PHSUBWrm,      TB_ALIGN_16 },
    { X86::PINSRBrr,        X86::PINSRBrm,      0 },
    { X86::PINSRDrr,        X86::PINSRDrm,      0 },
    { X86::PINSRQrr,        X86::PINSRQrm,      0 },
    { X86::PINSRWrri,       X86::PINSRWrmi,     0 },
    { X86::PMADDUBSWrr128,  X86::PMADDUBSWrm128, TB_ALIGN_16 },
    { X86::PMADDWDrr,       X86::PMADDWDrm,     TB_ALIGN_16 },
    { X86::PMAXSWrr,        X86::PMAXSWrm,      TB_ALIGN_16 },
    { X86::PMAXUBrr,        X86::PMAXUBrm,      TB_ALIGN_16 },
    { X86::PMINSWrr,        X86::PMINSWrm,      TB_ALIGN_16 },
    { X86::PMINUBrr,        X86::PMINUBrm,      TB_ALIGN_16 },
    { X86::PMINSBrr,        X86::PMINSBrm,      TB_ALIGN_16 },
    { X86::PMINSDrr,        X86::PMINSDrm,      TB_ALIGN_16 },
    { X86::PMINUDrr,        X86::PMINUDrm,      TB_ALIGN_16 },
    { X86::PMINUWrr,        X86::PMINUWrm,      TB_ALIGN_16 },
    { X86::PMAXSBrr,        X86::PMAXSBrm,      TB_ALIGN_16 },
    { X86::PMAXSDrr,        X86::PMAXSDrm,      TB_ALIGN_16 },
    { X86::PMAXUDrr,        X86::PMAXUDrm,      TB_ALIGN_16 },
    { X86::PMAXUWrr,        X86::PMAXUWrm,      TB_ALIGN_16 },
    { X86::PMULDQrr,        X86::PMULDQrm,      TB_ALIGN_16 },
    { X86::PMULHRSWrr128,   X86::PMULHRSWrm128, TB_ALIGN_16 },
    { X86::PMULHUWrr,       X86::PMULHUWrm,     TB_ALIGN_16 },
    { X86::PMULHWrr,        X86::PMULHWrm,      TB_ALIGN_16 },
    { X86::PMULLDrr,        X86::PMULLDrm,      TB_ALIGN_16 },
    { X86::PMULLWrr,        X86::PMULLWrm,      TB_ALIGN_16 },
    { X86::PMULUDQrr,       X86::PMULUDQrm,     TB_ALIGN_16 },
    { X86::PORrr,           X86::PORrm,         TB_ALIGN_16 },
    { X86::PSADBWrr,        X86::PSADBWrm,      TB_ALIGN_16 },
    { X86::PSHUFBrr,        X86::PSHUFBrm,      TB_ALIGN_16 },
    { X86::PSIGNBrr,        X86::PSIGNBrm,      TB_ALIGN_16 },
    { X86::PSIGNWrr,        X86::PSIGNWrm,      TB_ALIGN_16 },
    { X86::PSIGNDrr,        X86::PSIGNDrm,      TB_ALIGN_16 },
    { X86::PSLLDrr,         X86::PSLLDrm,       TB_ALIGN_16 },
    { X86::PSLLQrr,         X86::PSLLQrm,       TB_ALIGN_16 },
    { X86::PSLLWrr,         X86::PSLLWrm,       TB_ALIGN_16 },
    { X86::PSRADrr,         X86::PSRADrm,       TB_ALIGN_16 },
    { X86::PSRAWrr,         X86::PSRAWrm,       TB_ALIGN_16 },
    { X86::PSRLDrr,         X86::PSRLDrm,       TB_ALIGN_16 },
    { X86::PSRLQrr,         X86::PSRLQrm,       TB_ALIGN_16 },
    { X86::PSRLWrr,         X86::PSRLWrm,       TB_ALIGN_16 },
    { X86::PSUBBrr,         X86::PSUBBrm,       TB_ALIGN_16 },
    { X86::PSUBDrr,         X86::PSUBDrm,       TB_ALIGN_16 },
    { X86::PSUBQrr,         X86::PSUBQrm,       TB_ALIGN_16 },
    { X86::PSUBSBrr,        X86::PSUBSBrm,      TB_ALIGN_16 },
    { X86::PSUBSWrr,        X86::PSUBSWrm,      TB_ALIGN_16 },
    { X86::PSUBUSBrr,       X86::PSUBUSBrm,     TB_ALIGN_16 },
    { X86::PSUBUSWrr,       X86::PSUBUSWrm,     TB_ALIGN_16 },
    { X86::PSUBWrr,         X86::PSUBWrm,       TB_ALIGN_16 },
    { X86::PUNPCKHBWrr,     X86::PUNPCKHBWrm,   TB_ALIGN_16 },
    { X86::PUNPCKHDQrr,     X86::PUNPCKHDQrm,   TB_ALIGN_16 },
    { X86::PUNPCKHQDQrr,    X86::PUNPCKHQDQrm,  TB_ALIGN_16 },
    { X86::PUNPCKHWDrr,     X86::PUNPCKHWDrm,   TB_ALIGN_16 },
    { X86::PUNPCKLBWrr,     X86::PUNPCKLBWrm,   TB_ALIGN_16 },
    { X86::PUNPCKLDQrr,     X86::PUNPCKLDQrm,   TB_ALIGN_16 },
    { X86::PUNPCKLQDQrr,    X86::PUNPCKLQDQrm,  TB_ALIGN_16 },
    { X86::PUNPCKLWDrr,     X86::PUNPCKLWDrm,   TB_ALIGN_16 },
    { X86::PXORrr,          X86::PXORrm,        TB_ALIGN_16 },
    { X86::ROUNDSDr,        X86::ROUNDSDm,      0 },
    { X86::ROUNDSSr,        X86::ROUNDSSm,      0 },
    { X86::SBB32rr,         X86::SBB32rm,       0 },
    { X86::SBB64rr,         X86::SBB64rm,       0 },
    { X86::SHUFPDrri,       X86::SHUFPDrmi,     TB_ALIGN_16 },
    { X86::SHUFPSrri,       X86::SHUFPSrmi,     TB_ALIGN_16 },
    { X86::SUB16rr,         X86::SUB16rm,       0 },
    { X86::SUB32rr,         X86::SUB32rm,       0 },
    { X86::SUB64rr,         X86::SUB64rm,       0 },
    { X86::SUB8rr,          X86::SUB8rm,        0 },
    { X86::SUBPDrr,         X86::SUBPDrm,       TB_ALIGN_16 },
    { X86::SUBPSrr,         X86::SUBPSrm,       TB_ALIGN_16 },
    { X86::SUBSDrr,         X86::SUBSDrm,       0 },
    { X86::SUBSDrr_Int,     X86::SUBSDrm_Int,   0 },
    { X86::SUBSSrr,         X86::SUBSSrm,       0 },
    { X86::SUBSSrr_Int,     X86::SUBSSrm_Int,   0 },
    // FIXME: TEST*rr -> swapped operand of TEST*mr.
    { X86::UNPCKHPDrr,      X86::UNPCKHPDrm,    TB_ALIGN_16 },
    { X86::UNPCKHPSrr,      X86::UNPCKHPSrm,    TB_ALIGN_16 },
    { X86::UNPCKLPDrr,      X86::UNPCKLPDrm,    TB_ALIGN_16 },
    { X86::UNPCKLPSrr,      X86::UNPCKLPSrm,    TB_ALIGN_16 },
    { X86::XOR16rr,         X86::XOR16rm,       0 },
    { X86::XOR32rr,         X86::XOR32rm,       0 },
    { X86::XOR64rr,         X86::XOR64rm,       0 },
    { X86::XOR8rr,          X86::XOR8rm,        0 },
    { X86::XORPDrr,         X86::XORPDrm,       TB_ALIGN_16 },
    { X86::XORPSrr,         X86::XORPSrm,       TB_ALIGN_16 },

    // MMX version of foldable instructions
    { X86::MMX_CVTPI2PSirr,   X86::MMX_CVTPI2PSirm,   0 },
    { X86::MMX_PACKSSDWirr,   X86::MMX_PACKSSDWirm,   0 },
    { X86::MMX_PACKSSWBirr,   X86::MMX_PACKSSWBirm,   0 },
    { X86::MMX_PACKUSWBirr,   X86::MMX_PACKUSWBirm,   0 },
    { X86::MMX_PADDBirr,      X86::MMX_PADDBirm,      0 },
    { X86::MMX_PADDDirr,      X86::MMX_PADDDirm,      0 },
    { X86::MMX_PADDQirr,      X86::MMX_PADDQirm,      0 },
    { X86::MMX_PADDSBirr,     X86::MMX_PADDSBirm,     0 },
    { X86::MMX_PADDSWirr,     X86::MMX_PADDSWirm,     0 },
    { X86::MMX_PADDUSBirr,    X86::MMX_PADDUSBirm,    0 },
    { X86::MMX_PADDUSWirr,    X86::MMX_PADDUSWirm,    0 },
    { X86::MMX_PADDWirr,      X86::MMX_PADDWirm,      0 },
    { X86::MMX_PALIGNR64irr,  X86::MMX_PALIGNR64irm,  0 },
    { X86::MMX_PANDNirr,      X86::MMX_PANDNirm,      0 },
    { X86::MMX_PANDirr,       X86::MMX_PANDirm,       0 },
    { X86::MMX_PAVGBirr,      X86::MMX_PAVGBirm,      0 },
    { X86::MMX_PAVGWirr,      X86::MMX_PAVGWirm,      0 },
    { X86::MMX_PCMPEQBirr,    X86::MMX_PCMPEQBirm,    0 },
    { X86::MMX_PCMPEQDirr,    X86::MMX_PCMPEQDirm,    0 },
    { X86::MMX_PCMPEQWirr,    X86::MMX_PCMPEQWirm,    0 },
    { X86::MMX_PCMPGTBirr,    X86::MMX_PCMPGTBirm,    0 },
    { X86::MMX_PCMPGTDirr,    X86::MMX_PCMPGTDirm,    0 },
    { X86::MMX_PCMPGTWirr,    X86::MMX_PCMPGTWirm,    0 },
    { X86::MMX_PHADDSWrr64,   X86::MMX_PHADDSWrm64,   0 },
    { X86::MMX_PHADDWrr64,    X86::MMX_PHADDWrm64,    0 },
    { X86::MMX_PHADDrr64,     X86::MMX_PHADDrm64,     0 },
    { X86::MMX_PHSUBDrr64,    X86::MMX_PHSUBDrm64,    0 },
    { X86::MMX_PHSUBSWrr64,   X86::MMX_PHSUBSWrm64,   0 },
    { X86::MMX_PHSUBWrr64,    X86::MMX_PHSUBWrm64,    0 },
    { X86::MMX_PINSRWirri,    X86::MMX_PINSRWirmi,    0 },
    { X86::MMX_PMADDUBSWrr64, X86::MMX_PMADDUBSWrm64, 0 },
    { X86::MMX_PMADDWDirr,    X86::MMX_PMADDWDirm,    0 },
    { X86::MMX_PMAXSWirr,     X86::MMX_PMAXSWirm,     0 },
    { X86::MMX_PMAXUBirr,     X86::MMX_PMAXUBirm,     0 },
    { X86::MMX_PMINSWirr,     X86::MMX_PMINSWirm,     0 },
    { X86::MMX_PMINUBirr,     X86::MMX_PMINUBirm,     0 },
    { X86::MMX_PMULHRSWrr64,  X86::MMX_PMULHRSWrm64,  0 },
    { X86::MMX_PMULHUWirr,    X86::MMX_PMULHUWirm,    0 },
    { X86::MMX_PMULHWirr,     X86::MMX_PMULHWirm,     0 },
    { X86::MMX_PMULLWirr,     X86::MMX_PMULLWirm,     0 },
    { X86::MMX_PMULUDQirr,    X86::MMX_PMULUDQirm,    0 },
    { X86::MMX_PORirr,        X86::MMX_PORirm,        0 },
    { X86::MMX_PSADBWirr,     X86::MMX_PSADBWirm,     0 },
    { X86::MMX_PSHUFBrr64,    X86::MMX_PSHUFBrm64,    0 },
    { X86::MMX_PSIGNBrr64,    X86::MMX_PSIGNBrm64,    0 },
    { X86::MMX_PSIGNDrr64,    X86::MMX_PSIGNDrm64,    0 },
    { X86::MMX_PSIGNWrr64,    X86::MMX_PSIGNWrm64,    0 },
    { X86::MMX_PSLLDrr,       X86::MMX_PSLLDrm,       0 },
    { X86::MMX_PSLLQrr,       X86::MMX_PSLLQrm,       0 },
    { X86::MMX_PSLLWrr,       X86::MMX_PSLLWrm,       0 },
    { X86::MMX_PSRADrr,       X86::MMX_PSRADrm,       0 },
    { X86::MMX_PSRAWrr,       X86::MMX_PSRAWrm,       0 },
    { X86::MMX_PSRLDrr,       X86::MMX_PSRLDrm,       0 },
    { X86::MMX_PSRLQrr,       X86::MMX_PSRLQrm,       0 },
    { X86::MMX_PSRLWrr,       X86::MMX_PSRLWrm,       0 },
    { X86::MMX_PSUBBirr,      X86::MMX_PSUBBirm,      0 },
    { X86::MMX_PSUBDirr,      X86::MMX_PSUBDirm,      0 },
    { X86::MMX_PSUBQirr,      X86::MMX_PSUBQirm,      0 },
    { X86::MMX_PSUBSBirr,     X86::MMX_PSUBSBirm,     0 },
    { X86::MMX_PSUBSWirr,     X86::MMX_PSUBSWirm,     0 },
    { X86::MMX_PSUBUSBirr,    X86::MMX_PSUBUSBirm,    0 },
    { X86::MMX_PSUBUSWirr,    X86::MMX_PSUBUSWirm,    0 },
    { X86::MMX_PSUBWirr,      X86::MMX_PSUBWirm,      0 },
    { X86::MMX_PUNPCKHBWirr,  X86::MMX_PUNPCKHBWirm,  0 },
    { X86::MMX_PUNPCKHDQirr,  X86::MMX_PUNPCKHDQirm,  0 },
    { X86::MMX_PUNPCKHWDirr,  X86::MMX_PUNPCKHWDirm,  0 },
    { X86::MMX_PUNPCKLBWirr,  X86::MMX_PUNPCKLBWirm,  0 },
    { X86::MMX_PUNPCKLDQirr,  X86::MMX_PUNPCKLDQirm,  0 },
    { X86::MMX_PUNPCKLWDirr,  X86::MMX_PUNPCKLWDirm,  0 },
    { X86::MMX_PXORirr,       X86::MMX_PXORirm,       0 },

    // 3DNow! version of foldable instructions
    { X86::PAVGUSBrr,         X86::PAVGUSBrm,         0 },
    { X86::PFACCrr,           X86::PFACCrm,           0 },
    { X86::PFADDrr,           X86::PFADDrm,           0 },
    { X86::PFCMPEQrr,         X86::PFCMPEQrm,         0 },
    { X86::PFCMPGErr,         X86::PFCMPGErm,         0 },
    { X86::PFCMPGTrr,         X86::PFCMPGTrm,         0 },
    { X86::PFMAXrr,           X86::PFMAXrm,           0 },
    { X86::PFMINrr,           X86::PFMINrm,           0 },
    { X86::PFMULrr,           X86::PFMULrm,           0 },
    { X86::PFNACCrr,          X86::PFNACCrm,          0 },
    { X86::PFPNACCrr,         X86::PFPNACCrm,         0 },
    { X86::PFRCPIT1rr,        X86::PFRCPIT1rm,        0 },
    { X86::PFRCPIT2rr,        X86::PFRCPIT2rm,        0 },
    { X86::PFRSQIT1rr,        X86::PFRSQIT1rm,        0 },
    { X86::PFSUBrr,           X86::PFSUBrm,           0 },
    { X86::PFSUBRrr,          X86::PFSUBRrm,          0 },
    { X86::PMULHRWrr,         X86::PMULHRWrm,         0 },

    // AVX 128-bit versions of foldable instructions
    { X86::VCVTSD2SSrr,       X86::VCVTSD2SSrm,        0 },
    { X86::Int_VCVTSD2SSrr,   X86::Int_VCVTSD2SSrm,    0 },
    { X86::VCVTSI2SD64rr,     X86::VCVTSI2SD64rm,      0 },
    { X86::Int_VCVTSI2SD64rr, X86::Int_VCVTSI2SD64rm,  0 },
    { X86::VCVTSI2SDrr,       X86::VCVTSI2SDrm,        0 },
    { X86::Int_VCVTSI2SDrr,   X86::Int_VCVTSI2SDrm,    0 },
    { X86::VCVTSI2SS64rr,     X86::VCVTSI2SS64rm,      0 },
    { X86::Int_VCVTSI2SS64rr, X86::Int_VCVTSI2SS64rm,  0 },
    { X86::VCVTSI2SSrr,       X86::VCVTSI2SSrm,        0 },
    { X86::Int_VCVTSI2SSrr,   X86::Int_VCVTSI2SSrm,    0 },
    { X86::VCVTSS2SDrr,       X86::VCVTSS2SDrm,        0 },
    { X86::Int_VCVTSS2SDrr,   X86::Int_VCVTSS2SDrm,    0 },
    { X86::VRCPSSr,           X86::VRCPSSm,            0 },
    { X86::VRCPSSr_Int,       X86::VRCPSSm_Int,        0 },
    { X86::VRSQRTSSr,         X86::VRSQRTSSm,          0 },
    { X86::VRSQRTSSr_Int,     X86::VRSQRTSSm_Int,      0 },
    { X86::VSQRTSDr,          X86::VSQRTSDm,           0 },
    { X86::VSQRTSDr_Int,      X86::VSQRTSDm_Int,       0 },
    { X86::VSQRTSSr,          X86::VSQRTSSm,           0 },
    { X86::VSQRTSSr_Int,      X86::VSQRTSSm_Int,       0 },
    { X86::VADDPDrr,          X86::VADDPDrm,           0 },
    { X86::VADDPSrr,          X86::VADDPSrm,           0 },
    { X86::VADDSDrr,          X86::VADDSDrm,           0 },
    { X86::VADDSDrr_Int,      X86::VADDSDrm_Int,       0 },
    { X86::VADDSSrr,          X86::VADDSSrm,           0 },
    { X86::VADDSSrr_Int,      X86::VADDSSrm_Int,       0 },
    { X86::VADDSUBPDrr,       X86::VADDSUBPDrm,        0 },
    { X86::VADDSUBPSrr,       X86::VADDSUBPSrm,        0 },
    { X86::VANDNPDrr,         X86::VANDNPDrm,          0 },
    { X86::VANDNPSrr,         X86::VANDNPSrm,          0 },
    { X86::VANDPDrr,          X86::VANDPDrm,           0 },
    { X86::VANDPSrr,          X86::VANDPSrm,           0 },
    { X86::VBLENDPDrri,       X86::VBLENDPDrmi,        0 },
    { X86::VBLENDPSrri,       X86::VBLENDPSrmi,        0 },
    { X86::VBLENDVPDrr,       X86::VBLENDVPDrm,        0 },
    { X86::VBLENDVPSrr,       X86::VBLENDVPSrm,        0 },
    { X86::VCMPPDrri,         X86::VCMPPDrmi,          0 },
    { X86::VCMPPSrri,         X86::VCMPPSrmi,          0 },
    { X86::VCMPSDrr,          X86::VCMPSDrm,           0 },
    { X86::VCMPSSrr,          X86::VCMPSSrm,           0 },
    { X86::VDIVPDrr,          X86::VDIVPDrm,           0 },
    { X86::VDIVPSrr,          X86::VDIVPSrm,           0 },
    { X86::VDIVSDrr,          X86::VDIVSDrm,           0 },
    { X86::VDIVSDrr_Int,      X86::VDIVSDrm_Int,       0 },
    { X86::VDIVSSrr,          X86::VDIVSSrm,           0 },
    { X86::VDIVSSrr_Int,      X86::VDIVSSrm_Int,       0 },
    { X86::VDPPDrri,          X86::VDPPDrmi,           0 },
    { X86::VDPPSrri,          X86::VDPPSrmi,           0 },
    // Do not fold VFs* loads because there are no scalar load variants for
    // these instructions. When folded, the load is required to be 128-bits, so
    // the load size would not match.
    { X86::VFvANDNPDrr,       X86::VFvANDNPDrm,        0 },
    { X86::VFvANDNPSrr,       X86::VFvANDNPSrm,        0 },
    { X86::VFvANDPDrr,        X86::VFvANDPDrm,         0 },
    { X86::VFvANDPSrr,        X86::VFvANDPSrm,         0 },
    { X86::VFvORPDrr,         X86::VFvORPDrm,          0 },
    { X86::VFvORPSrr,         X86::VFvORPSrm,          0 },
    { X86::VFvXORPDrr,        X86::VFvXORPDrm,         0 },
    { X86::VFvXORPSrr,        X86::VFvXORPSrm,         0 },
    { X86::VHADDPDrr,         X86::VHADDPDrm,          0 },
    { X86::VHADDPSrr,         X86::VHADDPSrm,          0 },
    { X86::VHSUBPDrr,         X86::VHSUBPDrm,          0 },
    { X86::VHSUBPSrr,         X86::VHSUBPSrm,          0 },
    { X86::Int_VCMPSDrr,      X86::Int_VCMPSDrm,       0 },
    { X86::Int_VCMPSSrr,      X86::Int_VCMPSSrm,       0 },
    { X86::VMAXPDrr,          X86::VMAXPDrm,           0 },
    { X86::VMAXPSrr,          X86::VMAXPSrm,           0 },
    { X86::VMAXSDrr,          X86::VMAXSDrm,           0 },
    { X86::VMAXSDrr_Int,      X86::VMAXSDrm_Int,       0 },
    { X86::VMAXSSrr,          X86::VMAXSSrm,           0 },
    { X86::VMAXSSrr_Int,      X86::VMAXSSrm_Int,       0 },
    { X86::VMINPDrr,          X86::VMINPDrm,           0 },
    { X86::VMINPSrr,          X86::VMINPSrm,           0 },
    { X86::VMINSDrr,          X86::VMINSDrm,           0 },
    { X86::VMINSDrr_Int,      X86::VMINSDrm_Int,       0 },
    { X86::VMINSSrr,          X86::VMINSSrm,           0 },
    { X86::VMINSSrr_Int,      X86::VMINSSrm_Int,       0 },
    { X86::VMPSADBWrri,       X86::VMPSADBWrmi,        0 },
    { X86::VMULPDrr,          X86::VMULPDrm,           0 },
    { X86::VMULPSrr,          X86::VMULPSrm,           0 },
    { X86::VMULSDrr,          X86::VMULSDrm,           0 },
    { X86::VMULSDrr_Int,      X86::VMULSDrm_Int,       0 },
    { X86::VMULSSrr,          X86::VMULSSrm,           0 },
    { X86::VMULSSrr_Int,      X86::VMULSSrm_Int,       0 },
    { X86::VORPDrr,           X86::VORPDrm,            0 },
    { X86::VORPSrr,           X86::VORPSrm,            0 },
    { X86::VPACKSSDWrr,       X86::VPACKSSDWrm,        0 },
    { X86::VPACKSSWBrr,       X86::VPACKSSWBrm,        0 },
    { X86::VPACKUSDWrr,       X86::VPACKUSDWrm,        0 },
    { X86::VPACKUSWBrr,       X86::VPACKUSWBrm,        0 },
    { X86::VPADDBrr,          X86::VPADDBrm,           0 },
    { X86::VPADDDrr,          X86::VPADDDrm,           0 },
    { X86::VPADDQrr,          X86::VPADDQrm,           0 },
    { X86::VPADDSBrr,         X86::VPADDSBrm,          0 },
    { X86::VPADDSWrr,         X86::VPADDSWrm,          0 },
    { X86::VPADDUSBrr,        X86::VPADDUSBrm,         0 },
    { X86::VPADDUSWrr,        X86::VPADDUSWrm,         0 },
    { X86::VPADDWrr,          X86::VPADDWrm,           0 },
    { X86::VPALIGNR128rr,     X86::VPALIGNR128rm,      0 },
    { X86::VPANDNrr,          X86::VPANDNrm,           0 },
    { X86::VPANDrr,           X86::VPANDrm,            0 },
    { X86::VPAVGBrr,          X86::VPAVGBrm,           0 },
    { X86::VPAVGWrr,          X86::VPAVGWrm,           0 },
    { X86::VPBLENDVBrr,       X86::VPBLENDVBrm,        0 },
    { X86::VPBLENDWrri,       X86::VPBLENDWrmi,        0 },
    { X86::VPCLMULQDQrr,      X86::VPCLMULQDQrm,       0 },
    { X86::VPCMPEQBrr,        X86::VPCMPEQBrm,         0 },
    { X86::VPCMPEQDrr,        X86::VPCMPEQDrm,         0 },
    { X86::VPCMPEQQrr,        X86::VPCMPEQQrm,         0 },
    { X86::VPCMPEQWrr,        X86::VPCMPEQWrm,         0 },
    { X86::VPCMPGTBrr,        X86::VPCMPGTBrm,         0 },
    { X86::VPCMPGTDrr,        X86::VPCMPGTDrm,         0 },
    { X86::VPCMPGTQrr,        X86::VPCMPGTQrm,         0 },
    { X86::VPCMPGTWrr,        X86::VPCMPGTWrm,         0 },
    { X86::VPHADDDrr,         X86::VPHADDDrm,          0 },
    { X86::VPHADDSWrr128,     X86::VPHADDSWrm128,      0 },
    { X86::VPHADDWrr,         X86::VPHADDWrm,          0 },
    { X86::VPHSUBDrr,         X86::VPHSUBDrm,          0 },
    { X86::VPHSUBSWrr128,     X86::VPHSUBSWrm128,      0 },
    { X86::VPHSUBWrr,         X86::VPHSUBWrm,          0 },
    { X86::VPERMILPDrr,       X86::VPERMILPDrm,        0 },
    { X86::VPERMILPSrr,       X86::VPERMILPSrm,        0 },
    { X86::VPINSRBrr,         X86::VPINSRBrm,          0 },
    { X86::VPINSRDrr,         X86::VPINSRDrm,          0 },
    { X86::VPINSRQrr,         X86::VPINSRQrm,          0 },
    { X86::VPINSRWrri,        X86::VPINSRWrmi,         0 },
    { X86::VPMADDUBSWrr128,   X86::VPMADDUBSWrm128,    0 },
    { X86::VPMADDWDrr,        X86::VPMADDWDrm,         0 },
    { X86::VPMAXSWrr,         X86::VPMAXSWrm,          0 },
    { X86::VPMAXUBrr,         X86::VPMAXUBrm,          0 },
    { X86::VPMINSWrr,         X86::VPMINSWrm,          0 },
    { X86::VPMINUBrr,         X86::VPMINUBrm,          0 },
    { X86::VPMINSBrr,         X86::VPMINSBrm,          0 },
    { X86::VPMINSDrr,         X86::VPMINSDrm,          0 },
    { X86::VPMINUDrr,         X86::VPMINUDrm,          0 },
    { X86::VPMINUWrr,         X86::VPMINUWrm,          0 },
    { X86::VPMAXSBrr,         X86::VPMAXSBrm,          0 },
    { X86::VPMAXSDrr,         X86::VPMAXSDrm,          0 },
    { X86::VPMAXUDrr,         X86::VPMAXUDrm,          0 },
    { X86::VPMAXUWrr,         X86::VPMAXUWrm,          0 },
    { X86::VPMULDQrr,         X86::VPMULDQrm,          0 },
    { X86::VPMULHRSWrr128,    X86::VPMULHRSWrm128,     0 },
    { X86::VPMULHUWrr,        X86::VPMULHUWrm,         0 },
    { X86::VPMULHWrr,         X86::VPMULHWrm,          0 },
    { X86::VPMULLDrr,         X86::VPMULLDrm,          0 },
    { X86::VPMULLWrr,         X86::VPMULLWrm,          0 },
    { X86::VPMULUDQrr,        X86::VPMULUDQrm,         0 },
    { X86::VPORrr,            X86::VPORrm,             0 },
    { X86::VPSADBWrr,         X86::VPSADBWrm,          0 },
    { X86::VPSHUFBrr,         X86::VPSHUFBrm,          0 },
    { X86::VPSIGNBrr,         X86::VPSIGNBrm,          0 },
    { X86::VPSIGNWrr,         X86::VPSIGNWrm,          0 },
    { X86::VPSIGNDrr,         X86::VPSIGNDrm,          0 },
    { X86::VPSLLDrr,          X86::VPSLLDrm,           0 },
    { X86::VPSLLQrr,          X86::VPSLLQrm,           0 },
    { X86::VPSLLWrr,          X86::VPSLLWrm,           0 },
    { X86::VPSRADrr,          X86::VPSRADrm,           0 },
    { X86::VPSRAWrr,          X86::VPSRAWrm,           0 },
    { X86::VPSRLDrr,          X86::VPSRLDrm,           0 },
    { X86::VPSRLQrr,          X86::VPSRLQrm,           0 },
    { X86::VPSRLWrr,          X86::VPSRLWrm,           0 },
    { X86::VPSUBBrr,          X86::VPSUBBrm,           0 },
    { X86::VPSUBDrr,          X86::VPSUBDrm,           0 },
    { X86::VPSUBQrr,          X86::VPSUBQrm,           0 },
    { X86::VPSUBSBrr,         X86::VPSUBSBrm,          0 },
    { X86::VPSUBSWrr,         X86::VPSUBSWrm,          0 },
    { X86::VPSUBUSBrr,        X86::VPSUBUSBrm,         0 },
    { X86::VPSUBUSWrr,        X86::VPSUBUSWrm,         0 },
    { X86::VPSUBWrr,          X86::VPSUBWrm,           0 },
    { X86::VPUNPCKHBWrr,      X86::VPUNPCKHBWrm,       0 },
    { X86::VPUNPCKHDQrr,      X86::VPUNPCKHDQrm,       0 },
    { X86::VPUNPCKHQDQrr,     X86::VPUNPCKHQDQrm,      0 },
    { X86::VPUNPCKHWDrr,      X86::VPUNPCKHWDrm,       0 },
    { X86::VPUNPCKLBWrr,      X86::VPUNPCKLBWrm,       0 },
    { X86::VPUNPCKLDQrr,      X86::VPUNPCKLDQrm,       0 },
    { X86::VPUNPCKLQDQrr,     X86::VPUNPCKLQDQrm,      0 },
    { X86::VPUNPCKLWDrr,      X86::VPUNPCKLWDrm,       0 },
    { X86::VPXORrr,           X86::VPXORrm,            0 },
    { X86::VROUNDSDr,         X86::VROUNDSDm,          0 },
    { X86::VROUNDSSr,         X86::VROUNDSSm,          0 },
    { X86::VSHUFPDrri,        X86::VSHUFPDrmi,         0 },
    { X86::VSHUFPSrri,        X86::VSHUFPSrmi,         0 },
    { X86::VSUBPDrr,          X86::VSUBPDrm,           0 },
    { X86::VSUBPSrr,          X86::VSUBPSrm,           0 },
    { X86::VSUBSDrr,          X86::VSUBSDrm,           0 },
    { X86::VSUBSDrr_Int,      X86::VSUBSDrm_Int,       0 },
    { X86::VSUBSSrr,          X86::VSUBSSrm,           0 },
    { X86::VSUBSSrr_Int,      X86::VSUBSSrm_Int,       0 },
    { X86::VUNPCKHPDrr,       X86::VUNPCKHPDrm,        0 },
    { X86::VUNPCKHPSrr,       X86::VUNPCKHPSrm,        0 },
    { X86::VUNPCKLPDrr,       X86::VUNPCKLPDrm,        0 },
    { X86::VUNPCKLPSrr,       X86::VUNPCKLPSrm,        0 },
    { X86::VXORPDrr,          X86::VXORPDrm,           0 },
    { X86::VXORPSrr,          X86::VXORPSrm,           0 },

    // AVX 256-bit foldable instructions
    { X86::VADDPDYrr,         X86::VADDPDYrm,          0 },
    { X86::VADDPSYrr,         X86::VADDPSYrm,          0 },
    { X86::VADDSUBPDYrr,      X86::VADDSUBPDYrm,       0 },
    { X86::VADDSUBPSYrr,      X86::VADDSUBPSYrm,       0 },
    { X86::VANDNPDYrr,        X86::VANDNPDYrm,         0 },
    { X86::VANDNPSYrr,        X86::VANDNPSYrm,         0 },
    { X86::VANDPDYrr,         X86::VANDPDYrm,          0 },
    { X86::VANDPSYrr,         X86::VANDPSYrm,          0 },
    { X86::VBLENDPDYrri,      X86::VBLENDPDYrmi,       0 },
    { X86::VBLENDPSYrri,      X86::VBLENDPSYrmi,       0 },
    { X86::VBLENDVPDYrr,      X86::VBLENDVPDYrm,       0 },
    { X86::VBLENDVPSYrr,      X86::VBLENDVPSYrm,       0 },
    { X86::VCMPPDYrri,        X86::VCMPPDYrmi,         0 },
    { X86::VCMPPSYrri,        X86::VCMPPSYrmi,         0 },
    { X86::VDIVPDYrr,         X86::VDIVPDYrm,          0 },
    { X86::VDIVPSYrr,         X86::VDIVPSYrm,          0 },
    { X86::VDPPSYrri,         X86::VDPPSYrmi,          0 },
    { X86::VHADDPDYrr,        X86::VHADDPDYrm,         0 },
    { X86::VHADDPSYrr,        X86::VHADDPSYrm,         0 },
    { X86::VHSUBPDYrr,        X86::VHSUBPDYrm,         0 },
    { X86::VHSUBPSYrr,        X86::VHSUBPSYrm,         0 },
    { X86::VINSERTF128rr,     X86::VINSERTF128rm,      0 },
    { X86::VMAXPDYrr,         X86::VMAXPDYrm,          0 },
    { X86::VMAXPSYrr,         X86::VMAXPSYrm,          0 },
    { X86::VMINPDYrr,         X86::VMINPDYrm,          0 },
    { X86::VMINPSYrr,         X86::VMINPSYrm,          0 },
    { X86::VMULPDYrr,         X86::VMULPDYrm,          0 },
    { X86::VMULPSYrr,         X86::VMULPSYrm,          0 },
    { X86::VORPDYrr,          X86::VORPDYrm,           0 },
    { X86::VORPSYrr,          X86::VORPSYrm,           0 },
    { X86::VPERM2F128rr,      X86::VPERM2F128rm,       0 },
    { X86::VPERMILPDYrr,      X86::VPERMILPDYrm,       0 },
    { X86::VPERMILPSYrr,      X86::VPERMILPSYrm,       0 },
    { X86::VSHUFPDYrri,       X86::VSHUFPDYrmi,        0 },
    { X86::VSHUFPSYrri,       X86::VSHUFPSYrmi,        0 },
    { X86::VSUBPDYrr,         X86::VSUBPDYrm,          0 },
    { X86::VSUBPSYrr,         X86::VSUBPSYrm,          0 },
    { X86::VUNPCKHPDYrr,      X86::VUNPCKHPDYrm,       0 },
    { X86::VUNPCKHPSYrr,      X86::VUNPCKHPSYrm,       0 },
    { X86::VUNPCKLPDYrr,      X86::VUNPCKLPDYrm,       0 },
    { X86::VUNPCKLPSYrr,      X86::VUNPCKLPSYrm,       0 },
    { X86::VXORPDYrr,         X86::VXORPDYrm,          0 },
    { X86::VXORPSYrr,         X86::VXORPSYrm,          0 },

    // AVX2 foldable instructions
    { X86::VINSERTI128rr,     X86::VINSERTI128rm,      0 },
    { X86::VPACKSSDWYrr,      X86::VPACKSSDWYrm,       0 },
    { X86::VPACKSSWBYrr,      X86::VPACKSSWBYrm,       0 },
    { X86::VPACKUSDWYrr,      X86::VPACKUSDWYrm,       0 },
    { X86::VPACKUSWBYrr,      X86::VPACKUSWBYrm,       0 },
    { X86::VPADDBYrr,         X86::VPADDBYrm,          0 },
    { X86::VPADDDYrr,         X86::VPADDDYrm,          0 },
    { X86::VPADDQYrr,         X86::VPADDQYrm,          0 },
    { X86::VPADDSBYrr,        X86::VPADDSBYrm,         0 },
    { X86::VPADDSWYrr,        X86::VPADDSWYrm,         0 },
    { X86::VPADDUSBYrr,       X86::VPADDUSBYrm,        0 },
    { X86::VPADDUSWYrr,       X86::VPADDUSWYrm,        0 },
    { X86::VPADDWYrr,         X86::VPADDWYrm,          0 },
    { X86::VPALIGNR256rr,     X86::VPALIGNR256rm,      0 },
    { X86::VPANDNYrr,         X86::VPANDNYrm,          0 },
    { X86::VPANDYrr,          X86::VPANDYrm,           0 },
    { X86::VPAVGBYrr,         X86::VPAVGBYrm,          0 },
    { X86::VPAVGWYrr,         X86::VPAVGWYrm,          0 },
    { X86::VPBLENDDrri,       X86::VPBLENDDrmi,        0 },
    { X86::VPBLENDDYrri,      X86::VPBLENDDYrmi,       0 },
    { X86::VPBLENDVBYrr,      X86::VPBLENDVBYrm,       0 },
    { X86::VPBLENDWYrri,      X86::VPBLENDWYrmi,       0 },
    { X86::VPCMPEQBYrr,       X86::VPCMPEQBYrm,        0 },
    { X86::VPCMPEQDYrr,       X86::VPCMPEQDYrm,        0 },
    { X86::VPCMPEQQYrr,       X86::VPCMPEQQYrm,        0 },
    { X86::VPCMPEQWYrr,       X86::VPCMPEQWYrm,        0 },
    { X86::VPCMPGTBYrr,       X86::VPCMPGTBYrm,        0 },
    { X86::VPCMPGTDYrr,       X86::VPCMPGTDYrm,        0 },
    { X86::VPCMPGTQYrr,       X86::VPCMPGTQYrm,        0 },
    { X86::VPCMPGTWYrr,       X86::VPCMPGTWYrm,        0 },
    { X86::VPERM2I128rr,      X86::VPERM2I128rm,       0 },
    { X86::VPERMDYrr,         X86::VPERMDYrm,          0 },
    { X86::VPERMPSYrr,        X86::VPERMPSYrm,         0 },
    { X86::VPHADDDYrr,        X86::VPHADDDYrm,         0 },
    { X86::VPHADDSWrr256,     X86::VPHADDSWrm256,      0 },
    { X86::VPHADDWYrr,        X86::VPHADDWYrm,         0 },
    { X86::VPHSUBDYrr,        X86::VPHSUBDYrm,         0 },
    { X86::VPHSUBSWrr256,     X86::VPHSUBSWrm256,      0 },
    { X86::VPHSUBWYrr,        X86::VPHSUBWYrm,         0 },
    { X86::VPMADDUBSWrr256,   X86::VPMADDUBSWrm256,    0 },
    { X86::VPMADDWDYrr,       X86::VPMADDWDYrm,        0 },
    { X86::VPMAXSWYrr,        X86::VPMAXSWYrm,         0 },
    { X86::VPMAXUBYrr,        X86::VPMAXUBYrm,         0 },
    { X86::VPMINSWYrr,        X86::VPMINSWYrm,         0 },
    { X86::VPMINUBYrr,        X86::VPMINUBYrm,         0 },
    { X86::VPMINSBYrr,        X86::VPMINSBYrm,         0 },
    { X86::VPMINSDYrr,        X86::VPMINSDYrm,         0 },
    { X86::VPMINUDYrr,        X86::VPMINUDYrm,         0 },
    { X86::VPMINUWYrr,        X86::VPMINUWYrm,         0 },
    { X86::VPMAXSBYrr,        X86::VPMAXSBYrm,         0 },
    { X86::VPMAXSDYrr,        X86::VPMAXSDYrm,         0 },
    { X86::VPMAXUDYrr,        X86::VPMAXUDYrm,         0 },
    { X86::VPMAXUWYrr,        X86::VPMAXUWYrm,         0 },
    { X86::VMPSADBWYrri,      X86::VMPSADBWYrmi,       0 },
    { X86::VPMULDQYrr,        X86::VPMULDQYrm,         0 },
    { X86::VPMULHRSWrr256,    X86::VPMULHRSWrm256,     0 },
    { X86::VPMULHUWYrr,       X86::VPMULHUWYrm,        0 },
    { X86::VPMULHWYrr,        X86::VPMULHWYrm,         0 },
    { X86::VPMULLDYrr,        X86::VPMULLDYrm,         0 },
    { X86::VPMULLWYrr,        X86::VPMULLWYrm,         0 },
    { X86::VPMULUDQYrr,       X86::VPMULUDQYrm,        0 },
    { X86::VPORYrr,           X86::VPORYrm,            0 },
    { X86::VPSADBWYrr,        X86::VPSADBWYrm,         0 },
    { X86::VPSHUFBYrr,        X86::VPSHUFBYrm,         0 },
    { X86::VPSIGNBYrr,        X86::VPSIGNBYrm,         0 },
    { X86::VPSIGNWYrr,        X86::VPSIGNWYrm,         0 },
    { X86::VPSIGNDYrr,        X86::VPSIGNDYrm,         0 },
    { X86::VPSLLDYrr,         X86::VPSLLDYrm,          0 },
    { X86::VPSLLQYrr,         X86::VPSLLQYrm,          0 },
    { X86::VPSLLWYrr,         X86::VPSLLWYrm,          0 },
    { X86::VPSLLVDrr,         X86::VPSLLVDrm,          0 },
    { X86::VPSLLVDYrr,        X86::VPSLLVDYrm,         0 },
    { X86::VPSLLVQrr,         X86::VPSLLVQrm,          0 },
    { X86::VPSLLVQYrr,        X86::VPSLLVQYrm,         0 },
    { X86::VPSRADYrr,         X86::VPSRADYrm,          0 },
    { X86::VPSRAWYrr,         X86::VPSRAWYrm,          0 },
    { X86::VPSRAVDrr,         X86::VPSRAVDrm,          0 },
    { X86::VPSRAVDYrr,        X86::VPSRAVDYrm,         0 },
    { X86::VPSRLDYrr,         X86::VPSRLDYrm,          0 },
    { X86::VPSRLQYrr,         X86::VPSRLQYrm,          0 },
    { X86::VPSRLWYrr,         X86::VPSRLWYrm,          0 },
    { X86::VPSRLVDrr,         X86::VPSRLVDrm,          0 },
    { X86::VPSRLVDYrr,        X86::VPSRLVDYrm,         0 },
    { X86::VPSRLVQrr,         X86::VPSRLVQrm,          0 },
    { X86::VPSRLVQYrr,        X86::VPSRLVQYrm,         0 },
    { X86::VPSUBBYrr,         X86::VPSUBBYrm,          0 },
    { X86::VPSUBDYrr,         X86::VPSUBDYrm,          0 },
    { X86::VPSUBQYrr,         X86::VPSUBQYrm,          0 },
    { X86::VPSUBSBYrr,        X86::VPSUBSBYrm,         0 },
    { X86::VPSUBSWYrr,        X86::VPSUBSWYrm,         0 },
    { X86::VPSUBUSBYrr,       X86::VPSUBUSBYrm,        0 },
    { X86::VPSUBUSWYrr,       X86::VPSUBUSWYrm,        0 },
    { X86::VPSUBWYrr,         X86::VPSUBWYrm,          0 },
    { X86::VPUNPCKHBWYrr,     X86::VPUNPCKHBWYrm,      0 },
    { X86::VPUNPCKHDQYrr,     X86::VPUNPCKHDQYrm,      0 },
    { X86::VPUNPCKHQDQYrr,    X86::VPUNPCKHQDQYrm,     0 },
    { X86::VPUNPCKHWDYrr,     X86::VPUNPCKHWDYrm,      0 },
    { X86::VPUNPCKLBWYrr,     X86::VPUNPCKLBWYrm,      0 },
    { X86::VPUNPCKLDQYrr,     X86::VPUNPCKLDQYrm,      0 },
    { X86::VPUNPCKLQDQYrr,    X86::VPUNPCKLQDQYrm,     0 },
    { X86::VPUNPCKLWDYrr,     X86::VPUNPCKLWDYrm,      0 },
    { X86::VPXORYrr,          X86::VPXORYrm,           0 },

    // FMA4 foldable patterns
    { X86::VFMADDSS4rr,       X86::VFMADDSS4mr,        TB_ALIGN_NONE },
    { X86::VFMADDSD4rr,       X86::VFMADDSD4mr,        TB_ALIGN_NONE },
    { X86::VFMADDPS4rr,       X86::VFMADDPS4mr,        TB_ALIGN_NONE },
    { X86::VFMADDPD4rr,       X86::VFMADDPD4mr,        TB_ALIGN_NONE },
    { X86::VFMADDPS4rrY,      X86::VFMADDPS4mrY,       TB_ALIGN_NONE },
    { X86::VFMADDPD4rrY,      X86::VFMADDPD4mrY,       TB_ALIGN_NONE },
    { X86::VFNMADDSS4rr,      X86::VFNMADDSS4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDSD4rr,      X86::VFNMADDSD4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDPS4rr,      X86::VFNMADDPS4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDPD4rr,      X86::VFNMADDPD4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDPS4rrY,     X86::VFNMADDPS4mrY,      TB_ALIGN_NONE },
    { X86::VFNMADDPD4rrY,     X86::VFNMADDPD4mrY,      TB_ALIGN_NONE },
    { X86::VFMSUBSS4rr,       X86::VFMSUBSS4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBSD4rr,       X86::VFMSUBSD4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBPS4rr,       X86::VFMSUBPS4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBPD4rr,       X86::VFMSUBPD4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBPS4rrY,      X86::VFMSUBPS4mrY,       TB_ALIGN_NONE },
    { X86::VFMSUBPD4rrY,      X86::VFMSUBPD4mrY,       TB_ALIGN_NONE },
    { X86::VFNMSUBSS4rr,      X86::VFNMSUBSS4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBSD4rr,      X86::VFNMSUBSD4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBPS4rr,      X86::VFNMSUBPS4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBPD4rr,      X86::VFNMSUBPD4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBPS4rrY,     X86::VFNMSUBPS4mrY,      TB_ALIGN_NONE },
    { X86::VFNMSUBPD4rrY,     X86::VFNMSUBPD4mrY,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4rr,    X86::VFMADDSUBPS4mr,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4rr,    X86::VFMADDSUBPD4mr,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4rrY,   X86::VFMADDSUBPS4mrY,    TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4rrY,   X86::VFMADDSUBPD4mrY,    TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4rr,    X86::VFMSUBADDPS4mr,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4rr,    X86::VFMSUBADDPD4mr,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4rrY,   X86::VFMSUBADDPS4mrY,    TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4rrY,   X86::VFMSUBADDPD4mrY,    TB_ALIGN_NONE },

    // XOP foldable instructions
    { X86::VPCMOVrr,          X86::VPCMOVmr,            0 },
    { X86::VPCMOVrrY,         X86::VPCMOVmrY,           0 },
    { X86::VPCOMBri,          X86::VPCOMBmi,            0 },
    { X86::VPCOMDri,          X86::VPCOMDmi,            0 },
    { X86::VPCOMQri,          X86::VPCOMQmi,            0 },
    { X86::VPCOMWri,          X86::VPCOMWmi,            0 },
    { X86::VPCOMUBri,         X86::VPCOMUBmi,           0 },
    { X86::VPCOMUDri,         X86::VPCOMUDmi,           0 },
    { X86::VPCOMUQri,         X86::VPCOMUQmi,           0 },
    { X86::VPCOMUWri,         X86::VPCOMUWmi,           0 },
    { X86::VPERMIL2PDrr,      X86::VPERMIL2PDmr,        0 },
    { X86::VPERMIL2PDrrY,     X86::VPERMIL2PDmrY,       0 },
    { X86::VPERMIL2PSrr,      X86::VPERMIL2PSmr,        0 },
    { X86::VPERMIL2PSrrY,     X86::VPERMIL2PSmrY,       0 },
    { X86::VPMACSDDrr,        X86::VPMACSDDrm,          0 },
    { X86::VPMACSDQHrr,       X86::VPMACSDQHrm,         0 },
    { X86::VPMACSDQLrr,       X86::VPMACSDQLrm,         0 },
    { X86::VPMACSSDDrr,       X86::VPMACSSDDrm,         0 },
    { X86::VPMACSSDQHrr,      X86::VPMACSSDQHrm,        0 },
    { X86::VPMACSSDQLrr,      X86::VPMACSSDQLrm,        0 },
    { X86::VPMACSSWDrr,       X86::VPMACSSWDrm,         0 },
    { X86::VPMACSSWWrr,       X86::VPMACSSWWrm,         0 },
    { X86::VPMACSWDrr,        X86::VPMACSWDrm,          0 },
    { X86::VPMACSWWrr,        X86::VPMACSWWrm,          0 },
    { X86::VPMADCSSWDrr,      X86::VPMADCSSWDrm,        0 },
    { X86::VPMADCSWDrr,       X86::VPMADCSWDrm,         0 },
    { X86::VPPERMrr,          X86::VPPERMmr,            0 },
    { X86::VPROTBrr,          X86::VPROTBrm,            0 },
    { X86::VPROTDrr,          X86::VPROTDrm,            0 },
    { X86::VPROTQrr,          X86::VPROTQrm,            0 },
    { X86::VPROTWrr,          X86::VPROTWrm,            0 },
    { X86::VPSHABrr,          X86::VPSHABrm,            0 },
    { X86::VPSHADrr,          X86::VPSHADrm,            0 },
    { X86::VPSHAQrr,          X86::VPSHAQrm,            0 },
    { X86::VPSHAWrr,          X86::VPSHAWrm,            0 },
    { X86::VPSHLBrr,          X86::VPSHLBrm,            0 },
    { X86::VPSHLDrr,          X86::VPSHLDrm,            0 },
    { X86::VPSHLQrr,          X86::VPSHLQrm,            0 },
    { X86::VPSHLWrr,          X86::VPSHLWrm,            0 },

    // BMI/BMI2 foldable instructions
    { X86::ANDN32rr,          X86::ANDN32rm,            0 },
    { X86::ANDN64rr,          X86::ANDN64rm,            0 },
    { X86::MULX32rr,          X86::MULX32rm,            0 },
    { X86::MULX64rr,          X86::MULX64rm,            0 },
    { X86::PDEP32rr,          X86::PDEP32rm,            0 },
    { X86::PDEP64rr,          X86::PDEP64rm,            0 },
    { X86::PEXT32rr,          X86::PEXT32rm,            0 },
    { X86::PEXT64rr,          X86::PEXT64rm,            0 },

    // AVX-512 foldable instructions
    { X86::VADDPSZrr,         X86::VADDPSZrm,           0 },
    { X86::VADDPDZrr,         X86::VADDPDZrm,           0 },
    { X86::VSUBPSZrr,         X86::VSUBPSZrm,           0 },
    { X86::VSUBPDZrr,         X86::VSUBPDZrm,           0 },
    { X86::VMULPSZrr,         X86::VMULPSZrm,           0 },
    { X86::VMULPDZrr,         X86::VMULPDZrm,           0 },
    { X86::VDIVPSZrr,         X86::VDIVPSZrm,           0 },
    { X86::VDIVPDZrr,         X86::VDIVPDZrm,           0 },
    { X86::VMINPSZrr,         X86::VMINPSZrm,           0 },
    { X86::VMINPDZrr,         X86::VMINPDZrm,           0 },
    { X86::VMAXPSZrr,         X86::VMAXPSZrm,           0 },
    { X86::VMAXPDZrr,         X86::VMAXPDZrm,           0 },
    { X86::VPADDDZrr,         X86::VPADDDZrm,           0 },
    { X86::VPADDQZrr,         X86::VPADDQZrm,           0 },
    { X86::VPERMPDZri,        X86::VPERMPDZmi,          0 },
    { X86::VPERMPSZrr,        X86::VPERMPSZrm,          0 },
    { X86::VPMAXSDZrr,        X86::VPMAXSDZrm,          0 },
    { X86::VPMAXSQZrr,        X86::VPMAXSQZrm,          0 },
    { X86::VPMAXUDZrr,        X86::VPMAXUDZrm,          0 },
    { X86::VPMAXUQZrr,        X86::VPMAXUQZrm,          0 },
    { X86::VPMINSDZrr,        X86::VPMINSDZrm,          0 },
    { X86::VPMINSQZrr,        X86::VPMINSQZrm,          0 },
    { X86::VPMINUDZrr,        X86::VPMINUDZrm,          0 },
    { X86::VPMINUQZrr,        X86::VPMINUQZrm,          0 },
    { X86::VPMULDQZrr,        X86::VPMULDQZrm,          0 },
    { X86::VPSLLVDZrr,        X86::VPSLLVDZrm,          0 },
    { X86::VPSLLVQZrr,        X86::VPSLLVQZrm,          0 },
    { X86::VPSRAVDZrr,        X86::VPSRAVDZrm,          0 },
    { X86::VPSRLVDZrr,        X86::VPSRLVDZrm,          0 },
    { X86::VPSRLVQZrr,        X86::VPSRLVQZrm,          0 },
    { X86::VPSUBDZrr,         X86::VPSUBDZrm,           0 },
    { X86::VPSUBQZrr,         X86::VPSUBQZrm,           0 },
    { X86::VSHUFPDZrri,       X86::VSHUFPDZrmi,         0 },
    { X86::VSHUFPSZrri,       X86::VSHUFPSZrmi,         0 },
    { X86::VALIGNQZrri,       X86::VALIGNQZrmi,         0 },
    { X86::VALIGNDZrri,       X86::VALIGNDZrmi,         0 },
    { X86::VPMULUDQZrr,       X86::VPMULUDQZrm,         0 },
    { X86::VBROADCASTSSZrkz,  X86::VBROADCASTSSZmkz,    TB_NO_REVERSE },
    { X86::VBROADCASTSDZrkz,  X86::VBROADCASTSDZmkz,    TB_NO_REVERSE },

    // AVX-512{F,VL} foldable instructions
    { X86::VBROADCASTSSZ256rkz,  X86::VBROADCASTSSZ256mkz,      TB_NO_REVERSE },
    { X86::VBROADCASTSDZ256rkz,  X86::VBROADCASTSDZ256mkz,      TB_NO_REVERSE },
    { X86::VBROADCASTSSZ128rkz,  X86::VBROADCASTSSZ128mkz,      TB_NO_REVERSE },

    // AVX-512{F,VL} foldable instructions
    { X86::VADDPDZ128rr,      X86::VADDPDZ128rm,        0 },
    { X86::VADDPDZ256rr,      X86::VADDPDZ256rm,        0 },
    { X86::VADDPSZ128rr,      X86::VADDPSZ128rm,        0 },
    { X86::VADDPSZ256rr,      X86::VADDPSZ256rm,        0 },

    // AES foldable instructions
    { X86::AESDECLASTrr,      X86::AESDECLASTrm,        TB_ALIGN_16 },
    { X86::AESDECrr,          X86::AESDECrm,            TB_ALIGN_16 },
    { X86::AESENCLASTrr,      X86::AESENCLASTrm,        TB_ALIGN_16 },
    { X86::AESENCrr,          X86::AESENCrm,            TB_ALIGN_16 },
    { X86::VAESDECLASTrr,     X86::VAESDECLASTrm,       0 },
    { X86::VAESDECrr,         X86::VAESDECrm,           0 },
    { X86::VAESENCLASTrr,     X86::VAESENCLASTrm,       0 },
    { X86::VAESENCrr,         X86::VAESENCrm,           0 },

    // SHA foldable instructions
    { X86::SHA1MSG1rr,        X86::SHA1MSG1rm,          TB_ALIGN_16 },
    { X86::SHA1MSG2rr,        X86::SHA1MSG2rm,          TB_ALIGN_16 },
    { X86::SHA1NEXTErr,       X86::SHA1NEXTErm,         TB_ALIGN_16 },
    { X86::SHA1RNDS4rri,      X86::SHA1RNDS4rmi,        TB_ALIGN_16 },
    { X86::SHA256MSG1rr,      X86::SHA256MSG1rm,        TB_ALIGN_16 },
    { X86::SHA256MSG2rr,      X86::SHA256MSG2rm,        TB_ALIGN_16 },
    { X86::SHA256RNDS2rr,     X86::SHA256RNDS2rm,       TB_ALIGN_16 }
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable2) {
    AddTableEntry(RegOp2MemOpTable2, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 2, folded load
                  Entry.Flags | TB_INDEX_2 | TB_FOLDED_LOAD);
  }

  static const X86MemoryFoldTableEntry MemoryFoldTable3[] = {
    // FMA foldable instructions
    { X86::VFMADDSSr231r,         X86::VFMADDSSr231m,         TB_ALIGN_NONE },
    { X86::VFMADDSSr231r_Int,     X86::VFMADDSSr231m_Int,     TB_ALIGN_NONE },
    { X86::VFMADDSDr231r,         X86::VFMADDSDr231m,         TB_ALIGN_NONE },
    { X86::VFMADDSDr231r_Int,     X86::VFMADDSDr231m_Int,     TB_ALIGN_NONE },
    { X86::VFMADDSSr132r,         X86::VFMADDSSr132m,         TB_ALIGN_NONE },
    { X86::VFMADDSSr132r_Int,     X86::VFMADDSSr132m_Int,     TB_ALIGN_NONE },
    { X86::VFMADDSDr132r,         X86::VFMADDSDr132m,         TB_ALIGN_NONE },
    { X86::VFMADDSDr132r_Int,     X86::VFMADDSDr132m_Int,     TB_ALIGN_NONE },
    { X86::VFMADDSSr213r,         X86::VFMADDSSr213m,         TB_ALIGN_NONE },
    { X86::VFMADDSSr213r_Int,     X86::VFMADDSSr213m_Int,     TB_ALIGN_NONE },
    { X86::VFMADDSDr213r,         X86::VFMADDSDr213m,         TB_ALIGN_NONE },
    { X86::VFMADDSDr213r_Int,     X86::VFMADDSDr213m_Int,     TB_ALIGN_NONE },

    { X86::VFMADDPSr231r,         X86::VFMADDPSr231m,         TB_ALIGN_NONE },
    { X86::VFMADDPDr231r,         X86::VFMADDPDr231m,         TB_ALIGN_NONE },
    { X86::VFMADDPSr132r,         X86::VFMADDPSr132m,         TB_ALIGN_NONE },
    { X86::VFMADDPDr132r,         X86::VFMADDPDr132m,         TB_ALIGN_NONE },
    { X86::VFMADDPSr213r,         X86::VFMADDPSr213m,         TB_ALIGN_NONE },
    { X86::VFMADDPDr213r,         X86::VFMADDPDr213m,         TB_ALIGN_NONE },
    { X86::VFMADDPSr231rY,        X86::VFMADDPSr231mY,        TB_ALIGN_NONE },
    { X86::VFMADDPDr231rY,        X86::VFMADDPDr231mY,        TB_ALIGN_NONE },
    { X86::VFMADDPSr132rY,        X86::VFMADDPSr132mY,        TB_ALIGN_NONE },
    { X86::VFMADDPDr132rY,        X86::VFMADDPDr132mY,        TB_ALIGN_NONE },
    { X86::VFMADDPSr213rY,        X86::VFMADDPSr213mY,        TB_ALIGN_NONE },
    { X86::VFMADDPDr213rY,        X86::VFMADDPDr213mY,        TB_ALIGN_NONE },

    { X86::VFNMADDSSr231r,        X86::VFNMADDSSr231m,        TB_ALIGN_NONE },
    { X86::VFNMADDSSr231r_Int,    X86::VFNMADDSSr231m_Int,    TB_ALIGN_NONE },
    { X86::VFNMADDSDr231r,        X86::VFNMADDSDr231m,        TB_ALIGN_NONE },
    { X86::VFNMADDSDr231r_Int,    X86::VFNMADDSDr231m_Int,    TB_ALIGN_NONE },
    { X86::VFNMADDSSr132r,        X86::VFNMADDSSr132m,        TB_ALIGN_NONE },
    { X86::VFNMADDSSr132r_Int,    X86::VFNMADDSSr132m_Int,    TB_ALIGN_NONE },
    { X86::VFNMADDSDr132r,        X86::VFNMADDSDr132m,        TB_ALIGN_NONE },
    { X86::VFNMADDSDr132r_Int,    X86::VFNMADDSDr132m_Int,    TB_ALIGN_NONE },
    { X86::VFNMADDSSr213r,        X86::VFNMADDSSr213m,        TB_ALIGN_NONE },
    { X86::VFNMADDSSr213r_Int,    X86::VFNMADDSSr213m_Int,    TB_ALIGN_NONE },
    { X86::VFNMADDSDr213r,        X86::VFNMADDSDr213m,        TB_ALIGN_NONE },
    { X86::VFNMADDSDr213r_Int,    X86::VFNMADDSDr213m_Int,    TB_ALIGN_NONE },

    { X86::VFNMADDPSr231r,        X86::VFNMADDPSr231m,        TB_ALIGN_NONE },
    { X86::VFNMADDPDr231r,        X86::VFNMADDPDr231m,        TB_ALIGN_NONE },
    { X86::VFNMADDPSr132r,        X86::VFNMADDPSr132m,        TB_ALIGN_NONE },
    { X86::VFNMADDPDr132r,        X86::VFNMADDPDr132m,        TB_ALIGN_NONE },
    { X86::VFNMADDPSr213r,        X86::VFNMADDPSr213m,        TB_ALIGN_NONE },
    { X86::VFNMADDPDr213r,        X86::VFNMADDPDr213m,        TB_ALIGN_NONE },
    { X86::VFNMADDPSr231rY,       X86::VFNMADDPSr231mY,       TB_ALIGN_NONE },
    { X86::VFNMADDPDr231rY,       X86::VFNMADDPDr231mY,       TB_ALIGN_NONE },
    { X86::VFNMADDPSr132rY,       X86::VFNMADDPSr132mY,       TB_ALIGN_NONE },
    { X86::VFNMADDPDr132rY,       X86::VFNMADDPDr132mY,       TB_ALIGN_NONE },
    { X86::VFNMADDPSr213rY,       X86::VFNMADDPSr213mY,       TB_ALIGN_NONE },
    { X86::VFNMADDPDr213rY,       X86::VFNMADDPDr213mY,       TB_ALIGN_NONE },

    { X86::VFMSUBSSr231r,         X86::VFMSUBSSr231m,         TB_ALIGN_NONE },
    { X86::VFMSUBSSr231r_Int,     X86::VFMSUBSSr231m_Int,     TB_ALIGN_NONE },
    { X86::VFMSUBSDr231r,         X86::VFMSUBSDr231m,         TB_ALIGN_NONE },
    { X86::VFMSUBSDr231r_Int,     X86::VFMSUBSDr231m_Int,     TB_ALIGN_NONE },
    { X86::VFMSUBSSr132r,         X86::VFMSUBSSr132m,         TB_ALIGN_NONE },
    { X86::VFMSUBSSr132r_Int,     X86::VFMSUBSSr132m_Int,     TB_ALIGN_NONE },
    { X86::VFMSUBSDr132r,         X86::VFMSUBSDr132m,         TB_ALIGN_NONE },
    { X86::VFMSUBSDr132r_Int,     X86::VFMSUBSDr132m_Int,     TB_ALIGN_NONE },
    { X86::VFMSUBSSr213r,         X86::VFMSUBSSr213m,         TB_ALIGN_NONE },
    { X86::VFMSUBSSr213r_Int,     X86::VFMSUBSSr213m_Int,     TB_ALIGN_NONE },
    { X86::VFMSUBSDr213r,         X86::VFMSUBSDr213m,         TB_ALIGN_NONE },
    { X86::VFMSUBSDr213r_Int,     X86::VFMSUBSDr213m_Int,     TB_ALIGN_NONE },

    { X86::VFMSUBPSr231r,         X86::VFMSUBPSr231m,         TB_ALIGN_NONE },
    { X86::VFMSUBPDr231r,         X86::VFMSUBPDr231m,         TB_ALIGN_NONE },
    { X86::VFMSUBPSr132r,         X86::VFMSUBPSr132m,         TB_ALIGN_NONE },
    { X86::VFMSUBPDr132r,         X86::VFMSUBPDr132m,         TB_ALIGN_NONE },
    { X86::VFMSUBPSr213r,         X86::VFMSUBPSr213m,         TB_ALIGN_NONE },
    { X86::VFMSUBPDr213r,         X86::VFMSUBPDr213m,         TB_ALIGN_NONE },
    { X86::VFMSUBPSr231rY,        X86::VFMSUBPSr231mY,        TB_ALIGN_NONE },
    { X86::VFMSUBPDr231rY,        X86::VFMSUBPDr231mY,        TB_ALIGN_NONE },
    { X86::VFMSUBPSr132rY,        X86::VFMSUBPSr132mY,        TB_ALIGN_NONE },
    { X86::VFMSUBPDr132rY,        X86::VFMSUBPDr132mY,        TB_ALIGN_NONE },
    { X86::VFMSUBPSr213rY,        X86::VFMSUBPSr213mY,        TB_ALIGN_NONE },
    { X86::VFMSUBPDr213rY,        X86::VFMSUBPDr213mY,        TB_ALIGN_NONE },

    { X86::VFNMSUBSSr231r,        X86::VFNMSUBSSr231m,        TB_ALIGN_NONE },
    { X86::VFNMSUBSSr231r_Int,    X86::VFNMSUBSSr231m_Int,    TB_ALIGN_NONE },
    { X86::VFNMSUBSDr231r,        X86::VFNMSUBSDr231m,        TB_ALIGN_NONE },
    { X86::VFNMSUBSDr231r_Int,    X86::VFNMSUBSDr231m_Int,    TB_ALIGN_NONE },
    { X86::VFNMSUBSSr132r,        X86::VFNMSUBSSr132m,        TB_ALIGN_NONE },
    { X86::VFNMSUBSSr132r_Int,    X86::VFNMSUBSSr132m_Int,    TB_ALIGN_NONE },
    { X86::VFNMSUBSDr132r,        X86::VFNMSUBSDr132m,        TB_ALIGN_NONE },
    { X86::VFNMSUBSDr132r_Int,    X86::VFNMSUBSDr132m_Int,    TB_ALIGN_NONE },
    { X86::VFNMSUBSSr213r,        X86::VFNMSUBSSr213m,        TB_ALIGN_NONE },
    { X86::VFNMSUBSSr213r_Int,    X86::VFNMSUBSSr213m_Int,    TB_ALIGN_NONE },
    { X86::VFNMSUBSDr213r,        X86::VFNMSUBSDr213m,        TB_ALIGN_NONE },
    { X86::VFNMSUBSDr213r_Int,    X86::VFNMSUBSDr213m_Int,    TB_ALIGN_NONE },

    { X86::VFNMSUBPSr231r,        X86::VFNMSUBPSr231m,        TB_ALIGN_NONE },
    { X86::VFNMSUBPDr231r,        X86::VFNMSUBPDr231m,        TB_ALIGN_NONE },
    { X86::VFNMSUBPSr132r,        X86::VFNMSUBPSr132m,        TB_ALIGN_NONE },
    { X86::VFNMSUBPDr132r,        X86::VFNMSUBPDr132m,        TB_ALIGN_NONE },
    { X86::VFNMSUBPSr213r,        X86::VFNMSUBPSr213m,        TB_ALIGN_NONE },
    { X86::VFNMSUBPDr213r,        X86::VFNMSUBPDr213m,        TB_ALIGN_NONE },
    { X86::VFNMSUBPSr231rY,       X86::VFNMSUBPSr231mY,       TB_ALIGN_NONE },
    { X86::VFNMSUBPDr231rY,       X86::VFNMSUBPDr231mY,       TB_ALIGN_NONE },
    { X86::VFNMSUBPSr132rY,       X86::VFNMSUBPSr132mY,       TB_ALIGN_NONE },
    { X86::VFNMSUBPDr132rY,       X86::VFNMSUBPDr132mY,       TB_ALIGN_NONE },
    { X86::VFNMSUBPSr213rY,       X86::VFNMSUBPSr213mY,       TB_ALIGN_NONE },
    { X86::VFNMSUBPDr213rY,       X86::VFNMSUBPDr213mY,       TB_ALIGN_NONE },

    { X86::VFMADDSUBPSr231r,      X86::VFMADDSUBPSr231m,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPDr231r,      X86::VFMADDSUBPDr231m,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPSr132r,      X86::VFMADDSUBPSr132m,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPDr132r,      X86::VFMADDSUBPDr132m,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPSr213r,      X86::VFMADDSUBPSr213m,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPDr213r,      X86::VFMADDSUBPDr213m,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPSr231rY,     X86::VFMADDSUBPSr231mY,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPDr231rY,     X86::VFMADDSUBPDr231mY,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPSr132rY,     X86::VFMADDSUBPSr132mY,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPDr132rY,     X86::VFMADDSUBPDr132mY,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPSr213rY,     X86::VFMADDSUBPSr213mY,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPDr213rY,     X86::VFMADDSUBPDr213mY,     TB_ALIGN_NONE },

    { X86::VFMSUBADDPSr231r,      X86::VFMSUBADDPSr231m,      TB_ALIGN_NONE },
    { X86::VFMSUBADDPDr231r,      X86::VFMSUBADDPDr231m,      TB_ALIGN_NONE },
    { X86::VFMSUBADDPSr132r,      X86::VFMSUBADDPSr132m,      TB_ALIGN_NONE },
    { X86::VFMSUBADDPDr132r,      X86::VFMSUBADDPDr132m,      TB_ALIGN_NONE },
    { X86::VFMSUBADDPSr213r,      X86::VFMSUBADDPSr213m,      TB_ALIGN_NONE },
    { X86::VFMSUBADDPDr213r,      X86::VFMSUBADDPDr213m,      TB_ALIGN_NONE },
    { X86::VFMSUBADDPSr231rY,     X86::VFMSUBADDPSr231mY,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPDr231rY,     X86::VFMSUBADDPDr231mY,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPSr132rY,     X86::VFMSUBADDPSr132mY,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPDr132rY,     X86::VFMSUBADDPDr132mY,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPSr213rY,     X86::VFMSUBADDPSr213mY,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPDr213rY,     X86::VFMSUBADDPDr213mY,     TB_ALIGN_NONE },

    // FMA4 foldable patterns
    { X86::VFMADDSS4rr,           X86::VFMADDSS4rm,           TB_ALIGN_NONE },
    { X86::VFMADDSD4rr,           X86::VFMADDSD4rm,           TB_ALIGN_NONE },
    { X86::VFMADDPS4rr,           X86::VFMADDPS4rm,           TB_ALIGN_NONE },
    { X86::VFMADDPD4rr,           X86::VFMADDPD4rm,           TB_ALIGN_NONE },
    { X86::VFMADDPS4rrY,          X86::VFMADDPS4rmY,          TB_ALIGN_NONE },
    { X86::VFMADDPD4rrY,          X86::VFMADDPD4rmY,          TB_ALIGN_NONE },
    { X86::VFNMADDSS4rr,          X86::VFNMADDSS4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDSD4rr,          X86::VFNMADDSD4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDPS4rr,          X86::VFNMADDPS4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDPD4rr,          X86::VFNMADDPD4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDPS4rrY,         X86::VFNMADDPS4rmY,         TB_ALIGN_NONE },
    { X86::VFNMADDPD4rrY,         X86::VFNMADDPD4rmY,         TB_ALIGN_NONE },
    { X86::VFMSUBSS4rr,           X86::VFMSUBSS4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBSD4rr,           X86::VFMSUBSD4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBPS4rr,           X86::VFMSUBPS4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBPD4rr,           X86::VFMSUBPD4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBPS4rrY,          X86::VFMSUBPS4rmY,          TB_ALIGN_NONE },
    { X86::VFMSUBPD4rrY,          X86::VFMSUBPD4rmY,          TB_ALIGN_NONE },
    { X86::VFNMSUBSS4rr,          X86::VFNMSUBSS4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBSD4rr,          X86::VFNMSUBSD4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBPS4rr,          X86::VFNMSUBPS4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBPD4rr,          X86::VFNMSUBPD4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBPS4rrY,         X86::VFNMSUBPS4rmY,         TB_ALIGN_NONE },
    { X86::VFNMSUBPD4rrY,         X86::VFNMSUBPD4rmY,         TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4rr,        X86::VFMADDSUBPS4rm,        TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4rr,        X86::VFMADDSUBPD4rm,        TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4rrY,       X86::VFMADDSUBPS4rmY,       TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4rrY,       X86::VFMADDSUBPD4rmY,       TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4rr,        X86::VFMSUBADDPS4rm,        TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4rr,        X86::VFMSUBADDPD4rm,        TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4rrY,       X86::VFMSUBADDPS4rmY,       TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4rrY,       X86::VFMSUBADDPD4rmY,       TB_ALIGN_NONE },

    // XOP foldable instructions
    { X86::VPCMOVrr,              X86::VPCMOVrm,              0 },
    { X86::VPCMOVrrY,             X86::VPCMOVrmY,             0 },
    { X86::VPERMIL2PDrr,          X86::VPERMIL2PDrm,          0 },
    { X86::VPERMIL2PDrrY,         X86::VPERMIL2PDrmY,         0 },
    { X86::VPERMIL2PSrr,          X86::VPERMIL2PSrm,          0 },
    { X86::VPERMIL2PSrrY,         X86::VPERMIL2PSrmY,         0 },
    { X86::VPPERMrr,              X86::VPPERMrm,              0 },

    // AVX-512 VPERMI instructions with 3 source operands.
    { X86::VPERMI2Drr,            X86::VPERMI2Drm,            0 },
    { X86::VPERMI2Qrr,            X86::VPERMI2Qrm,            0 },
    { X86::VPERMI2PSrr,           X86::VPERMI2PSrm,           0 },
    { X86::VPERMI2PDrr,           X86::VPERMI2PDrm,           0 },
    { X86::VBLENDMPDZrr,          X86::VBLENDMPDZrm,          0 },
    { X86::VBLENDMPSZrr,          X86::VBLENDMPSZrm,          0 },
    { X86::VPBLENDMDZrr,          X86::VPBLENDMDZrm,          0 },
    { X86::VPBLENDMQZrr,          X86::VPBLENDMQZrm,          0 },
    { X86::VBROADCASTSSZrk,       X86::VBROADCASTSSZmk,       TB_NO_REVERSE },
    { X86::VBROADCASTSDZrk,       X86::VBROADCASTSDZmk,       TB_NO_REVERSE },
    { X86::VBROADCASTSSZ256rk,    X86::VBROADCASTSSZ256mk,    TB_NO_REVERSE },
    { X86::VBROADCASTSDZ256rk,    X86::VBROADCASTSDZ256mk,    TB_NO_REVERSE },
    { X86::VBROADCASTSSZ128rk,    X86::VBROADCASTSSZ128mk,    TB_NO_REVERSE },
     // AVX-512 arithmetic instructions
    { X86::VADDPSZrrkz,           X86::VADDPSZrmkz,           0 },
    { X86::VADDPDZrrkz,           X86::VADDPDZrmkz,           0 },
    { X86::VSUBPSZrrkz,           X86::VSUBPSZrmkz,           0 },
    { X86::VSUBPDZrrkz,           X86::VSUBPDZrmkz,           0 },
    { X86::VMULPSZrrkz,           X86::VMULPSZrmkz,           0 },
    { X86::VMULPDZrrkz,           X86::VMULPDZrmkz,           0 },
    { X86::VDIVPSZrrkz,           X86::VDIVPSZrmkz,           0 },
    { X86::VDIVPDZrrkz,           X86::VDIVPDZrmkz,           0 },
    { X86::VMINPSZrrkz,           X86::VMINPSZrmkz,           0 },
    { X86::VMINPDZrrkz,           X86::VMINPDZrmkz,           0 },
    { X86::VMAXPSZrrkz,           X86::VMAXPSZrmkz,           0 },
    { X86::VMAXPDZrrkz,           X86::VMAXPDZrmkz,           0 },
    // AVX-512{F,VL} arithmetic instructions 256-bit
    { X86::VADDPSZ256rrkz,        X86::VADDPSZ256rmkz,        0 },
    { X86::VADDPDZ256rrkz,        X86::VADDPDZ256rmkz,        0 },
    { X86::VSUBPSZ256rrkz,        X86::VSUBPSZ256rmkz,        0 },
    { X86::VSUBPDZ256rrkz,        X86::VSUBPDZ256rmkz,        0 },
    { X86::VMULPSZ256rrkz,        X86::VMULPSZ256rmkz,        0 },
    { X86::VMULPDZ256rrkz,        X86::VMULPDZ256rmkz,        0 },
    { X86::VDIVPSZ256rrkz,        X86::VDIVPSZ256rmkz,        0 },
    { X86::VDIVPDZ256rrkz,        X86::VDIVPDZ256rmkz,        0 },
    { X86::VMINPSZ256rrkz,        X86::VMINPSZ256rmkz,        0 },
    { X86::VMINPDZ256rrkz,        X86::VMINPDZ256rmkz,        0 },
    { X86::VMAXPSZ256rrkz,        X86::VMAXPSZ256rmkz,        0 },
    { X86::VMAXPDZ256rrkz,        X86::VMAXPDZ256rmkz,        0 },
    // AVX-512{F,VL} arithmetic instructions 128-bit
    { X86::VADDPSZ128rrkz,        X86::VADDPSZ128rmkz,        0 },
    { X86::VADDPDZ128rrkz,        X86::VADDPDZ128rmkz,        0 },
    { X86::VSUBPSZ128rrkz,        X86::VSUBPSZ128rmkz,        0 },
    { X86::VSUBPDZ128rrkz,        X86::VSUBPDZ128rmkz,        0 },
    { X86::VMULPSZ128rrkz,        X86::VMULPSZ128rmkz,        0 },
    { X86::VMULPDZ128rrkz,        X86::VMULPDZ128rmkz,        0 },
    { X86::VDIVPSZ128rrkz,        X86::VDIVPSZ128rmkz,        0 },
    { X86::VDIVPDZ128rrkz,        X86::VDIVPDZ128rmkz,        0 },
    { X86::VMINPSZ128rrkz,        X86::VMINPSZ128rmkz,        0 },
    { X86::VMINPDZ128rrkz,        X86::VMINPDZ128rmkz,        0 },
    { X86::VMAXPSZ128rrkz,        X86::VMAXPSZ128rmkz,        0 },
    { X86::VMAXPDZ128rrkz,        X86::VMAXPDZ128rmkz,        0 }
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable3) {
    AddTableEntry(RegOp2MemOpTable3, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 3, folded load
                  Entry.Flags | TB_INDEX_3 | TB_FOLDED_LOAD);
  }

  static const X86MemoryFoldTableEntry MemoryFoldTable4[] = {
     // AVX-512 foldable instructions
    { X86::VADDPSZrrk,         X86::VADDPSZrmk,           0 },
    { X86::VADDPDZrrk,         X86::VADDPDZrmk,           0 },
    { X86::VSUBPSZrrk,         X86::VSUBPSZrmk,           0 },
    { X86::VSUBPDZrrk,         X86::VSUBPDZrmk,           0 },
    { X86::VMULPSZrrk,         X86::VMULPSZrmk,           0 },
    { X86::VMULPDZrrk,         X86::VMULPDZrmk,           0 },
    { X86::VDIVPSZrrk,         X86::VDIVPSZrmk,           0 },
    { X86::VDIVPDZrrk,         X86::VDIVPDZrmk,           0 },
    { X86::VMINPSZrrk,         X86::VMINPSZrmk,           0 },
    { X86::VMINPDZrrk,         X86::VMINPDZrmk,           0 },
    { X86::VMAXPSZrrk,         X86::VMAXPSZrmk,           0 },
    { X86::VMAXPDZrrk,         X86::VMAXPDZrmk,           0 },
    // AVX-512{F,VL} foldable instructions 256-bit
    { X86::VADDPSZ256rrk,      X86::VADDPSZ256rmk,        0 },
    { X86::VADDPDZ256rrk,      X86::VADDPDZ256rmk,        0 },
    { X86::VSUBPSZ256rrk,      X86::VSUBPSZ256rmk,        0 },
    { X86::VSUBPDZ256rrk,      X86::VSUBPDZ256rmk,        0 },
    { X86::VMULPSZ256rrk,      X86::VMULPSZ256rmk,        0 },
    { X86::VMULPDZ256rrk,      X86::VMULPDZ256rmk,        0 },
    { X86::VDIVPSZ256rrk,      X86::VDIVPSZ256rmk,        0 },
    { X86::VDIVPDZ256rrk,      X86::VDIVPDZ256rmk,        0 },
    { X86::VMINPSZ256rrk,      X86::VMINPSZ256rmk,        0 },
    { X86::VMINPDZ256rrk,      X86::VMINPDZ256rmk,        0 },
    { X86::VMAXPSZ256rrk,      X86::VMAXPSZ256rmk,        0 },
    { X86::VMAXPDZ256rrk,      X86::VMAXPDZ256rmk,        0 },
    // AVX-512{F,VL} foldable instructions 128-bit
    { X86::VADDPSZ128rrk,      X86::VADDPSZ128rmk,        0 },
    { X86::VADDPDZ128rrk,      X86::VADDPDZ128rmk,        0 },
    { X86::VSUBPSZ128rrk,      X86::VSUBPSZ128rmk,        0 },
    { X86::VSUBPDZ128rrk,      X86::VSUBPDZ128rmk,        0 },
    { X86::VMULPSZ128rrk,      X86::VMULPSZ128rmk,        0 },
    { X86::VMULPDZ128rrk,      X86::VMULPDZ128rmk,        0 },
    { X86::VDIVPSZ128rrk,      X86::VDIVPSZ128rmk,        0 },
    { X86::VDIVPDZ128rrk,      X86::VDIVPDZ128rmk,        0 },
    { X86::VMINPSZ128rrk,      X86::VMINPSZ128rmk,        0 },
    { X86::VMINPDZ128rrk,      X86::VMINPDZ128rmk,        0 },
    { X86::VMAXPSZ128rrk,      X86::VMAXPSZ128rmk,        0 },
    { X86::VMAXPDZ128rrk,      X86::VMAXPDZ128rmk,        0 }
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable4) {
    AddTableEntry(RegOp2MemOpTable4, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 4, folded load
                  Entry.Flags | TB_INDEX_4 | TB_FOLDED_LOAD);
  }
}

void
X86InstrInfo::AddTableEntry(RegOp2MemOpTableType &R2MTable,
                            MemOp2RegOpTableType &M2RTable,
                            unsigned RegOp, unsigned MemOp, unsigned Flags) {
    if ((Flags & TB_NO_FORWARD) == 0) {
      assert(!R2MTable.count(RegOp) && "Duplicate entry!");
      R2MTable[RegOp] = std::make_pair(MemOp, Flags);
    }
    if ((Flags & TB_NO_REVERSE) == 0) {
      assert(!M2RTable.count(MemOp) &&
           "Duplicated entries in unfolding maps?");
      M2RTable[MemOp] = std::make_pair(RegOp, Flags);
    }
}

bool
X86InstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                    unsigned &SrcReg, unsigned &DstReg,
                                    unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default: break;
  case X86::MOVSX16rr8:
  case X86::MOVZX16rr8:
  case X86::MOVSX32rr8:
  case X86::MOVZX32rr8:
  case X86::MOVSX64rr8:
    if (!Subtarget.is64Bit())
      // It's not always legal to reference the low 8-bit of the larger
      // register in 32-bit mode.
      return false;
  case X86::MOVSX32rr16:
  case X86::MOVZX32rr16:
  case X86::MOVSX64rr16:
  case X86::MOVSX64rr32: {
    if (MI.getOperand(0).getSubReg() || MI.getOperand(1).getSubReg())
      // Be conservative.
      return false;
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    switch (MI.getOpcode()) {
    default: llvm_unreachable("Unreachable!");
    case X86::MOVSX16rr8:
    case X86::MOVZX16rr8:
    case X86::MOVSX32rr8:
    case X86::MOVZX32rr8:
    case X86::MOVSX64rr8:
      SubIdx = X86::sub_8bit;
      break;
    case X86::MOVSX32rr16:
    case X86::MOVZX32rr16:
    case X86::MOVSX64rr16:
      SubIdx = X86::sub_16bit;
      break;
    case X86::MOVSX64rr32:
      SubIdx = X86::sub_32bit;
      break;
    }
    return true;
  }
  }
  return false;
}

int X86InstrInfo::getSPAdjust(const MachineInstr *MI) const {
  const MachineFunction *MF = MI->getParent()->getParent();
  const TargetFrameLowering *TFI = MF->getSubtarget().getFrameLowering();

  if (MI->getOpcode() == getCallFrameSetupOpcode() ||
      MI->getOpcode() == getCallFrameDestroyOpcode()) {
    unsigned StackAlign = TFI->getStackAlignment();
    int SPAdj = (MI->getOperand(0).getImm() + StackAlign - 1) / StackAlign *
                 StackAlign;

    SPAdj -= MI->getOperand(1).getImm();

    if (MI->getOpcode() == getCallFrameSetupOpcode())
      return SPAdj;
    else
      return -SPAdj;
  }

  // To know whether a call adjusts the stack, we need information
  // that is bound to the following ADJCALLSTACKUP pseudo.
  // Look for the next ADJCALLSTACKUP that follows the call.
  if (MI->isCall()) {
    const MachineBasicBlock* MBB = MI->getParent();
    auto I = ++MachineBasicBlock::const_iterator(MI);
    for (auto E = MBB->end(); I != E; ++I) {
      if (I->getOpcode() == getCallFrameDestroyOpcode() ||
          I->isCall())
        break;
    }

    // If we could not find a frame destroy opcode, then it has already
    // been simplified, so we don't care.
    if (I->getOpcode() != getCallFrameDestroyOpcode())
      return 0;

    return -(I->getOperand(1).getImm());
  }

  // Currently handle only PUSHes we can reasonably expect to see
  // in call sequences
  switch (MI->getOpcode()) {
  default:
    return 0;
  case X86::PUSH32i8:
  case X86::PUSH32r:
  case X86::PUSH32rmm:
  case X86::PUSH32rmr:
  case X86::PUSHi32:
    return 4;
  }
}

/// Return true and the FrameIndex if the specified
/// operand and follow operands form a reference to the stack frame.
bool X86InstrInfo::isFrameOperand(const MachineInstr *MI, unsigned int Op,
                                  int &FrameIndex) const {
  if (MI->getOperand(Op+X86::AddrBaseReg).isFI() &&
      MI->getOperand(Op+X86::AddrScaleAmt).isImm() &&
      MI->getOperand(Op+X86::AddrIndexReg).isReg() &&
      MI->getOperand(Op+X86::AddrDisp).isImm() &&
      MI->getOperand(Op+X86::AddrScaleAmt).getImm() == 1 &&
      MI->getOperand(Op+X86::AddrIndexReg).getReg() == 0 &&
      MI->getOperand(Op+X86::AddrDisp).getImm() == 0) {
    FrameIndex = MI->getOperand(Op+X86::AddrBaseReg).getIndex();
    return true;
  }
  return false;
}

static bool isFrameLoadOpcode(int Opcode) {
  switch (Opcode) {
  default:
    return false;
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
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPDYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVDQUYrm:
  case X86::VMOVDQAYrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::VMOVAPSZrm:
  case X86::VMOVUPSZrm:
    return true;
  }
}

static bool isFrameStoreOpcode(int Opcode) {
  switch (Opcode) {
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
  case X86::VMOVSSmr:
  case X86::VMOVSDmr:
  case X86::VMOVAPSmr:
  case X86::VMOVAPDmr:
  case X86::VMOVDQAmr:
  case X86::VMOVUPSYmr:
  case X86::VMOVAPSYmr:
  case X86::VMOVUPDYmr:
  case X86::VMOVAPDYmr:
  case X86::VMOVDQUYmr:
  case X86::VMOVDQAYmr:
  case X86::VMOVUPSZmr:
  case X86::VMOVAPSZmr:
  case X86::MMX_MOVD64mr:
  case X86::MMX_MOVQ64mr:
  case X86::MMX_MOVNTQmr:
    return true;
  }
  return false;
}

unsigned X86InstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                           int &FrameIndex) const {
  if (isFrameLoadOpcode(MI->getOpcode()))
    if (MI->getOperand(0).getSubReg() == 0 && isFrameOperand(MI, 1, FrameIndex))
      return MI->getOperand(0).getReg();
  return 0;
}

unsigned X86InstrInfo::isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                                 int &FrameIndex) const {
  if (isFrameLoadOpcode(MI->getOpcode())) {
    unsigned Reg;
    if ((Reg = isLoadFromStackSlot(MI, FrameIndex)))
      return Reg;
    // Check for post-frame index elimination operations
    const MachineMemOperand *Dummy;
    return hasLoadFromStackSlot(MI, Dummy, FrameIndex);
  }
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  if (isFrameStoreOpcode(MI->getOpcode()))
    if (MI->getOperand(X86::AddrNumOperands).getSubReg() == 0 &&
        isFrameOperand(MI, 0, FrameIndex))
      return MI->getOperand(X86::AddrNumOperands).getReg();
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlotPostFE(const MachineInstr *MI,
                                                int &FrameIndex) const {
  if (isFrameStoreOpcode(MI->getOpcode())) {
    unsigned Reg;
    if ((Reg = isStoreToStackSlot(MI, FrameIndex)))
      return Reg;
    // Check for post-frame index elimination operations
    const MachineMemOperand *Dummy;
    return hasStoreToStackSlot(MI, Dummy, FrameIndex);
  }
  return 0;
}

/// Return true if register is PIC base; i.e.g defined by X86::MOVPC32r.
static bool regIsPICBase(unsigned BaseReg, const MachineRegisterInfo &MRI) {
  // Don't waste compile time scanning use-def chains of physregs.
  if (!TargetRegisterInfo::isVirtualRegister(BaseReg))
    return false;
  bool isPICBase = false;
  for (MachineRegisterInfo::def_instr_iterator I = MRI.def_instr_begin(BaseReg),
         E = MRI.def_instr_end(); I != E; ++I) {
    MachineInstr *DefMI = &*I;
    if (DefMI->getOpcode() != X86::MOVPC32r)
      return false;
    assert(!isPICBase && "More than one PIC base?");
    isPICBase = true;
  }
  return isPICBase;
}

bool
X86InstrInfo::isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                                AliasAnalysis *AA) const {
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
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVDQAYrm:
  case X86::VMOVDQUYrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::FsVMOVAPSrm:
  case X86::FsVMOVAPDrm:
  case X86::FsMOVAPSrm:
  case X86::FsMOVAPDrm:
  // AVX-512
  case X86::VMOVAPDZ128rm:
  case X86::VMOVAPDZ256rm:
  case X86::VMOVAPDZrm:
  case X86::VMOVAPSZ128rm:
  case X86::VMOVAPSZ256rm:
  case X86::VMOVAPSZrm:
  case X86::VMOVDQA32Z128rm:
  case X86::VMOVDQA32Z256rm:
  case X86::VMOVDQA32Zrm:
  case X86::VMOVDQA64Z128rm:
  case X86::VMOVDQA64Z256rm:
  case X86::VMOVDQA64Zrm:
  case X86::VMOVDQU16Z128rm:
  case X86::VMOVDQU16Z256rm:
  case X86::VMOVDQU16Zrm:
  case X86::VMOVDQU32Z128rm:
  case X86::VMOVDQU32Z256rm:
  case X86::VMOVDQU32Zrm:
  case X86::VMOVDQU64Z128rm:
  case X86::VMOVDQU64Z256rm:
  case X86::VMOVDQU64Zrm:
  case X86::VMOVDQU8Z128rm:
  case X86::VMOVDQU8Z256rm:
  case X86::VMOVDQU8Zrm:
  case X86::VMOVUPSZ128rm:
  case X86::VMOVUPSZ256rm:
  case X86::VMOVUPSZrm: {
    // Loads from constant pools are trivially rematerializable.
    if (MI->getOperand(1+X86::AddrBaseReg).isReg() &&
        MI->getOperand(1+X86::AddrScaleAmt).isImm() &&
        MI->getOperand(1+X86::AddrIndexReg).isReg() &&
        MI->getOperand(1+X86::AddrIndexReg).getReg() == 0 &&
        MI->isInvariantLoad(AA)) {
      unsigned BaseReg = MI->getOperand(1+X86::AddrBaseReg).getReg();
      if (BaseReg == 0 || BaseReg == X86::RIP)
        return true;
      // Allow re-materialization of PIC load.
      if (!ReMatPICStubLoad && MI->getOperand(1+X86::AddrDisp).isGlobal())
        return false;
      const MachineFunction &MF = *MI->getParent()->getParent();
      const MachineRegisterInfo &MRI = MF.getRegInfo();
      return regIsPICBase(BaseReg, MRI);
    }
    return false;
  }

  case X86::LEA32r:
  case X86::LEA64r: {
    if (MI->getOperand(1+X86::AddrScaleAmt).isImm() &&
        MI->getOperand(1+X86::AddrIndexReg).isReg() &&
        MI->getOperand(1+X86::AddrIndexReg).getReg() == 0 &&
        !MI->getOperand(1+X86::AddrDisp).isReg()) {
      // lea fi#, lea GV, etc. are all rematerializable.
      if (!MI->getOperand(1+X86::AddrBaseReg).isReg())
        return true;
      unsigned BaseReg = MI->getOperand(1+X86::AddrBaseReg).getReg();
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

bool X86InstrInfo::isSafeToClobberEFLAGS(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I) const {
  MachineBasicBlock::iterator E = MBB.end();

  // For compile time consideration, if we are not able to determine the
  // safety after visiting 4 instructions in each direction, we will assume
  // it's not safe.
  MachineBasicBlock::iterator Iter = I;
  for (unsigned i = 0; Iter != E && i < 4; ++i) {
    bool SeenDef = false;
    for (unsigned j = 0, e = Iter->getNumOperands(); j != e; ++j) {
      MachineOperand &MO = Iter->getOperand(j);
      if (MO.isRegMask() && MO.clobbersPhysReg(X86::EFLAGS))
        SeenDef = true;
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
    ++Iter;
    // Skip over DBG_VALUE.
    while (Iter != E && Iter->isDebugValue())
      ++Iter;
  }

  // It is safe to clobber EFLAGS at the end of a block of no successor has it
  // live in.
  if (Iter == E) {
    for (MachineBasicBlock::succ_iterator SI = MBB.succ_begin(),
           SE = MBB.succ_end(); SI != SE; ++SI)
      if ((*SI)->isLiveIn(X86::EFLAGS))
        return false;
    return true;
  }

  MachineBasicBlock::iterator B = MBB.begin();
  Iter = I;
  for (unsigned i = 0; i < 4; ++i) {
    // If we make it to the beginning of the block, it's safe to clobber
    // EFLAGS iff EFLAGS is not live-in.
    if (Iter == B)
      return !MBB.isLiveIn(X86::EFLAGS);

    --Iter;
    // Skip over DBG_VALUE.
    while (Iter != B && Iter->isDebugValue())
      --Iter;

    bool SawKill = false;
    for (unsigned j = 0, e = Iter->getNumOperands(); j != e; ++j) {
      MachineOperand &MO = Iter->getOperand(j);
      // A register mask may clobber EFLAGS, but we should still look for a
      // live EFLAGS def.
      if (MO.isRegMask() && MO.clobbersPhysReg(X86::EFLAGS))
        SawKill = true;
      if (MO.isReg() && MO.getReg() == X86::EFLAGS) {
        if (MO.isDef()) return MO.isDead();
        if (MO.isKill()) SawKill = true;
      }
    }

    if (SawKill)
      // This instruction kills EFLAGS and doesn't redefine it, so
      // there's no need to look further.
      return true;
  }

  // Conservative answer.
  return false;
}

void X86InstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I,
                                 unsigned DestReg, unsigned SubIdx,
                                 const MachineInstr *Orig,
                                 const TargetRegisterInfo &TRI) const {
  // MOV32r0 is implemented with a xor which clobbers condition code.
  // Re-materialize it as movri instructions to avoid side effects.
  unsigned Opc = Orig->getOpcode();
  if (Opc == X86::MOV32r0 && !isSafeToClobberEFLAGS(MBB, I)) {
    DebugLoc DL = Orig->getDebugLoc();
    BuildMI(MBB, I, DL, get(X86::MOV32ri)).addOperand(Orig->getOperand(0))
      .addImm(0);
  } else {
    MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
    MBB.insert(I, MI);
  }

  MachineInstr *NewMI = std::prev(I);
  NewMI->substituteRegister(Orig->getOperand(0).getReg(), DestReg, SubIdx, TRI);
}

/// True if MI has a condition code def, e.g. EFLAGS, that is not marked dead.
bool X86InstrInfo::hasLiveCondCodeDef(MachineInstr *MI) const {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        MO.getReg() == X86::EFLAGS && !MO.isDead()) {
      return true;
    }
  }
  return false;
}

/// Check whether the shift count for a machine operand is non-zero.
inline static unsigned getTruncatedShiftCount(MachineInstr *MI,
                                              unsigned ShiftAmtOperandIdx) {
  // The shift count is six bits with the REX.W prefix and five bits without.
  unsigned ShiftCountMask = (MI->getDesc().TSFlags & X86II::REX_W) ? 63 : 31;
  unsigned Imm = MI->getOperand(ShiftAmtOperandIdx).getImm();
  return Imm & ShiftCountMask;
}

/// Check whether the given shift count is appropriate
/// can be represented by a LEA instruction.
inline static bool isTruncatedShiftCountForLEA(unsigned ShAmt) {
  // Left shift instructions can be transformed into load-effective-address
  // instructions if we can encode them appropriately.
  // A LEA instruction utilizes a SIB byte to encode its scale factor.
  // The SIB.scale field is two bits wide which means that we can encode any
  // shift amount less than 4.
  return ShAmt < 4 && ShAmt > 0;
}

bool X86InstrInfo::classifyLEAReg(MachineInstr *MI, const MachineOperand &Src,
                                  unsigned Opc, bool AllowSP,
                                  unsigned &NewSrc, bool &isKill, bool &isUndef,
                                  MachineOperand &ImplicitOp) const {
  MachineFunction &MF = *MI->getParent()->getParent();
  const TargetRegisterClass *RC;
  if (AllowSP) {
    RC = Opc != X86::LEA32r ? &X86::GR64RegClass : &X86::GR32RegClass;
  } else {
    RC = Opc != X86::LEA32r ?
      &X86::GR64_NOSPRegClass : &X86::GR32_NOSPRegClass;
  }
  unsigned SrcReg = Src.getReg();

  // For both LEA64 and LEA32 the register already has essentially the right
  // type (32-bit or 64-bit) we may just need to forbid SP.
  if (Opc != X86::LEA64_32r) {
    NewSrc = SrcReg;
    isKill = Src.isKill();
    isUndef = Src.isUndef();

    if (TargetRegisterInfo::isVirtualRegister(NewSrc) &&
        !MF.getRegInfo().constrainRegClass(NewSrc, RC))
      return false;

    return true;
  }

  // This is for an LEA64_32r and incoming registers are 32-bit. One way or
  // another we need to add 64-bit registers to the final MI.
  if (TargetRegisterInfo::isPhysicalRegister(SrcReg)) {
    ImplicitOp = Src;
    ImplicitOp.setImplicit();

    NewSrc = getX86SubSuperRegister(Src.getReg(), MVT::i64);
    MachineBasicBlock::LivenessQueryResult LQR =
      MI->getParent()->computeRegisterLiveness(&getRegisterInfo(), NewSrc, MI);

    switch (LQR) {
    case MachineBasicBlock::LQR_Unknown:
      // We can't give sane liveness flags to the instruction, abandon LEA
      // formation.
      return false;
    case MachineBasicBlock::LQR_Live:
      isKill = MI->killsRegister(SrcReg);
      isUndef = false;
      break;
    default:
      // The physreg itself is dead, so we have to use it as an <undef>.
      isKill = false;
      isUndef = true;
      break;
    }
  } else {
    // Virtual register of the wrong class, we have to create a temporary 64-bit
    // vreg to feed into the LEA.
    NewSrc = MF.getRegInfo().createVirtualRegister(RC);
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
            get(TargetOpcode::COPY))
      .addReg(NewSrc, RegState::Define | RegState::Undef, X86::sub_32bit)
        .addOperand(Src);

    // Which is obviously going to be dead after we're done with it.
    isKill = true;
    isUndef = false;
  }

  // We've set all the parameters without issue.
  return true;
}

/// Helper for convertToThreeAddress when 16-bit LEA is disabled, use 32-bit
/// LEA to form 3-address code by promoting to a 32-bit superregister and then
/// truncating back down to a 16-bit subregister.
MachineInstr *
X86InstrInfo::convertToThreeAddressWithLEA(unsigned MIOpc,
                                           MachineFunction::iterator &MFI,
                                           MachineBasicBlock::iterator &MBBI,
                                           LiveVariables *LV) const {
  MachineInstr *MI = MBBI;
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();
  bool isDead = MI->getOperand(0).isDead();
  bool isKill = MI->getOperand(1).isKill();

  MachineRegisterInfo &RegInfo = MFI->getParent()->getRegInfo();
  unsigned leaOutReg = RegInfo.createVirtualRegister(&X86::GR32RegClass);
  unsigned Opc, leaInReg;
  if (Subtarget.is64Bit()) {
    Opc = X86::LEA64_32r;
    leaInReg = RegInfo.createVirtualRegister(&X86::GR64_NOSPRegClass);
  } else {
    Opc = X86::LEA32r;
    leaInReg = RegInfo.createVirtualRegister(&X86::GR32_NOSPRegClass);
  }

  // Build and insert into an implicit UNDEF value. This is OK because
  // well be shifting and then extracting the lower 16-bits.
  // This has the potential to cause partial register stall. e.g.
  //   movw    (%rbp,%rcx,2), %dx
  //   leal    -65(%rdx), %esi
  // But testing has shown this *does* help performance in 64-bit mode (at
  // least on modern x86 machines).
  BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(X86::IMPLICIT_DEF), leaInReg);
  MachineInstr *InsMI =
    BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(TargetOpcode::COPY))
    .addReg(leaInReg, RegState::Define, X86::sub_16bit)
    .addReg(Src, getKillRegState(isKill));

  MachineInstrBuilder MIB = BuildMI(*MFI, MBBI, MI->getDebugLoc(),
                                    get(Opc), leaOutReg);
  switch (MIOpc) {
  default: llvm_unreachable("Unreachable!");
  case X86::SHL16ri: {
    unsigned ShAmt = MI->getOperand(2).getImm();
    MIB.addReg(0).addImm(1 << ShAmt)
       .addReg(leaInReg, RegState::Kill).addImm(0).addReg(0);
    break;
  }
  case X86::INC16r:
    addRegOffset(MIB, leaInReg, true, 1);
    break;
  case X86::DEC16r:
    addRegOffset(MIB, leaInReg, true, -1);
    break;
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri_DB:
  case X86::ADD16ri8_DB:
    addRegOffset(MIB, leaInReg, true, MI->getOperand(2).getImm());
    break;
  case X86::ADD16rr:
  case X86::ADD16rr_DB: {
    unsigned Src2 = MI->getOperand(2).getReg();
    bool isKill2 = MI->getOperand(2).isKill();
    unsigned leaInReg2 = 0;
    MachineInstr *InsMI2 = nullptr;
    if (Src == Src2) {
      // ADD16rr %reg1028<kill>, %reg1028
      // just a single insert_subreg.
      addRegReg(MIB, leaInReg, true, leaInReg, false);
    } else {
      if (Subtarget.is64Bit())
        leaInReg2 = RegInfo.createVirtualRegister(&X86::GR64_NOSPRegClass);
      else
        leaInReg2 = RegInfo.createVirtualRegister(&X86::GR32_NOSPRegClass);
      // Build and insert into an implicit UNDEF value. This is OK because
      // well be shifting and then extracting the lower 16-bits.
      BuildMI(*MFI, &*MIB, MI->getDebugLoc(), get(X86::IMPLICIT_DEF),leaInReg2);
      InsMI2 =
        BuildMI(*MFI, &*MIB, MI->getDebugLoc(), get(TargetOpcode::COPY))
        .addReg(leaInReg2, RegState::Define, X86::sub_16bit)
        .addReg(Src2, getKillRegState(isKill2));
      addRegReg(MIB, leaInReg, true, leaInReg2, true);
    }
    if (LV && isKill2 && InsMI2)
      LV->replaceKillInstruction(Src2, MI, InsMI2);
    break;
  }
  }

  MachineInstr *NewMI = MIB;
  MachineInstr *ExtMI =
    BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(TargetOpcode::COPY))
    .addReg(Dest, RegState::Define | getDeadRegState(isDead))
    .addReg(leaOutReg, RegState::Kill, X86::sub_16bit);

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
}

/// This method must be implemented by targets that
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

  // The following opcodes also sets the condition code register(s). Only
  // convert them to equivalent lea if the condition code register def's
  // are dead!
  if (hasLiveCondCodeDef(MI))
    return nullptr;

  MachineFunction &MF = *MI->getParent()->getParent();
  // All instructions input are two-addr instructions.  Get the known operands.
  const MachineOperand &Dest = MI->getOperand(0);
  const MachineOperand &Src = MI->getOperand(1);

  MachineInstr *NewMI = nullptr;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  // 16-bit LEA is also slow on Core2.
  bool DisableLEA16 = true;
  bool is64Bit = Subtarget.is64Bit();

  unsigned MIOpc = MI->getOpcode();
  switch (MIOpc) {
  default: return nullptr;
  case X86::SHL64ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return nullptr;

    // LEA can't handle RSP.
    if (TargetRegisterInfo::isVirtualRegister(Src.getReg()) &&
        !MF.getRegInfo().constrainRegClass(Src.getReg(),
                                           &X86::GR64_NOSPRegClass))
      return nullptr;

    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
      .addOperand(Dest)
      .addReg(0).addImm(1 << ShAmt).addOperand(Src).addImm(0).addReg(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return nullptr;

    unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    // LEA can't handle ESP.
    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
      .addOperand(Dest)
      .addReg(0).addImm(1 << ShAmt)
      .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef))
      .addImm(0).addReg(0);
    if (ImplicitOp.getReg() != 0)
      MIB.addOperand(ImplicitOp);
    NewMI = MIB;

    break;
  }
  case X86::SHL16ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return nullptr;

    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : nullptr;
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
      .addOperand(Dest)
      .addReg(0).addImm(1 << ShAmt).addOperand(Src).addImm(0).addReg(0);
    break;
  }
  case X86::INC64r:
  case X86::INC32r: {
    assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
    unsigned Opc = MIOpc == X86::INC64r ? X86::LEA64r
      : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
        .addOperand(Dest)
        .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef));
    if (ImplicitOp.getReg() != 0)
      MIB.addOperand(ImplicitOp);

    NewMI = addOffset(MIB, 1);
    break;
  }
  case X86::INC16r:
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV)
                     : nullptr;
    assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
    NewMI = addOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                      .addOperand(Dest).addOperand(Src), 1);
    break;
  case X86::DEC64r:
  case X86::DEC32r: {
    assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
    unsigned Opc = MIOpc == X86::DEC64r ? X86::LEA64r
      : (is64Bit ? X86::LEA64_32r : X86::LEA32r);

    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
        .addOperand(Dest)
        .addReg(SrcReg, getUndefRegState(isUndef) | getKillRegState(isKill));
    if (ImplicitOp.getReg() != 0)
      MIB.addOperand(ImplicitOp);

    NewMI = addOffset(MIB, -1);

    break;
  }
  case X86::DEC16r:
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV)
                     : nullptr;
    assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
    NewMI = addOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                      .addOperand(Dest).addOperand(Src), -1);
    break;
  case X86::ADD64rr:
  case X86::ADD64rr_DB:
  case X86::ADD32rr:
  case X86::ADD32rr_DB: {
    assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
    unsigned Opc;
    if (MIOpc == X86::ADD64rr || MIOpc == X86::ADD64rr_DB)
      Opc = X86::LEA64r;
    else
      Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ true,
                        SrcReg, isKill, isUndef, ImplicitOp))
      return nullptr;

    const MachineOperand &Src2 = MI->getOperand(2);
    bool isKill2, isUndef2;
    unsigned SrcReg2;
    MachineOperand ImplicitOp2 = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src2, Opc, /*AllowSP=*/ false,
                        SrcReg2, isKill2, isUndef2, ImplicitOp2))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
      .addOperand(Dest);
    if (ImplicitOp.getReg() != 0)
      MIB.addOperand(ImplicitOp);
    if (ImplicitOp2.getReg() != 0)
      MIB.addOperand(ImplicitOp2);

    NewMI = addRegReg(MIB, SrcReg, isKill, SrcReg2, isKill2);

    // Preserve undefness of the operands.
    NewMI->getOperand(1).setIsUndef(isUndef);
    NewMI->getOperand(3).setIsUndef(isUndef2);

    if (LV && Src2.isKill())
      LV->replaceKillInstruction(SrcReg2, MI, NewMI);
    break;
  }
  case X86::ADD16rr:
  case X86::ADD16rr_DB: {
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV)
                     : nullptr;
    assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
    unsigned Src2 = MI->getOperand(2).getReg();
    bool isKill2 = MI->getOperand(2).isKill();
    NewMI = addRegReg(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                      .addOperand(Dest),
                      Src.getReg(), Src.isKill(), Src2, isKill2);

    // Preserve undefness of the operands.
    bool isUndef = MI->getOperand(1).isUndef();
    bool isUndef2 = MI->getOperand(2).isUndef();
    NewMI->getOperand(1).setIsUndef(isUndef);
    NewMI->getOperand(3).setIsUndef(isUndef2);

    if (LV && isKill2)
      LV->replaceKillInstruction(Src2, MI, NewMI);
    break;
  }
  case X86::ADD64ri32:
  case X86::ADD64ri8:
  case X86::ADD64ri32_DB:
  case X86::ADD64ri8_DB:
    assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
    NewMI = addOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
                      .addOperand(Dest).addOperand(Src),
                      MI->getOperand(2).getImm());
    break;
  case X86::ADD32ri:
  case X86::ADD32ri8:
  case X86::ADD32ri_DB:
  case X86::ADD32ri8_DB: {
    assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
    unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ true,
                        SrcReg, isKill, isUndef, ImplicitOp))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
        .addOperand(Dest)
        .addReg(SrcReg, getUndefRegState(isUndef) | getKillRegState(isKill));
    if (ImplicitOp.getReg() != 0)
      MIB.addOperand(ImplicitOp);

    NewMI = addOffset(MIB, MI->getOperand(2).getImm());
    break;
  }
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri_DB:
  case X86::ADD16ri8_DB:
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV)
                     : nullptr;
    assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
    NewMI = addOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                      .addOperand(Dest).addOperand(Src),
                      MI->getOperand(2).getImm());
    break;
  }

  if (!NewMI) return nullptr;

  if (LV) {  // Update live variables
    if (Src.isKill())
      LV->replaceKillInstruction(Src.getReg(), MI, NewMI);
    if (Dest.isDead())
      LV->replaceKillInstruction(Dest.getReg(), MI, NewMI);
  }

  MFI->insert(MBBI, NewMI);          // Insert the new inst
  return NewMI;
}

/// Returns true if the given instruction opcode is FMA3.
/// Otherwise, returns false.
/// The second parameter is optional and is used as the second return from
/// the function. It is set to true if the given instruction has FMA3 opcode
/// that is used for lowering of scalar FMA intrinsics, and it is set to false
/// otherwise.
static bool isFMA3(unsigned Opcode, bool *IsIntrinsic = nullptr) {
  if (IsIntrinsic)
    *IsIntrinsic = false;

  switch (Opcode) {
    case X86::VFMADDSDr132r:      case X86::VFMADDSDr132m:
    case X86::VFMADDSSr132r:      case X86::VFMADDSSr132m:
    case X86::VFMSUBSDr132r:      case X86::VFMSUBSDr132m:
    case X86::VFMSUBSSr132r:      case X86::VFMSUBSSr132m:
    case X86::VFNMADDSDr132r:     case X86::VFNMADDSDr132m:
    case X86::VFNMADDSSr132r:     case X86::VFNMADDSSr132m:
    case X86::VFNMSUBSDr132r:     case X86::VFNMSUBSDr132m:
    case X86::VFNMSUBSSr132r:     case X86::VFNMSUBSSr132m:

    case X86::VFMADDSDr213r:      case X86::VFMADDSDr213m:
    case X86::VFMADDSSr213r:      case X86::VFMADDSSr213m:
    case X86::VFMSUBSDr213r:      case X86::VFMSUBSDr213m:
    case X86::VFMSUBSSr213r:      case X86::VFMSUBSSr213m:
    case X86::VFNMADDSDr213r:     case X86::VFNMADDSDr213m:
    case X86::VFNMADDSSr213r:     case X86::VFNMADDSSr213m:
    case X86::VFNMSUBSDr213r:     case X86::VFNMSUBSDr213m:
    case X86::VFNMSUBSSr213r:     case X86::VFNMSUBSSr213m:

    case X86::VFMADDSDr231r:      case X86::VFMADDSDr231m:
    case X86::VFMADDSSr231r:      case X86::VFMADDSSr231m:
    case X86::VFMSUBSDr231r:      case X86::VFMSUBSDr231m:
    case X86::VFMSUBSSr231r:      case X86::VFMSUBSSr231m:
    case X86::VFNMADDSDr231r:     case X86::VFNMADDSDr231m:
    case X86::VFNMADDSSr231r:     case X86::VFNMADDSSr231m:
    case X86::VFNMSUBSDr231r:     case X86::VFNMSUBSDr231m:
    case X86::VFNMSUBSSr231r:     case X86::VFNMSUBSSr231m:

    case X86::VFMADDSUBPDr132r:   case X86::VFMADDSUBPDr132m:
    case X86::VFMADDSUBPSr132r:   case X86::VFMADDSUBPSr132m:
    case X86::VFMSUBADDPDr132r:   case X86::VFMSUBADDPDr132m:
    case X86::VFMSUBADDPSr132r:   case X86::VFMSUBADDPSr132m:
    case X86::VFMADDSUBPDr132rY:  case X86::VFMADDSUBPDr132mY:
    case X86::VFMADDSUBPSr132rY:  case X86::VFMADDSUBPSr132mY:
    case X86::VFMSUBADDPDr132rY:  case X86::VFMSUBADDPDr132mY:
    case X86::VFMSUBADDPSr132rY:  case X86::VFMSUBADDPSr132mY:

    case X86::VFMADDPDr132r:      case X86::VFMADDPDr132m:
    case X86::VFMADDPSr132r:      case X86::VFMADDPSr132m:
    case X86::VFMSUBPDr132r:      case X86::VFMSUBPDr132m:
    case X86::VFMSUBPSr132r:      case X86::VFMSUBPSr132m:
    case X86::VFNMADDPDr132r:     case X86::VFNMADDPDr132m:
    case X86::VFNMADDPSr132r:     case X86::VFNMADDPSr132m:
    case X86::VFNMSUBPDr132r:     case X86::VFNMSUBPDr132m:
    case X86::VFNMSUBPSr132r:     case X86::VFNMSUBPSr132m:
    case X86::VFMADDPDr132rY:     case X86::VFMADDPDr132mY:
    case X86::VFMADDPSr132rY:     case X86::VFMADDPSr132mY:
    case X86::VFMSUBPDr132rY:     case X86::VFMSUBPDr132mY:
    case X86::VFMSUBPSr132rY:     case X86::VFMSUBPSr132mY:
    case X86::VFNMADDPDr132rY:    case X86::VFNMADDPDr132mY:
    case X86::VFNMADDPSr132rY:    case X86::VFNMADDPSr132mY:
    case X86::VFNMSUBPDr132rY:    case X86::VFNMSUBPDr132mY:
    case X86::VFNMSUBPSr132rY:    case X86::VFNMSUBPSr132mY:

    case X86::VFMADDSUBPDr213r:   case X86::VFMADDSUBPDr213m:
    case X86::VFMADDSUBPSr213r:   case X86::VFMADDSUBPSr213m:
    case X86::VFMSUBADDPDr213r:   case X86::VFMSUBADDPDr213m:
    case X86::VFMSUBADDPSr213r:   case X86::VFMSUBADDPSr213m:
    case X86::VFMADDSUBPDr213rY:  case X86::VFMADDSUBPDr213mY:
    case X86::VFMADDSUBPSr213rY:  case X86::VFMADDSUBPSr213mY:
    case X86::VFMSUBADDPDr213rY:  case X86::VFMSUBADDPDr213mY:
    case X86::VFMSUBADDPSr213rY:  case X86::VFMSUBADDPSr213mY:

    case X86::VFMADDPDr213r:      case X86::VFMADDPDr213m:
    case X86::VFMADDPSr213r:      case X86::VFMADDPSr213m:
    case X86::VFMSUBPDr213r:      case X86::VFMSUBPDr213m:
    case X86::VFMSUBPSr213r:      case X86::VFMSUBPSr213m:
    case X86::VFNMADDPDr213r:     case X86::VFNMADDPDr213m:
    case X86::VFNMADDPSr213r:     case X86::VFNMADDPSr213m:
    case X86::VFNMSUBPDr213r:     case X86::VFNMSUBPDr213m:
    case X86::VFNMSUBPSr213r:     case X86::VFNMSUBPSr213m:
    case X86::VFMADDPDr213rY:     case X86::VFMADDPDr213mY:
    case X86::VFMADDPSr213rY:     case X86::VFMADDPSr213mY:
    case X86::VFMSUBPDr213rY:     case X86::VFMSUBPDr213mY:
    case X86::VFMSUBPSr213rY:     case X86::VFMSUBPSr213mY:
    case X86::VFNMADDPDr213rY:    case X86::VFNMADDPDr213mY:
    case X86::VFNMADDPSr213rY:    case X86::VFNMADDPSr213mY:
    case X86::VFNMSUBPDr213rY:    case X86::VFNMSUBPDr213mY:
    case X86::VFNMSUBPSr213rY:    case X86::VFNMSUBPSr213mY:

    case X86::VFMADDSUBPDr231r:   case X86::VFMADDSUBPDr231m:
    case X86::VFMADDSUBPSr231r:   case X86::VFMADDSUBPSr231m:
    case X86::VFMSUBADDPDr231r:   case X86::VFMSUBADDPDr231m:
    case X86::VFMSUBADDPSr231r:   case X86::VFMSUBADDPSr231m:
    case X86::VFMADDSUBPDr231rY:  case X86::VFMADDSUBPDr231mY:
    case X86::VFMADDSUBPSr231rY:  case X86::VFMADDSUBPSr231mY:
    case X86::VFMSUBADDPDr231rY:  case X86::VFMSUBADDPDr231mY:
    case X86::VFMSUBADDPSr231rY:  case X86::VFMSUBADDPSr231mY:

    case X86::VFMADDPDr231r:      case X86::VFMADDPDr231m:
    case X86::VFMADDPSr231r:      case X86::VFMADDPSr231m:
    case X86::VFMSUBPDr231r:      case X86::VFMSUBPDr231m:
    case X86::VFMSUBPSr231r:      case X86::VFMSUBPSr231m:
    case X86::VFNMADDPDr231r:     case X86::VFNMADDPDr231m:
    case X86::VFNMADDPSr231r:     case X86::VFNMADDPSr231m:
    case X86::VFNMSUBPDr231r:     case X86::VFNMSUBPDr231m:
    case X86::VFNMSUBPSr231r:     case X86::VFNMSUBPSr231m:
    case X86::VFMADDPDr231rY:     case X86::VFMADDPDr231mY:
    case X86::VFMADDPSr231rY:     case X86::VFMADDPSr231mY:
    case X86::VFMSUBPDr231rY:     case X86::VFMSUBPDr231mY:
    case X86::VFMSUBPSr231rY:     case X86::VFMSUBPSr231mY:
    case X86::VFNMADDPDr231rY:    case X86::VFNMADDPDr231mY:
    case X86::VFNMADDPSr231rY:    case X86::VFNMADDPSr231mY:
    case X86::VFNMSUBPDr231rY:    case X86::VFNMSUBPDr231mY:
    case X86::VFNMSUBPSr231rY:    case X86::VFNMSUBPSr231mY:
      return true;

    case X86::VFMADDSDr132r_Int:  case X86::VFMADDSDr132m_Int:
    case X86::VFMADDSSr132r_Int:  case X86::VFMADDSSr132m_Int:
    case X86::VFMSUBSDr132r_Int:  case X86::VFMSUBSDr132m_Int:
    case X86::VFMSUBSSr132r_Int:  case X86::VFMSUBSSr132m_Int:
    case X86::VFNMADDSDr132r_Int: case X86::VFNMADDSDr132m_Int:
    case X86::VFNMADDSSr132r_Int: case X86::VFNMADDSSr132m_Int:
    case X86::VFNMSUBSDr132r_Int: case X86::VFNMSUBSDr132m_Int:
    case X86::VFNMSUBSSr132r_Int: case X86::VFNMSUBSSr132m_Int:

    case X86::VFMADDSDr213r_Int:  case X86::VFMADDSDr213m_Int:
    case X86::VFMADDSSr213r_Int:  case X86::VFMADDSSr213m_Int:
    case X86::VFMSUBSDr213r_Int:  case X86::VFMSUBSDr213m_Int:
    case X86::VFMSUBSSr213r_Int:  case X86::VFMSUBSSr213m_Int:
    case X86::VFNMADDSDr213r_Int: case X86::VFNMADDSDr213m_Int:
    case X86::VFNMADDSSr213r_Int: case X86::VFNMADDSSr213m_Int:
    case X86::VFNMSUBSDr213r_Int: case X86::VFNMSUBSDr213m_Int:
    case X86::VFNMSUBSSr213r_Int: case X86::VFNMSUBSSr213m_Int:

    case X86::VFMADDSDr231r_Int:  case X86::VFMADDSDr231m_Int:
    case X86::VFMADDSSr231r_Int:  case X86::VFMADDSSr231m_Int:
    case X86::VFMSUBSDr231r_Int:  case X86::VFMSUBSDr231m_Int:
    case X86::VFMSUBSSr231r_Int:  case X86::VFMSUBSSr231m_Int:
    case X86::VFNMADDSDr231r_Int: case X86::VFNMADDSDr231m_Int:
    case X86::VFNMADDSSr231r_Int: case X86::VFNMADDSSr231m_Int:
    case X86::VFNMSUBSDr231r_Int: case X86::VFNMSUBSDr231m_Int:
    case X86::VFNMSUBSSr231r_Int: case X86::VFNMSUBSSr231m_Int:
      if (IsIntrinsic)
        *IsIntrinsic = true;
      return true;
    default:
      return false;
  }
  llvm_unreachable("Opcode not handled by the switch");
}

MachineInstr *X86InstrInfo::commuteInstructionImpl(MachineInstr *MI,
                                                   bool NewMI,
                                                   unsigned OpIdx1,
                                                   unsigned OpIdx2) const {
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
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
  case X86::BLENDPDrri:
  case X86::BLENDPSrri:
  case X86::PBLENDWrri:
  case X86::VBLENDPDrri:
  case X86::VBLENDPSrri:
  case X86::VBLENDPDYrri:
  case X86::VBLENDPSYrri:
  case X86::VPBLENDDrri:
  case X86::VPBLENDWrri:
  case X86::VPBLENDDYrri:
  case X86::VPBLENDWYrri:{
    unsigned Mask;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Unreachable!");
    case X86::BLENDPDrri:    Mask = 0x03; break;
    case X86::BLENDPSrri:    Mask = 0x0F; break;
    case X86::PBLENDWrri:    Mask = 0xFF; break;
    case X86::VBLENDPDrri:   Mask = 0x03; break;
    case X86::VBLENDPSrri:   Mask = 0x0F; break;
    case X86::VBLENDPDYrri:  Mask = 0x0F; break;
    case X86::VBLENDPSYrri:  Mask = 0xFF; break;
    case X86::VPBLENDDrri:   Mask = 0x0F; break;
    case X86::VPBLENDWrri:   Mask = 0xFF; break;
    case X86::VPBLENDDYrri:  Mask = 0xFF; break;
    case X86::VPBLENDWYrri:  Mask = 0xFF; break;
    }
    // Only the least significant bits of Imm are used.
    unsigned Imm = MI->getOperand(3).getImm() & Mask;
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->getOperand(3).setImm(Mask ^ Imm);
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
  case X86::PCLMULQDQrr:
  case X86::VPCLMULQDQrr:{
    // SRC1 64bits = Imm[0] ? SRC1[127:64] : SRC1[63:0]
    // SRC2 64bits = Imm[4] ? SRC2[127:64] : SRC2[63:0]
    unsigned Imm = MI->getOperand(3).getImm();
    unsigned Src1Hi = Imm & 0x01;
    unsigned Src2Hi = Imm & 0x10;
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->getOperand(3).setImm((Src1Hi << 4) | (Src2Hi >> 4));
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
  case X86::CMPPDrri:
  case X86::CMPPSrri:
  case X86::VCMPPDrri:
  case X86::VCMPPSrri:
  case X86::VCMPPDYrri:
  case X86::VCMPPSYrri: {
    // Float comparison can be safely commuted for
    // Ordered/Unordered/Equal/NotEqual tests
    unsigned Imm = MI->getOperand(3).getImm() & 0x7;
    switch (Imm) {
    case 0x00: // EQUAL
    case 0x03: // UNORDERED
    case 0x04: // NOT EQUAL
    case 0x07: // ORDERED
      if (NewMI) {
        MachineFunction &MF = *MI->getParent()->getParent();
        MI = MF.CloneMachineInstr(MI);
        NewMI = false;
      }
      return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
    default:
      return nullptr;
    }
  }
  case X86::VPCOMBri: case X86::VPCOMUBri:
  case X86::VPCOMDri: case X86::VPCOMUDri:
  case X86::VPCOMQri: case X86::VPCOMUQri:
  case X86::VPCOMWri: case X86::VPCOMUWri: {
    // Flip comparison mode immediate (if necessary).
    unsigned Imm = MI->getOperand(3).getImm() & 0x7;
    switch (Imm) {
    case 0x00: Imm = 0x02; break; // LT -> GT
    case 0x01: Imm = 0x03; break; // LE -> GE
    case 0x02: Imm = 0x00; break; // GT -> LT
    case 0x03: Imm = 0x01; break; // GE -> LE
    case 0x04: // EQ
    case 0x05: // NE
    case 0x06: // FALSE
    case 0x07: // TRUE
    default:
      break;
    }
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->getOperand(3).setImm(Imm);
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
  case X86::CMOVB16rr:  case X86::CMOVB32rr:  case X86::CMOVB64rr:
  case X86::CMOVAE16rr: case X86::CMOVAE32rr: case X86::CMOVAE64rr:
  case X86::CMOVE16rr:  case X86::CMOVE32rr:  case X86::CMOVE64rr:
  case X86::CMOVNE16rr: case X86::CMOVNE32rr: case X86::CMOVNE64rr:
  case X86::CMOVBE16rr: case X86::CMOVBE32rr: case X86::CMOVBE64rr:
  case X86::CMOVA16rr:  case X86::CMOVA32rr:  case X86::CMOVA64rr:
  case X86::CMOVL16rr:  case X86::CMOVL32rr:  case X86::CMOVL64rr:
  case X86::CMOVGE16rr: case X86::CMOVGE32rr: case X86::CMOVGE64rr:
  case X86::CMOVLE16rr: case X86::CMOVLE32rr: case X86::CMOVLE64rr:
  case X86::CMOVG16rr:  case X86::CMOVG32rr:  case X86::CMOVG64rr:
  case X86::CMOVS16rr:  case X86::CMOVS32rr:  case X86::CMOVS64rr:
  case X86::CMOVNS16rr: case X86::CMOVNS32rr: case X86::CMOVNS64rr:
  case X86::CMOVP16rr:  case X86::CMOVP32rr:  case X86::CMOVP64rr:
  case X86::CMOVNP16rr: case X86::CMOVNP32rr: case X86::CMOVNP64rr:
  case X86::CMOVO16rr:  case X86::CMOVO32rr:  case X86::CMOVO64rr:
  case X86::CMOVNO16rr: case X86::CMOVNO32rr: case X86::CMOVNO64rr: {
    unsigned Opc;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Unreachable!");
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
    if (isFMA3(MI->getOpcode())) {
      unsigned Opc = getFMA3OpcodeToCommuteOperands(MI, OpIdx1, OpIdx2);
      if (Opc == 0)
        return nullptr;
      if (NewMI) {
        MachineFunction &MF = *MI->getParent()->getParent();
        MI = MF.CloneMachineInstr(MI);
        NewMI = false;
      }
      MI->setDesc(get(Opc));
    }
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
}

bool X86InstrInfo::findFMA3CommutedOpIndices(MachineInstr *MI,
                                             unsigned &SrcOpIdx1,
                                             unsigned &SrcOpIdx2) const {

  unsigned RegOpsNum = isMem(MI, 3) ? 2 : 3;

  // Only the first RegOpsNum operands are commutable.
  // Also, the value 'CommuteAnyOperandIndex' is valid here as it means
  // that the operand is not specified/fixed.
  if (SrcOpIdx1 != CommuteAnyOperandIndex &&
      (SrcOpIdx1 < 1 || SrcOpIdx1 > RegOpsNum))
    return false;
  if (SrcOpIdx2 != CommuteAnyOperandIndex &&
      (SrcOpIdx2 < 1 || SrcOpIdx2 > RegOpsNum))
    return false;

  // Look for two different register operands assumed to be commutable
  // regardless of the FMA opcode. The FMA opcode is adjusted later.
  if (SrcOpIdx1 == CommuteAnyOperandIndex ||
      SrcOpIdx2 == CommuteAnyOperandIndex) {
    unsigned CommutableOpIdx1 = SrcOpIdx1;
    unsigned CommutableOpIdx2 = SrcOpIdx2;

    // At least one of operands to be commuted is not specified and
    // this method is free to choose appropriate commutable operands.
    if (SrcOpIdx1 == SrcOpIdx2)
      // Both of operands are not fixed. By default set one of commutable
      // operands to the last register operand of the instruction.
      CommutableOpIdx2 = RegOpsNum;
    else if (SrcOpIdx2 == CommuteAnyOperandIndex)
      // Only one of operands is not fixed.
      CommutableOpIdx2 = SrcOpIdx1;

    // CommutableOpIdx2 is well defined now. Let's choose another commutable
    // operand and assign its index to CommutableOpIdx1.
    unsigned Op2Reg = MI->getOperand(CommutableOpIdx2).getReg();
    for (CommutableOpIdx1 = RegOpsNum; CommutableOpIdx1 > 0; CommutableOpIdx1--) {
      // The commuted operands must have different registers.
      // Otherwise, the commute transformation does not change anything and
      // is useless then.
      if (Op2Reg != MI->getOperand(CommutableOpIdx1).getReg())
        break;
    }

    // No appropriate commutable operands were found.
    if (CommutableOpIdx1 == 0)
      return false;

    // Assign the found pair of commutable indices to SrcOpIdx1 and SrcOpidx2
    // to return those values.
    if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2,
                              CommutableOpIdx1, CommutableOpIdx2))
      return false;
  }

  // Check if we can adjust the opcode to preserve the semantics when
  // commute the register operands.
  return getFMA3OpcodeToCommuteOperands(MI, SrcOpIdx1, SrcOpIdx2) != 0;
}

unsigned X86InstrInfo::getFMA3OpcodeToCommuteOperands(MachineInstr *MI,
                                                      unsigned SrcOpIdx1,
                                                      unsigned SrcOpIdx2) const {
  unsigned Opc = MI->getOpcode();

  // Define the array that holds FMA opcodes in groups
  // of 3 opcodes(132, 213, 231) in each group.
  static const unsigned RegularOpcodeGroups[][3] = {
    { X86::VFMADDSSr132r,   X86::VFMADDSSr213r,   X86::VFMADDSSr231r  },
    { X86::VFMADDSDr132r,   X86::VFMADDSDr213r,   X86::VFMADDSDr231r  },
    { X86::VFMADDPSr132r,   X86::VFMADDPSr213r,   X86::VFMADDPSr231r  },
    { X86::VFMADDPDr132r,   X86::VFMADDPDr213r,   X86::VFMADDPDr231r  },
    { X86::VFMADDPSr132rY,  X86::VFMADDPSr213rY,  X86::VFMADDPSr231rY },
    { X86::VFMADDPDr132rY,  X86::VFMADDPDr213rY,  X86::VFMADDPDr231rY },
    { X86::VFMADDSSr132m,   X86::VFMADDSSr213m,   X86::VFMADDSSr231m  },
    { X86::VFMADDSDr132m,   X86::VFMADDSDr213m,   X86::VFMADDSDr231m  },
    { X86::VFMADDPSr132m,   X86::VFMADDPSr213m,   X86::VFMADDPSr231m  },
    { X86::VFMADDPDr132m,   X86::VFMADDPDr213m,   X86::VFMADDPDr231m  },
    { X86::VFMADDPSr132mY,  X86::VFMADDPSr213mY,  X86::VFMADDPSr231mY },
    { X86::VFMADDPDr132mY,  X86::VFMADDPDr213mY,  X86::VFMADDPDr231mY },

    { X86::VFMSUBSSr132r,   X86::VFMSUBSSr213r,   X86::VFMSUBSSr231r  },
    { X86::VFMSUBSDr132r,   X86::VFMSUBSDr213r,   X86::VFMSUBSDr231r  },
    { X86::VFMSUBPSr132r,   X86::VFMSUBPSr213r,   X86::VFMSUBPSr231r  },
    { X86::VFMSUBPDr132r,   X86::VFMSUBPDr213r,   X86::VFMSUBPDr231r  },
    { X86::VFMSUBPSr132rY,  X86::VFMSUBPSr213rY,  X86::VFMSUBPSr231rY },
    { X86::VFMSUBPDr132rY,  X86::VFMSUBPDr213rY,  X86::VFMSUBPDr231rY },
    { X86::VFMSUBSSr132m,   X86::VFMSUBSSr213m,   X86::VFMSUBSSr231m  },
    { X86::VFMSUBSDr132m,   X86::VFMSUBSDr213m,   X86::VFMSUBSDr231m  },
    { X86::VFMSUBPSr132m,   X86::VFMSUBPSr213m,   X86::VFMSUBPSr231m  },
    { X86::VFMSUBPDr132m,   X86::VFMSUBPDr213m,   X86::VFMSUBPDr231m  },
    { X86::VFMSUBPSr132mY,  X86::VFMSUBPSr213mY,  X86::VFMSUBPSr231mY },
    { X86::VFMSUBPDr132mY,  X86::VFMSUBPDr213mY,  X86::VFMSUBPDr231mY },

    { X86::VFNMADDSSr132r,  X86::VFNMADDSSr213r,  X86::VFNMADDSSr231r  },
    { X86::VFNMADDSDr132r,  X86::VFNMADDSDr213r,  X86::VFNMADDSDr231r  },
    { X86::VFNMADDPSr132r,  X86::VFNMADDPSr213r,  X86::VFNMADDPSr231r  },
    { X86::VFNMADDPDr132r,  X86::VFNMADDPDr213r,  X86::VFNMADDPDr231r  },
    { X86::VFNMADDPSr132rY, X86::VFNMADDPSr213rY, X86::VFNMADDPSr231rY },
    { X86::VFNMADDPDr132rY, X86::VFNMADDPDr213rY, X86::VFNMADDPDr231rY },
    { X86::VFNMADDSSr132m,  X86::VFNMADDSSr213m,  X86::VFNMADDSSr231m  },
    { X86::VFNMADDSDr132m,  X86::VFNMADDSDr213m,  X86::VFNMADDSDr231m  },
    { X86::VFNMADDPSr132m,  X86::VFNMADDPSr213m,  X86::VFNMADDPSr231m  },
    { X86::VFNMADDPDr132m,  X86::VFNMADDPDr213m,  X86::VFNMADDPDr231m  },
    { X86::VFNMADDPSr132mY, X86::VFNMADDPSr213mY, X86::VFNMADDPSr231mY },
    { X86::VFNMADDPDr132mY, X86::VFNMADDPDr213mY, X86::VFNMADDPDr231mY },

    { X86::VFNMSUBSSr132r,  X86::VFNMSUBSSr213r,  X86::VFNMSUBSSr231r  },
    { X86::VFNMSUBSDr132r,  X86::VFNMSUBSDr213r,  X86::VFNMSUBSDr231r  },
    { X86::VFNMSUBPSr132r,  X86::VFNMSUBPSr213r,  X86::VFNMSUBPSr231r  },
    { X86::VFNMSUBPDr132r,  X86::VFNMSUBPDr213r,  X86::VFNMSUBPDr231r  },
    { X86::VFNMSUBPSr132rY, X86::VFNMSUBPSr213rY, X86::VFNMSUBPSr231rY },
    { X86::VFNMSUBPDr132rY, X86::VFNMSUBPDr213rY, X86::VFNMSUBPDr231rY },
    { X86::VFNMSUBSSr132m,  X86::VFNMSUBSSr213m,  X86::VFNMSUBSSr231m  },
    { X86::VFNMSUBSDr132m,  X86::VFNMSUBSDr213m,  X86::VFNMSUBSDr231m  },
    { X86::VFNMSUBPSr132m,  X86::VFNMSUBPSr213m,  X86::VFNMSUBPSr231m  },
    { X86::VFNMSUBPDr132m,  X86::VFNMSUBPDr213m,  X86::VFNMSUBPDr231m  },
    { X86::VFNMSUBPSr132mY, X86::VFNMSUBPSr213mY, X86::VFNMSUBPSr231mY },
    { X86::VFNMSUBPDr132mY, X86::VFNMSUBPDr213mY, X86::VFNMSUBPDr231mY },

    { X86::VFMADDSUBPSr132r,  X86::VFMADDSUBPSr213r,  X86::VFMADDSUBPSr231r  },
    { X86::VFMADDSUBPDr132r,  X86::VFMADDSUBPDr213r,  X86::VFMADDSUBPDr231r  },
    { X86::VFMADDSUBPSr132rY, X86::VFMADDSUBPSr213rY, X86::VFMADDSUBPSr231rY },
    { X86::VFMADDSUBPDr132rY, X86::VFMADDSUBPDr213rY, X86::VFMADDSUBPDr231rY },
    { X86::VFMADDSUBPSr132m,  X86::VFMADDSUBPSr213m,  X86::VFMADDSUBPSr231m  },
    { X86::VFMADDSUBPDr132m,  X86::VFMADDSUBPDr213m,  X86::VFMADDSUBPDr231m  },
    { X86::VFMADDSUBPSr132mY, X86::VFMADDSUBPSr213mY, X86::VFMADDSUBPSr231mY },
    { X86::VFMADDSUBPDr132mY, X86::VFMADDSUBPDr213mY, X86::VFMADDSUBPDr231mY },

    { X86::VFMSUBADDPSr132r,  X86::VFMSUBADDPSr213r,  X86::VFMSUBADDPSr231r  },
    { X86::VFMSUBADDPDr132r,  X86::VFMSUBADDPDr213r,  X86::VFMSUBADDPDr231r  },
    { X86::VFMSUBADDPSr132rY, X86::VFMSUBADDPSr213rY, X86::VFMSUBADDPSr231rY },
    { X86::VFMSUBADDPDr132rY, X86::VFMSUBADDPDr213rY, X86::VFMSUBADDPDr231rY },
    { X86::VFMSUBADDPSr132m,  X86::VFMSUBADDPSr213m,  X86::VFMSUBADDPSr231m  },
    { X86::VFMSUBADDPDr132m,  X86::VFMSUBADDPDr213m,  X86::VFMSUBADDPDr231m  },
    { X86::VFMSUBADDPSr132mY, X86::VFMSUBADDPSr213mY, X86::VFMSUBADDPSr231mY },
    { X86::VFMSUBADDPDr132mY, X86::VFMSUBADDPDr213mY, X86::VFMSUBADDPDr231mY }
  };

  // Define the array that holds FMA*_Int opcodes in groups
  // of 3 opcodes(132, 213, 231) in each group.
  static const unsigned IntrinOpcodeGroups[][3] = {
    { X86::VFMADDSSr132r_Int,  X86::VFMADDSSr213r_Int,  X86::VFMADDSSr231r_Int },
    { X86::VFMADDSDr132r_Int,  X86::VFMADDSDr213r_Int,  X86::VFMADDSDr231r_Int },
    { X86::VFMADDSSr132m_Int,  X86::VFMADDSSr213m_Int,  X86::VFMADDSSr231m_Int },
    { X86::VFMADDSDr132m_Int,  X86::VFMADDSDr213m_Int,  X86::VFMADDSDr231m_Int },

    { X86::VFMSUBSSr132r_Int,  X86::VFMSUBSSr213r_Int,  X86::VFMSUBSSr231r_Int },
    { X86::VFMSUBSDr132r_Int,  X86::VFMSUBSDr213r_Int,  X86::VFMSUBSDr231r_Int },
    { X86::VFMSUBSSr132m_Int,  X86::VFMSUBSSr213m_Int,  X86::VFMSUBSSr231m_Int },
    { X86::VFMSUBSDr132m_Int,  X86::VFMSUBSDr213m_Int,  X86::VFMSUBSDr231m_Int },

    { X86::VFNMADDSSr132r_Int, X86::VFNMADDSSr213r_Int, X86::VFNMADDSSr231r_Int },
    { X86::VFNMADDSDr132r_Int, X86::VFNMADDSDr213r_Int, X86::VFNMADDSDr231r_Int },
    { X86::VFNMADDSSr132m_Int, X86::VFNMADDSSr213m_Int, X86::VFNMADDSSr231m_Int },
    { X86::VFNMADDSDr132m_Int, X86::VFNMADDSDr213m_Int, X86::VFNMADDSDr231m_Int },

    { X86::VFNMSUBSSr132r_Int, X86::VFNMSUBSSr213r_Int, X86::VFNMSUBSSr231r_Int },
    { X86::VFNMSUBSDr132r_Int, X86::VFNMSUBSDr213r_Int, X86::VFNMSUBSDr231r_Int },
    { X86::VFNMSUBSSr132m_Int, X86::VFNMSUBSSr213m_Int, X86::VFNMSUBSSr231m_Int },
    { X86::VFNMSUBSDr132m_Int, X86::VFNMSUBSDr213m_Int, X86::VFNMSUBSDr231m_Int },
  };

  const unsigned Form132Index = 0;
  const unsigned Form213Index = 1;
  const unsigned Form231Index = 2;
  const unsigned FormsNum = 3;

  bool IsIntrinOpcode;
  isFMA3(Opc, &IsIntrinOpcode);

  unsigned GroupsNum;
  const unsigned (*OpcodeGroups)[3];
  if (IsIntrinOpcode) {
    GroupsNum = sizeof(IntrinOpcodeGroups) / sizeof(IntrinOpcodeGroups[0]);
    OpcodeGroups = IntrinOpcodeGroups;
  } else {
    GroupsNum = sizeof(RegularOpcodeGroups) / sizeof(RegularOpcodeGroups[0]);
    OpcodeGroups = RegularOpcodeGroups;
  }

  const unsigned *FoundOpcodesGroup = nullptr;
  unsigned FormIndex;

  // Look for the input opcode in the corresponding opcodes table.
  unsigned GroupIndex = 0;
  for (; GroupIndex < GroupsNum && !FoundOpcodesGroup; GroupIndex++) {
    for (FormIndex = 0; FormIndex < FormsNum; FormIndex++) {
      if (OpcodeGroups[GroupIndex][FormIndex] == Opc) {
        FoundOpcodesGroup = OpcodeGroups[GroupIndex];
        break;
      }
    }
  }

  // The input opcode does not match with any of the opcodes from the tables.
  // The unsupported FMA opcode must be added to one of the two opcode groups
  // defined above.
  assert(FoundOpcodesGroup != nullptr && "Unexpected FMA3 opcode");

  // Put the lowest index to SrcOpIdx1 to simplify the checks below.
  if (SrcOpIdx1 > SrcOpIdx2)
    std::swap(SrcOpIdx1, SrcOpIdx2);

  // TODO: Commuting the 1st operand of FMA*_Int requires some additional
  // analysis. The commute optimization is legal only if all users of FMA*_Int
  // use only the lowest element of the FMA*_Int instruction. Such analysis are
  // not implemented yet. So, just return 0 in that case.
  // When such analysis are available this place will be the right place for
  // calling it.
  if (IsIntrinOpcode && SrcOpIdx1 == 1)
    return 0;

  unsigned Case;
  if (SrcOpIdx1 == 1 && SrcOpIdx2 == 2)
    Case = 0;
  else if (SrcOpIdx1 == 1 && SrcOpIdx2 == 3)
    Case = 1;
  else if (SrcOpIdx1 == 2 && SrcOpIdx2 == 3)
    Case = 2;
  else
    return 0;

  // Define the FMA forms mapping array that helps to map input FMA form
  // to output FMA form to preserve the operation semantics after
  // commuting the operands.
  static const unsigned FormMapping[][3] = {
    // 0: SrcOpIdx1 == 1 && SrcOpIdx2 == 2;
    // FMA132 A, C, b; ==> FMA231 C, A, b;
    // FMA213 B, A, c; ==> FMA213 A, B, c;
    // FMA231 C, A, b; ==> FMA132 A, C, b;
    { Form231Index, Form213Index, Form132Index },
    // 1: SrcOpIdx1 == 1 && SrcOpIdx2 == 3;
    // FMA132 A, c, B; ==> FMA132 B, c, A;
    // FMA213 B, a, C; ==> FMA231 C, a, B;
    // FMA231 C, a, B; ==> FMA213 B, a, C;
    { Form132Index, Form231Index, Form213Index },
    // 2: SrcOpIdx1 == 2 && SrcOpIdx2 == 3;
    // FMA132 a, C, B; ==> FMA213 a, B, C;
    // FMA213 b, A, C; ==> FMA132 b, C, A;
    // FMA231 c, A, B; ==> FMA231 c, B, A;
    { Form213Index, Form132Index, Form231Index }
  };

  // Everything is ready, just adjust the FMA opcode and return it.
  FormIndex = FormMapping[Case][FormIndex];
  return FoundOpcodesGroup[FormIndex];
}

bool X86InstrInfo::findCommutedOpIndices(MachineInstr *MI,
                                         unsigned &SrcOpIdx1,
                                         unsigned &SrcOpIdx2) const {
  switch (MI->getOpcode()) {
    case X86::CMPPDrri:
    case X86::CMPPSrri:
    case X86::VCMPPDrri:
    case X86::VCMPPSrri:
    case X86::VCMPPDYrri:
    case X86::VCMPPSYrri: {
      // Float comparison can be safely commuted for
      // Ordered/Unordered/Equal/NotEqual tests
      unsigned Imm = MI->getOperand(3).getImm() & 0x7;
      switch (Imm) {
        case 0x00: // EQUAL
        case 0x03: // UNORDERED
        case 0x04: // NOT EQUAL
        case 0x07: // ORDERED
          // The indices of the commutable operands are 1 and 2.
          // Assign them to the returned operand indices here.
          return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 1, 2);
      }
      return false;
    }
    default:
      if (isFMA3(MI->getOpcode()))
        return findFMA3CommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
      return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
  }
  return false;
}

static X86::CondCode getCondFromBranchOpc(unsigned BrOpc) {
  switch (BrOpc) {
  default: return X86::COND_INVALID;
  case X86::JE_1:  return X86::COND_E;
  case X86::JNE_1: return X86::COND_NE;
  case X86::JL_1:  return X86::COND_L;
  case X86::JLE_1: return X86::COND_LE;
  case X86::JG_1:  return X86::COND_G;
  case X86::JGE_1: return X86::COND_GE;
  case X86::JB_1:  return X86::COND_B;
  case X86::JBE_1: return X86::COND_BE;
  case X86::JA_1:  return X86::COND_A;
  case X86::JAE_1: return X86::COND_AE;
  case X86::JS_1:  return X86::COND_S;
  case X86::JNS_1: return X86::COND_NS;
  case X86::JP_1:  return X86::COND_P;
  case X86::JNP_1: return X86::COND_NP;
  case X86::JO_1:  return X86::COND_O;
  case X86::JNO_1: return X86::COND_NO;
  }
}

/// Return condition code of a SET opcode.
static X86::CondCode getCondFromSETOpc(unsigned Opc) {
  switch (Opc) {
  default: return X86::COND_INVALID;
  case X86::SETAr:  case X86::SETAm:  return X86::COND_A;
  case X86::SETAEr: case X86::SETAEm: return X86::COND_AE;
  case X86::SETBr:  case X86::SETBm:  return X86::COND_B;
  case X86::SETBEr: case X86::SETBEm: return X86::COND_BE;
  case X86::SETEr:  case X86::SETEm:  return X86::COND_E;
  case X86::SETGr:  case X86::SETGm:  return X86::COND_G;
  case X86::SETGEr: case X86::SETGEm: return X86::COND_GE;
  case X86::SETLr:  case X86::SETLm:  return X86::COND_L;
  case X86::SETLEr: case X86::SETLEm: return X86::COND_LE;
  case X86::SETNEr: case X86::SETNEm: return X86::COND_NE;
  case X86::SETNOr: case X86::SETNOm: return X86::COND_NO;
  case X86::SETNPr: case X86::SETNPm: return X86::COND_NP;
  case X86::SETNSr: case X86::SETNSm: return X86::COND_NS;
  case X86::SETOr:  case X86::SETOm:  return X86::COND_O;
  case X86::SETPr:  case X86::SETPm:  return X86::COND_P;
  case X86::SETSr:  case X86::SETSm:  return X86::COND_S;
  }
}

/// Return condition code of a CMov opcode.
X86::CondCode X86::getCondFromCMovOpc(unsigned Opc) {
  switch (Opc) {
  default: return X86::COND_INVALID;
  case X86::CMOVA16rm:  case X86::CMOVA16rr:  case X86::CMOVA32rm:
  case X86::CMOVA32rr:  case X86::CMOVA64rm:  case X86::CMOVA64rr:
    return X86::COND_A;
  case X86::CMOVAE16rm: case X86::CMOVAE16rr: case X86::CMOVAE32rm:
  case X86::CMOVAE32rr: case X86::CMOVAE64rm: case X86::CMOVAE64rr:
    return X86::COND_AE;
  case X86::CMOVB16rm:  case X86::CMOVB16rr:  case X86::CMOVB32rm:
  case X86::CMOVB32rr:  case X86::CMOVB64rm:  case X86::CMOVB64rr:
    return X86::COND_B;
  case X86::CMOVBE16rm: case X86::CMOVBE16rr: case X86::CMOVBE32rm:
  case X86::CMOVBE32rr: case X86::CMOVBE64rm: case X86::CMOVBE64rr:
    return X86::COND_BE;
  case X86::CMOVE16rm:  case X86::CMOVE16rr:  case X86::CMOVE32rm:
  case X86::CMOVE32rr:  case X86::CMOVE64rm:  case X86::CMOVE64rr:
    return X86::COND_E;
  case X86::CMOVG16rm:  case X86::CMOVG16rr:  case X86::CMOVG32rm:
  case X86::CMOVG32rr:  case X86::CMOVG64rm:  case X86::CMOVG64rr:
    return X86::COND_G;
  case X86::CMOVGE16rm: case X86::CMOVGE16rr: case X86::CMOVGE32rm:
  case X86::CMOVGE32rr: case X86::CMOVGE64rm: case X86::CMOVGE64rr:
    return X86::COND_GE;
  case X86::CMOVL16rm:  case X86::CMOVL16rr:  case X86::CMOVL32rm:
  case X86::CMOVL32rr:  case X86::CMOVL64rm:  case X86::CMOVL64rr:
    return X86::COND_L;
  case X86::CMOVLE16rm: case X86::CMOVLE16rr: case X86::CMOVLE32rm:
  case X86::CMOVLE32rr: case X86::CMOVLE64rm: case X86::CMOVLE64rr:
    return X86::COND_LE;
  case X86::CMOVNE16rm: case X86::CMOVNE16rr: case X86::CMOVNE32rm:
  case X86::CMOVNE32rr: case X86::CMOVNE64rm: case X86::CMOVNE64rr:
    return X86::COND_NE;
  case X86::CMOVNO16rm: case X86::CMOVNO16rr: case X86::CMOVNO32rm:
  case X86::CMOVNO32rr: case X86::CMOVNO64rm: case X86::CMOVNO64rr:
    return X86::COND_NO;
  case X86::CMOVNP16rm: case X86::CMOVNP16rr: case X86::CMOVNP32rm:
  case X86::CMOVNP32rr: case X86::CMOVNP64rm: case X86::CMOVNP64rr:
    return X86::COND_NP;
  case X86::CMOVNS16rm: case X86::CMOVNS16rr: case X86::CMOVNS32rm:
  case X86::CMOVNS32rr: case X86::CMOVNS64rm: case X86::CMOVNS64rr:
    return X86::COND_NS;
  case X86::CMOVO16rm:  case X86::CMOVO16rr:  case X86::CMOVO32rm:
  case X86::CMOVO32rr:  case X86::CMOVO64rm:  case X86::CMOVO64rr:
    return X86::COND_O;
  case X86::CMOVP16rm:  case X86::CMOVP16rr:  case X86::CMOVP32rm:
  case X86::CMOVP32rr:  case X86::CMOVP64rm:  case X86::CMOVP64rr:
    return X86::COND_P;
  case X86::CMOVS16rm:  case X86::CMOVS16rr:  case X86::CMOVS32rm:
  case X86::CMOVS32rr:  case X86::CMOVS64rm:  case X86::CMOVS64rr:
    return X86::COND_S;
  }
}

unsigned X86::GetCondBranchFromCond(X86::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case X86::COND_E:  return X86::JE_1;
  case X86::COND_NE: return X86::JNE_1;
  case X86::COND_L:  return X86::JL_1;
  case X86::COND_LE: return X86::JLE_1;
  case X86::COND_G:  return X86::JG_1;
  case X86::COND_GE: return X86::JGE_1;
  case X86::COND_B:  return X86::JB_1;
  case X86::COND_BE: return X86::JBE_1;
  case X86::COND_A:  return X86::JA_1;
  case X86::COND_AE: return X86::JAE_1;
  case X86::COND_S:  return X86::JS_1;
  case X86::COND_NS: return X86::JNS_1;
  case X86::COND_P:  return X86::JP_1;
  case X86::COND_NP: return X86::JNP_1;
  case X86::COND_O:  return X86::JO_1;
  case X86::COND_NO: return X86::JNO_1;
  }
}

/// Return the inverse of the specified condition,
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

/// Assuming the flags are set by MI(a,b), return the condition code if we
/// modify the instructions such that flags are set by MI(b,a).
static X86::CondCode getSwappedCondition(X86::CondCode CC) {
  switch (CC) {
  default: return X86::COND_INVALID;
  case X86::COND_E:  return X86::COND_E;
  case X86::COND_NE: return X86::COND_NE;
  case X86::COND_L:  return X86::COND_G;
  case X86::COND_LE: return X86::COND_GE;
  case X86::COND_G:  return X86::COND_L;
  case X86::COND_GE: return X86::COND_LE;
  case X86::COND_B:  return X86::COND_A;
  case X86::COND_BE: return X86::COND_AE;
  case X86::COND_A:  return X86::COND_B;
  case X86::COND_AE: return X86::COND_BE;
  }
}

/// Return a set opcode for the given condition and
/// whether it has memory operand.
unsigned X86::getSETFromCond(CondCode CC, bool HasMemoryOperand) {
  static const uint16_t Opc[16][2] = {
    { X86::SETAr,  X86::SETAm  },
    { X86::SETAEr, X86::SETAEm },
    { X86::SETBr,  X86::SETBm  },
    { X86::SETBEr, X86::SETBEm },
    { X86::SETEr,  X86::SETEm  },
    { X86::SETGr,  X86::SETGm  },
    { X86::SETGEr, X86::SETGEm },
    { X86::SETLr,  X86::SETLm  },
    { X86::SETLEr, X86::SETLEm },
    { X86::SETNEr, X86::SETNEm },
    { X86::SETNOr, X86::SETNOm },
    { X86::SETNPr, X86::SETNPm },
    { X86::SETNSr, X86::SETNSm },
    { X86::SETOr,  X86::SETOm  },
    { X86::SETPr,  X86::SETPm  },
    { X86::SETSr,  X86::SETSm  }
  };

  assert(CC <= LAST_VALID_COND && "Can only handle standard cond codes");
  return Opc[CC][HasMemoryOperand ? 1 : 0];
}

/// Return a cmov opcode for the given condition,
/// register size in bytes, and operand type.
unsigned X86::getCMovFromCond(CondCode CC, unsigned RegBytes,
                              bool HasMemoryOperand) {
  static const uint16_t Opc[32][3] = {
    { X86::CMOVA16rr,  X86::CMOVA32rr,  X86::CMOVA64rr  },
    { X86::CMOVAE16rr, X86::CMOVAE32rr, X86::CMOVAE64rr },
    { X86::CMOVB16rr,  X86::CMOVB32rr,  X86::CMOVB64rr  },
    { X86::CMOVBE16rr, X86::CMOVBE32rr, X86::CMOVBE64rr },
    { X86::CMOVE16rr,  X86::CMOVE32rr,  X86::CMOVE64rr  },
    { X86::CMOVG16rr,  X86::CMOVG32rr,  X86::CMOVG64rr  },
    { X86::CMOVGE16rr, X86::CMOVGE32rr, X86::CMOVGE64rr },
    { X86::CMOVL16rr,  X86::CMOVL32rr,  X86::CMOVL64rr  },
    { X86::CMOVLE16rr, X86::CMOVLE32rr, X86::CMOVLE64rr },
    { X86::CMOVNE16rr, X86::CMOVNE32rr, X86::CMOVNE64rr },
    { X86::CMOVNO16rr, X86::CMOVNO32rr, X86::CMOVNO64rr },
    { X86::CMOVNP16rr, X86::CMOVNP32rr, X86::CMOVNP64rr },
    { X86::CMOVNS16rr, X86::CMOVNS32rr, X86::CMOVNS64rr },
    { X86::CMOVO16rr,  X86::CMOVO32rr,  X86::CMOVO64rr  },
    { X86::CMOVP16rr,  X86::CMOVP32rr,  X86::CMOVP64rr  },
    { X86::CMOVS16rr,  X86::CMOVS32rr,  X86::CMOVS64rr  },
    { X86::CMOVA16rm,  X86::CMOVA32rm,  X86::CMOVA64rm  },
    { X86::CMOVAE16rm, X86::CMOVAE32rm, X86::CMOVAE64rm },
    { X86::CMOVB16rm,  X86::CMOVB32rm,  X86::CMOVB64rm  },
    { X86::CMOVBE16rm, X86::CMOVBE32rm, X86::CMOVBE64rm },
    { X86::CMOVE16rm,  X86::CMOVE32rm,  X86::CMOVE64rm  },
    { X86::CMOVG16rm,  X86::CMOVG32rm,  X86::CMOVG64rm  },
    { X86::CMOVGE16rm, X86::CMOVGE32rm, X86::CMOVGE64rm },
    { X86::CMOVL16rm,  X86::CMOVL32rm,  X86::CMOVL64rm  },
    { X86::CMOVLE16rm, X86::CMOVLE32rm, X86::CMOVLE64rm },
    { X86::CMOVNE16rm, X86::CMOVNE32rm, X86::CMOVNE64rm },
    { X86::CMOVNO16rm, X86::CMOVNO32rm, X86::CMOVNO64rm },
    { X86::CMOVNP16rm, X86::CMOVNP32rm, X86::CMOVNP64rm },
    { X86::CMOVNS16rm, X86::CMOVNS32rm, X86::CMOVNS64rm },
    { X86::CMOVO16rm,  X86::CMOVO32rm,  X86::CMOVO64rm  },
    { X86::CMOVP16rm,  X86::CMOVP32rm,  X86::CMOVP64rm  },
    { X86::CMOVS16rm,  X86::CMOVS32rm,  X86::CMOVS64rm  }
  };

  assert(CC < 16 && "Can only handle standard cond codes");
  unsigned Idx = HasMemoryOperand ? 16+CC : CC;
  switch(RegBytes) {
  default: llvm_unreachable("Illegal register size!");
  case 2: return Opc[Idx][0];
  case 4: return Opc[Idx][1];
  case 8: return Opc[Idx][2];
  }
}

bool X86InstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  if (!MI->isTerminator()) return false;

  // Conditional branch is a special case.
  if (MI->isBranch() && !MI->isBarrier())
    return true;
  if (!MI->isPredicable())
    return true;
  return !isPredicated(MI);
}

bool X86InstrInfo::AnalyzeBranchImpl(
    MachineBasicBlock &MBB, MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
    SmallVectorImpl<MachineOperand> &Cond,
    SmallVectorImpl<MachineInstr *> &CondBranches, bool AllowModify) const {

  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  MachineBasicBlock::iterator UnCondBrIter = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;

    // Working from the bottom, when we see a non-terminator instruction, we're
    // done.
    if (!isUnpredicatedTerminator(I))
      break;

    // A terminator that isn't a branch can't easily be handled by this
    // analysis.
    if (!I->isBranch())
      return true;

    // Handle unconditional branches.
    if (I->getOpcode() == X86::JMP_1) {
      UnCondBrIter = I;

      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      while (std::next(I) != MBB.end())
        std::next(I)->eraseFromParent();

      Cond.clear();
      FBB = nullptr;

      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = nullptr;
        I->eraseFromParent();
        I = MBB.end();
        UnCondBrIter = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditional destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }

    // Handle conditional branches.
    X86::CondCode BranchCode = getCondFromBranchOpc(I->getOpcode());
    if (BranchCode == X86::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      MachineBasicBlock *TargetBB = I->getOperand(0).getMBB();
      if (AllowModify && UnCondBrIter != MBB.end() &&
          MBB.isLayoutSuccessor(TargetBB)) {
        // If we can modify the code and it ends in something like:
        //
        //     jCC L1
        //     jmp L2
        //   L1:
        //     ...
        //   L2:
        //
        // Then we can change this to:
        //
        //     jnCC L2
        //   L1:
        //     ...
        //   L2:
        //
        // Which is a bit more efficient.
        // We conditionally jump to the fall-through block.
        BranchCode = GetOppositeBranchCondition(BranchCode);
        unsigned JNCC = GetCondBranchFromCond(BranchCode);
        MachineBasicBlock::iterator OldInst = I;

        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(JNCC))
          .addMBB(UnCondBrIter->getOperand(0).getMBB());
        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(X86::JMP_1))
          .addMBB(TargetBB);

        OldInst->eraseFromParent();
        UnCondBrIter->eraseFromParent();

        // Restart the analysis.
        UnCondBrIter = MBB.end();
        I = MBB.end();
        continue;
      }

      FBB = TBB;
      TBB = I->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));
      CondBranches.push_back(I);
      continue;
    }

    // Handle subsequent conditional branches. Only handle the case where all
    // conditional branches branch to the same destination and their condition
    // opcodes fit one of the special multi-branch idioms.
    assert(Cond.size() == 1);
    assert(TBB);

    // Only handle the case where all conditional branches branch to the same
    // destination.
    if (TBB != I->getOperand(0).getMBB())
      return true;

    // If the conditions are the same, we can leave them alone.
    X86::CondCode OldBranchCode = (X86::CondCode)Cond[0].getImm();
    if (OldBranchCode == BranchCode)
      continue;

    // If they differ, see if they fit one of the known patterns. Theoretically,
    // we could handle more patterns here, but we shouldn't expect to see them
    // if instruction selection has done a reasonable job.
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
    CondBranches.push_back(I);
  }

  return false;
}

bool X86InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  SmallVector<MachineInstr *, 4> CondBranches;
  return AnalyzeBranchImpl(MBB, TBB, FBB, Cond, CondBranches, AllowModify);
}

bool X86InstrInfo::AnalyzeBranchPredicate(MachineBasicBlock &MBB,
                                          MachineBranchPredicate &MBP,
                                          bool AllowModify) const {
  using namespace std::placeholders;

  SmallVector<MachineOperand, 4> Cond;
  SmallVector<MachineInstr *, 4> CondBranches;
  if (AnalyzeBranchImpl(MBB, MBP.TrueDest, MBP.FalseDest, Cond, CondBranches,
                        AllowModify))
    return true;

  if (Cond.size() != 1)
    return true;

  assert(MBP.TrueDest && "expected!");

  if (!MBP.FalseDest)
    MBP.FalseDest = MBB.getNextNode();

  const TargetRegisterInfo *TRI = &getRegisterInfo();

  MachineInstr *ConditionDef = nullptr;
  bool SingleUseCondition = true;

  for (auto I = std::next(MBB.rbegin()), E = MBB.rend(); I != E; ++I) {
    if (I->modifiesRegister(X86::EFLAGS, TRI)) {
      ConditionDef = &*I;
      break;
    }

    if (I->readsRegister(X86::EFLAGS, TRI))
      SingleUseCondition = false;
  }

  if (!ConditionDef)
    return true;

  if (SingleUseCondition) {
    for (auto *Succ : MBB.successors())
      if (Succ->isLiveIn(X86::EFLAGS))
        SingleUseCondition = false;
  }

  MBP.ConditionDef = ConditionDef;
  MBP.SingleUseCondition = SingleUseCondition;

  // Currently we only recognize the simple pattern:
  //
  //   test %reg, %reg
  //   je %label
  //
  const unsigned TestOpcode =
      Subtarget.is64Bit() ? X86::TEST64rr : X86::TEST32rr;

  if (ConditionDef->getOpcode() == TestOpcode &&
      ConditionDef->getNumOperands() == 3 &&
      ConditionDef->getOperand(0).isIdenticalTo(ConditionDef->getOperand(1)) &&
      (Cond[0].getImm() == X86::COND_NE || Cond[0].getImm() == X86::COND_E)) {
    MBP.LHS = ConditionDef->getOperand(0);
    MBP.RHS = MachineOperand::CreateImm(0);
    MBP.Predicate = Cond[0].getImm() == X86::COND_NE
                        ? MachineBranchPredicate::PRED_NE
                        : MachineBranchPredicate::PRED_EQ;
    return false;
  }

  return true;
}

unsigned X86InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (I->getOpcode() != X86::JMP_1 &&
        getCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
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
                           MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                           DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(X86::JMP_1)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  X86::CondCode CC = (X86::CondCode)Cond[0].getImm();
  switch (CC) {
  case X86::COND_NP_OR_E:
    // Synthesize NP_OR_E with two branches.
    BuildMI(&MBB, DL, get(X86::JNP_1)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JE_1)).addMBB(TBB);
    ++Count;
    break;
  case X86::COND_NE_OR_P:
    // Synthesize NE_OR_P with two branches.
    BuildMI(&MBB, DL, get(X86::JNE_1)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JP_1)).addMBB(TBB);
    ++Count;
    break;
  default: {
    unsigned Opc = GetCondBranchFromCond(CC);
    BuildMI(&MBB, DL, get(Opc)).addMBB(TBB);
    ++Count;
  }
  }
  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    BuildMI(&MBB, DL, get(X86::JMP_1)).addMBB(FBB);
    ++Count;
  }
  return Count;
}

bool X86InstrInfo::
canInsertSelect(const MachineBasicBlock &MBB,
                ArrayRef<MachineOperand> Cond,
                unsigned TrueReg, unsigned FalseReg,
                int &CondCycles, int &TrueCycles, int &FalseCycles) const {
  // Not all subtargets have cmov instructions.
  if (!Subtarget.hasCMov())
    return false;
  if (Cond.size() != 1)
    return false;
  // We cannot do the composite conditions, at least not in SSA form.
  if ((X86::CondCode)Cond[0].getImm() > X86::COND_S)
    return false;

  // Check register classes.
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
    RI.getCommonSubClass(MRI.getRegClass(TrueReg), MRI.getRegClass(FalseReg));
  if (!RC)
    return false;

  // We have cmov instructions for 16, 32, and 64 bit general purpose registers.
  if (X86::GR16RegClass.hasSubClassEq(RC) ||
      X86::GR32RegClass.hasSubClassEq(RC) ||
      X86::GR64RegClass.hasSubClassEq(RC)) {
    // This latency applies to Pentium M, Merom, Wolfdale, Nehalem, and Sandy
    // Bridge. Probably Ivy Bridge as well.
    CondCycles = 2;
    TrueCycles = 2;
    FalseCycles = 2;
    return true;
  }

  // Can't do vectors.
  return false;
}

void X86InstrInfo::insertSelect(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I, DebugLoc DL,
                                unsigned DstReg, ArrayRef<MachineOperand> Cond,
                                unsigned TrueReg, unsigned FalseReg) const {
   MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
   assert(Cond.size() == 1 && "Invalid Cond array");
   unsigned Opc = getCMovFromCond((X86::CondCode)Cond[0].getImm(),
                                  MRI.getRegClass(DstReg)->getSize(),
                                  false/*HasMemoryOperand*/);
   BuildMI(MBB, I, DL, get(Opc), DstReg).addReg(FalseReg).addReg(TrueReg);
}

/// Test if the given register is a physical h register.
static bool isHReg(unsigned Reg) {
  return X86::GR8_ABCD_HRegClass.contains(Reg);
}

// Try and copy between VR128/VR64 and GR64 registers.
static unsigned CopyToFromAsymmetricReg(unsigned DestReg, unsigned SrcReg,
                                        const X86Subtarget &Subtarget) {

  // SrcReg(VR128) -> DestReg(GR64)
  // SrcReg(VR64)  -> DestReg(GR64)
  // SrcReg(GR64)  -> DestReg(VR128)
  // SrcReg(GR64)  -> DestReg(VR64)

  bool HasAVX = Subtarget.hasAVX();
  bool HasAVX512 = Subtarget.hasAVX512();
  if (X86::GR64RegClass.contains(DestReg)) {
    if (X86::VR128XRegClass.contains(SrcReg))
      // Copy from a VR128 register to a GR64 register.
      return HasAVX512 ? X86::VMOVPQIto64Zrr: (HasAVX ? X86::VMOVPQIto64rr :
                                               X86::MOVPQIto64rr);
    if (X86::VR64RegClass.contains(SrcReg))
      // Copy from a VR64 register to a GR64 register.
      return X86::MMX_MOVD64from64rr;
  } else if (X86::GR64RegClass.contains(SrcReg)) {
    // Copy from a GR64 register to a VR128 register.
    if (X86::VR128XRegClass.contains(DestReg))
      return HasAVX512 ? X86::VMOV64toPQIZrr: (HasAVX ? X86::VMOV64toPQIrr :
                                               X86::MOV64toPQIrr);
    // Copy from a GR64 register to a VR64 register.
    if (X86::VR64RegClass.contains(DestReg))
      return X86::MMX_MOVD64to64rr;
  }

  // SrcReg(FR32) -> DestReg(GR32)
  // SrcReg(GR32) -> DestReg(FR32)

  if (X86::GR32RegClass.contains(DestReg) && X86::FR32XRegClass.contains(SrcReg))
    // Copy from a FR32 register to a GR32 register.
    return HasAVX512 ? X86::VMOVSS2DIZrr : (HasAVX ? X86::VMOVSS2DIrr : X86::MOVSS2DIrr);

  if (X86::FR32XRegClass.contains(DestReg) && X86::GR32RegClass.contains(SrcReg))
    // Copy from a GR32 register to a FR32 register.
    return HasAVX512 ? X86::VMOVDI2SSZrr : (HasAVX ? X86::VMOVDI2SSrr : X86::MOVDI2SSrr);
  return 0;
}

static bool MaskRegClassContains(unsigned Reg) {
  return X86::VK8RegClass.contains(Reg) ||
         X86::VK16RegClass.contains(Reg) ||
         X86::VK32RegClass.contains(Reg) ||
         X86::VK64RegClass.contains(Reg) ||
         X86::VK1RegClass.contains(Reg);
}

static bool GRRegClassContains(unsigned Reg) {
  return X86::GR64RegClass.contains(Reg) ||
         X86::GR32RegClass.contains(Reg) ||
         X86::GR16RegClass.contains(Reg) ||
         X86::GR8RegClass.contains(Reg);
}
static
unsigned copyPhysRegOpcode_AVX512_DQ(unsigned& DestReg, unsigned& SrcReg) {
  if (MaskRegClassContains(SrcReg) && X86::GR8RegClass.contains(DestReg)) {
    DestReg = getX86SubSuperRegister(DestReg, MVT::i32);
    return X86::KMOVBrk;
  }
  if (MaskRegClassContains(DestReg) && X86::GR8RegClass.contains(SrcReg)) {
    SrcReg = getX86SubSuperRegister(SrcReg, MVT::i32);
    return X86::KMOVBkr;
  }
  return 0;
}

static
unsigned copyPhysRegOpcode_AVX512_BW(unsigned& DestReg, unsigned& SrcReg) {
  if (MaskRegClassContains(SrcReg) && MaskRegClassContains(DestReg))
    return X86::KMOVQkk;
  if (MaskRegClassContains(SrcReg) && X86::GR32RegClass.contains(DestReg))
    return X86::KMOVDrk;
  if (MaskRegClassContains(SrcReg) && X86::GR64RegClass.contains(DestReg))
    return X86::KMOVQrk;
  if (MaskRegClassContains(DestReg) && X86::GR32RegClass.contains(SrcReg))
    return X86::KMOVDkr;
  if (MaskRegClassContains(DestReg) && X86::GR64RegClass.contains(SrcReg))
    return X86::KMOVQkr;
  return 0;
}

static
unsigned copyPhysRegOpcode_AVX512(unsigned& DestReg, unsigned& SrcReg,
                                  const X86Subtarget &Subtarget)
{
  if (Subtarget.hasDQI())
    if (auto Opc = copyPhysRegOpcode_AVX512_DQ(DestReg, SrcReg))
      return Opc;
  if (Subtarget.hasBWI())
    if (auto Opc = copyPhysRegOpcode_AVX512_BW(DestReg, SrcReg))
      return Opc;
  if (X86::VR128XRegClass.contains(DestReg, SrcReg) ||
      X86::VR256XRegClass.contains(DestReg, SrcReg) ||
      X86::VR512RegClass.contains(DestReg, SrcReg)) {
     DestReg = get512BitSuperRegister(DestReg);
     SrcReg = get512BitSuperRegister(SrcReg);
     return X86::VMOVAPSZrr;
  }
  if (MaskRegClassContains(DestReg) && MaskRegClassContains(SrcReg))
    return X86::KMOVWkk;
  if (MaskRegClassContains(DestReg) && GRRegClassContains(SrcReg)) {
    SrcReg = getX86SubSuperRegister(SrcReg, MVT::i32);
    return X86::KMOVWkr;
  }
  if (GRRegClassContains(DestReg) && MaskRegClassContains(SrcReg)) {
    DestReg = getX86SubSuperRegister(DestReg, MVT::i32);
    return X86::KMOVWrk;
  }
  return 0;
}

void X86InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI, DebugLoc DL,
                               unsigned DestReg, unsigned SrcReg,
                               bool KillSrc) const {
  // First deal with the normal symmetric copies.
  bool HasAVX = Subtarget.hasAVX();
  bool HasAVX512 = Subtarget.hasAVX512();
  unsigned Opc = 0;
  if (X86::GR64RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOV64rr;
  else if (X86::GR32RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOV32rr;
  else if (X86::GR16RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOV16rr;
  else if (X86::GR8RegClass.contains(DestReg, SrcReg)) {
    // Copying to or from a physical H register on x86-64 requires a NOREX
    // move.  Otherwise use a normal move.
    if ((isHReg(DestReg) || isHReg(SrcReg)) &&
        Subtarget.is64Bit()) {
      Opc = X86::MOV8rr_NOREX;
      // Both operands must be encodable without an REX prefix.
      assert(X86::GR8_NOREXRegClass.contains(SrcReg, DestReg) &&
             "8-bit H register can not be copied outside GR8_NOREX");
    } else
      Opc = X86::MOV8rr;
  }
  else if (X86::VR64RegClass.contains(DestReg, SrcReg))
    Opc = X86::MMX_MOVQ64rr;
  else if (HasAVX512)
    Opc = copyPhysRegOpcode_AVX512(DestReg, SrcReg, Subtarget);
  else if (X86::VR128RegClass.contains(DestReg, SrcReg))
    Opc = HasAVX ? X86::VMOVAPSrr : X86::MOVAPSrr;
  else if (X86::VR256RegClass.contains(DestReg, SrcReg))
    Opc = X86::VMOVAPSYrr;
  if (!Opc)
    Opc = CopyToFromAsymmetricReg(DestReg, SrcReg, Subtarget);

  if (Opc) {
    BuildMI(MBB, MI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  bool FromEFLAGS = SrcReg == X86::EFLAGS;
  bool ToEFLAGS = DestReg == X86::EFLAGS;
  int Reg = FromEFLAGS ? DestReg : SrcReg;
  bool is32 = X86::GR32RegClass.contains(Reg);
  bool is64 = X86::GR64RegClass.contains(Reg);
  if ((FromEFLAGS || ToEFLAGS) && (is32 || is64)) {
    // The flags need to be saved, but saving EFLAGS with PUSHF/POPF is
    // inefficient. Instead:
    //   - Save the overflow flag OF into AL using SETO, and restore it using a
    //     signed 8-bit addition of AL and INT8_MAX.
    //   - Save/restore the bottom 8 EFLAGS bits (CF, PF, AF, ZF, SF) to/from AH
    //     using LAHF/SAHF.
    //   - When RAX/EAX is live and isn't the destination register, make sure it
    //     isn't clobbered by PUSH/POP'ing it before and after saving/restoring
    //     the flags.
    // This approach is ~2.25x faster than using PUSHF/POPF.
    //
    // This is still somewhat inefficient because we don't know which flags are
    // actually live inside EFLAGS. Were we able to do a single SETcc instead of
    // SETO+LAHF / ADDB+SAHF the code could be 1.02x faster.
    //
    // PUSHF/POPF is also potentially incorrect because it affects other flags
    // such as TF/IF/DF, which LLVM doesn't model.
    //
    // Notice that we have to adjust the stack if we don't want to clobber the
    // first frame index. See X86FrameLowering.cpp - clobbersTheStack.

    int Mov = is64 ? X86::MOV64rr : X86::MOV32rr;
    int Push = is64 ? X86::PUSH64r : X86::PUSH32r;
    int Pop = is64 ? X86::POP64r : X86::POP32r;
    int AX = is64 ? X86::RAX : X86::EAX;

    bool AXDead = (Reg == AX) ||
                  (MachineBasicBlock::LQR_Dead ==
                   MBB.computeRegisterLiveness(&getRegisterInfo(), AX, MI));

    if (!AXDead)
      BuildMI(MBB, MI, DL, get(Push)).addReg(AX, getKillRegState(true));
    if (FromEFLAGS) {
      BuildMI(MBB, MI, DL, get(X86::SETOr), X86::AL);
      BuildMI(MBB, MI, DL, get(X86::LAHF));
      BuildMI(MBB, MI, DL, get(Mov), Reg).addReg(AX);
    }
    if (ToEFLAGS) {
      BuildMI(MBB, MI, DL, get(Mov), AX).addReg(Reg, getKillRegState(KillSrc));
      BuildMI(MBB, MI, DL, get(X86::ADD8ri), X86::AL)
          .addReg(X86::AL)
          .addImm(INT8_MAX);
      BuildMI(MBB, MI, DL, get(X86::SAHF));
    }
    if (!AXDead)
      BuildMI(MBB, MI, DL, get(Pop), AX);
    return;
  }

  DEBUG(dbgs() << "Cannot copy " << RI.getName(SrcReg)
               << " to " << RI.getName(DestReg) << '\n');
  llvm_unreachable("Cannot emit physreg copy instruction");
}

static unsigned getLoadStoreRegOpcode(unsigned Reg,
                                      const TargetRegisterClass *RC,
                                      bool isStackAligned,
                                      const X86Subtarget &STI,
                                      bool load) {
  if (STI.hasAVX512()) {
    if (X86::VK8RegClass.hasSubClassEq(RC)  ||
      X86::VK16RegClass.hasSubClassEq(RC))
      return load ? X86::KMOVWkm : X86::KMOVWmk;
    if (RC->getSize() == 4 && X86::FR32XRegClass.hasSubClassEq(RC))
      return load ? X86::VMOVSSZrm : X86::VMOVSSZmr;
    if (RC->getSize() == 8 && X86::FR64XRegClass.hasSubClassEq(RC))
      return load ? X86::VMOVSDZrm : X86::VMOVSDZmr;
    if (X86::VR512RegClass.hasSubClassEq(RC))
      return load ? X86::VMOVUPSZrm : X86::VMOVUPSZmr;
  }

  bool HasAVX = STI.hasAVX();
  switch (RC->getSize()) {
  default:
    llvm_unreachable("Unknown spill size");
  case 1:
    assert(X86::GR8RegClass.hasSubClassEq(RC) && "Unknown 1-byte regclass");
    if (STI.is64Bit())
      // Copying to or from a physical H register on x86-64 requires a NOREX
      // move.  Otherwise use a normal move.
      if (isHReg(Reg) || X86::GR8_ABCD_HRegClass.hasSubClassEq(RC))
        return load ? X86::MOV8rm_NOREX : X86::MOV8mr_NOREX;
    return load ? X86::MOV8rm : X86::MOV8mr;
  case 2:
    assert(X86::GR16RegClass.hasSubClassEq(RC) && "Unknown 2-byte regclass");
    return load ? X86::MOV16rm : X86::MOV16mr;
  case 4:
    if (X86::GR32RegClass.hasSubClassEq(RC))
      return load ? X86::MOV32rm : X86::MOV32mr;
    if (X86::FR32RegClass.hasSubClassEq(RC))
      return load ?
        (HasAVX ? X86::VMOVSSrm : X86::MOVSSrm) :
        (HasAVX ? X86::VMOVSSmr : X86::MOVSSmr);
    if (X86::RFP32RegClass.hasSubClassEq(RC))
      return load ? X86::LD_Fp32m : X86::ST_Fp32m;
    llvm_unreachable("Unknown 4-byte regclass");
  case 8:
    if (X86::GR64RegClass.hasSubClassEq(RC))
      return load ? X86::MOV64rm : X86::MOV64mr;
    if (X86::FR64RegClass.hasSubClassEq(RC))
      return load ?
        (HasAVX ? X86::VMOVSDrm : X86::MOVSDrm) :
        (HasAVX ? X86::VMOVSDmr : X86::MOVSDmr);
    if (X86::VR64RegClass.hasSubClassEq(RC))
      return load ? X86::MMX_MOVQ64rm : X86::MMX_MOVQ64mr;
    if (X86::RFP64RegClass.hasSubClassEq(RC))
      return load ? X86::LD_Fp64m : X86::ST_Fp64m;
    llvm_unreachable("Unknown 8-byte regclass");
  case 10:
    assert(X86::RFP80RegClass.hasSubClassEq(RC) && "Unknown 10-byte regclass");
    return load ? X86::LD_Fp80m : X86::ST_FpP80m;
  case 16: {
    assert((X86::VR128RegClass.hasSubClassEq(RC) ||
            X86::VR128XRegClass.hasSubClassEq(RC))&& "Unknown 16-byte regclass");
    // If stack is realigned we can use aligned stores.
    if (isStackAligned)
      return load ?
        (HasAVX ? X86::VMOVAPSrm : X86::MOVAPSrm) :
        (HasAVX ? X86::VMOVAPSmr : X86::MOVAPSmr);
    else
      return load ?
        (HasAVX ? X86::VMOVUPSrm : X86::MOVUPSrm) :
        (HasAVX ? X86::VMOVUPSmr : X86::MOVUPSmr);
  }
  case 32:
    assert((X86::VR256RegClass.hasSubClassEq(RC) ||
            X86::VR256XRegClass.hasSubClassEq(RC)) && "Unknown 32-byte regclass");
    // If stack is realigned we can use aligned stores.
    if (isStackAligned)
      return load ? X86::VMOVAPSYrm : X86::VMOVAPSYmr;
    else
      return load ? X86::VMOVUPSYrm : X86::VMOVUPSYmr;
  case 64:
    assert(X86::VR512RegClass.hasSubClassEq(RC) && "Unknown 64-byte regclass");
    if (isStackAligned)
      return load ? X86::VMOVAPSZrm : X86::VMOVAPSZmr;
    else
      return load ? X86::VMOVUPSZrm : X86::VMOVUPSZmr;
  }
}

bool X86InstrInfo::getMemOpBaseRegImmOfs(MachineInstr *MemOp, unsigned &BaseReg,
                                         unsigned &Offset,
                                         const TargetRegisterInfo *TRI) const {
  const MCInstrDesc &Desc = MemOp->getDesc();
  int MemRefBegin = X86II::getMemoryOperandNo(Desc.TSFlags, MemOp->getOpcode());
  if (MemRefBegin < 0)
    return false;

  MemRefBegin += X86II::getOperandBias(Desc);

  BaseReg = MemOp->getOperand(MemRefBegin + X86::AddrBaseReg).getReg();
  if (MemOp->getOperand(MemRefBegin + X86::AddrScaleAmt).getImm() != 1)
    return false;

  if (MemOp->getOperand(MemRefBegin + X86::AddrIndexReg).getReg() !=
      X86::NoRegister)
    return false;

  const MachineOperand &DispMO = MemOp->getOperand(MemRefBegin + X86::AddrDisp);

  // Displacement can be symbolic
  if (!DispMO.isImm())
    return false;

  Offset = DispMO.getImm();

  return (MemOp->getOperand(MemRefBegin + X86::AddrIndexReg).getReg() ==
          X86::NoRegister);
}

static unsigned getStoreRegOpcode(unsigned SrcReg,
                                  const TargetRegisterClass *RC,
                                  bool isStackAligned,
                                  const X86Subtarget &STI) {
  return getLoadStoreRegOpcode(SrcReg, RC, isStackAligned, STI, false);
}


static unsigned getLoadRegOpcode(unsigned DestReg,
                                 const TargetRegisterClass *RC,
                                 bool isStackAligned,
                                 const X86Subtarget &STI) {
  return getLoadStoreRegOpcode(DestReg, RC, isStackAligned, STI, true);
}

void X86InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned SrcReg, bool isKill, int FrameIdx,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  const MachineFunction &MF = *MBB.getParent();
  assert(MF.getFrameInfo()->getObjectSize(FrameIdx) >= RC->getSize() &&
         "Stack slot too small for store");
  unsigned Alignment = std::max<uint32_t>(RC->getSize(), 16);
  bool isAligned =
      (Subtarget.getFrameLowering()->getStackAlignment() >= Alignment) ||
      RI.canRealignStack(MF);
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, Subtarget);
  DebugLoc DL = MBB.findDebugLoc(MI);
  addFrameReference(BuildMI(MBB, MI, DL, get(Opc)), FrameIdx)
    .addReg(SrcReg, getKillRegState(isKill));
}

void X86InstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                  bool isKill,
                                  SmallVectorImpl<MachineOperand> &Addr,
                                  const TargetRegisterClass *RC,
                                  MachineInstr::mmo_iterator MMOBegin,
                                  MachineInstr::mmo_iterator MMOEnd,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Alignment = std::max<uint32_t>(RC->getSize(), 16);
  bool isAligned = MMOBegin != MMOEnd &&
                   (*MMOBegin)->getAlignment() >= Alignment;
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, Subtarget);
  DebugLoc DL;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  MIB.addReg(SrcReg, getKillRegState(isKill));
  (*MIB).setMemRefs(MMOBegin, MMOEnd);
  NewMIs.push_back(MIB);
}


void X86InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  const MachineFunction &MF = *MBB.getParent();
  unsigned Alignment = std::max<uint32_t>(RC->getSize(), 16);
  bool isAligned =
      (Subtarget.getFrameLowering()->getStackAlignment() >= Alignment) ||
      RI.canRealignStack(MF);
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, Subtarget);
  DebugLoc DL = MBB.findDebugLoc(MI);
  addFrameReference(BuildMI(MBB, MI, DL, get(Opc), DestReg), FrameIdx);
}

void X86InstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 MachineInstr::mmo_iterator MMOBegin,
                                 MachineInstr::mmo_iterator MMOEnd,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Alignment = std::max<uint32_t>(RC->getSize(), 16);
  bool isAligned = MMOBegin != MMOEnd &&
                   (*MMOBegin)->getAlignment() >= Alignment;
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, Subtarget);
  DebugLoc DL;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  (*MIB).setMemRefs(MMOBegin, MMOEnd);
  NewMIs.push_back(MIB);
}

bool X86InstrInfo::
analyzeCompare(const MachineInstr *MI, unsigned &SrcReg, unsigned &SrcReg2,
               int &CmpMask, int &CmpValue) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP8ri:
    SrcReg = MI->getOperand(0).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    CmpValue = MI->getOperand(1).getImm();
    return true;
  // A SUB can be used to perform comparison.
  case X86::SUB64rm:
  case X86::SUB32rm:
  case X86::SUB16rm:
  case X86::SUB8rm:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  case X86::SUB64rr:
  case X86::SUB32rr:
  case X86::SUB16rr:
  case X86::SUB8rr:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = MI->getOperand(2).getReg();
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  case X86::SUB64ri32:
  case X86::SUB64ri8:
  case X86::SUB32ri:
  case X86::SUB32ri8:
  case X86::SUB16ri:
  case X86::SUB16ri8:
  case X86::SUB8ri:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    CmpValue = MI->getOperand(2).getImm();
    return true;
  case X86::CMP64rr:
  case X86::CMP32rr:
  case X86::CMP16rr:
  case X86::CMP8rr:
    SrcReg = MI->getOperand(0).getReg();
    SrcReg2 = MI->getOperand(1).getReg();
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
    SrcReg = MI->getOperand(0).getReg();
    if (MI->getOperand(1).getReg() != SrcReg) return false;
    // Compare against zero.
    SrcReg2 = 0;
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  }
  return false;
}

/// Check whether the first instruction, whose only
/// purpose is to update flags, can be made redundant.
/// CMPrr can be made redundant by SUBrr if the operands are the same.
/// This function can be extended later on.
/// SrcReg, SrcRegs: register operands for FlagI.
/// ImmValue: immediate for FlagI if it takes an immediate.
inline static bool isRedundantFlagInstr(MachineInstr *FlagI, unsigned SrcReg,
                                        unsigned SrcReg2, int ImmValue,
                                        MachineInstr *OI) {
  if (((FlagI->getOpcode() == X86::CMP64rr &&
        OI->getOpcode() == X86::SUB64rr) ||
       (FlagI->getOpcode() == X86::CMP32rr &&
        OI->getOpcode() == X86::SUB32rr)||
       (FlagI->getOpcode() == X86::CMP16rr &&
        OI->getOpcode() == X86::SUB16rr)||
       (FlagI->getOpcode() == X86::CMP8rr &&
        OI->getOpcode() == X86::SUB8rr)) &&
      ((OI->getOperand(1).getReg() == SrcReg &&
        OI->getOperand(2).getReg() == SrcReg2) ||
       (OI->getOperand(1).getReg() == SrcReg2 &&
        OI->getOperand(2).getReg() == SrcReg)))
    return true;

  if (((FlagI->getOpcode() == X86::CMP64ri32 &&
        OI->getOpcode() == X86::SUB64ri32) ||
       (FlagI->getOpcode() == X86::CMP64ri8 &&
        OI->getOpcode() == X86::SUB64ri8) ||
       (FlagI->getOpcode() == X86::CMP32ri &&
        OI->getOpcode() == X86::SUB32ri) ||
       (FlagI->getOpcode() == X86::CMP32ri8 &&
        OI->getOpcode() == X86::SUB32ri8) ||
       (FlagI->getOpcode() == X86::CMP16ri &&
        OI->getOpcode() == X86::SUB16ri) ||
       (FlagI->getOpcode() == X86::CMP16ri8 &&
        OI->getOpcode() == X86::SUB16ri8) ||
       (FlagI->getOpcode() == X86::CMP8ri &&
        OI->getOpcode() == X86::SUB8ri)) &&
      OI->getOperand(1).getReg() == SrcReg &&
      OI->getOperand(2).getImm() == ImmValue)
    return true;
  return false;
}

/// Check whether the definition can be converted
/// to remove a comparison against zero.
inline static bool isDefConvertible(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: return false;

  // The shift instructions only modify ZF if their shift count is non-zero.
  // N.B.: The processor truncates the shift count depending on the encoding.
  case X86::SAR8ri:    case X86::SAR16ri:  case X86::SAR32ri:case X86::SAR64ri:
  case X86::SHR8ri:    case X86::SHR16ri:  case X86::SHR32ri:case X86::SHR64ri:
     return getTruncatedShiftCount(MI, 2) != 0;

  // Some left shift instructions can be turned into LEA instructions but only
  // if their flags aren't used. Avoid transforming such instructions.
  case X86::SHL8ri:    case X86::SHL16ri:  case X86::SHL32ri:case X86::SHL64ri:{
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (isTruncatedShiftCountForLEA(ShAmt)) return false;
    return ShAmt != 0;
  }

  case X86::SHRD16rri8:case X86::SHRD32rri8:case X86::SHRD64rri8:
  case X86::SHLD16rri8:case X86::SHLD32rri8:case X86::SHLD64rri8:
     return getTruncatedShiftCount(MI, 3) != 0;

  case X86::SUB64ri32: case X86::SUB64ri8: case X86::SUB32ri:
  case X86::SUB32ri8:  case X86::SUB16ri:  case X86::SUB16ri8:
  case X86::SUB8ri:    case X86::SUB64rr:  case X86::SUB32rr:
  case X86::SUB16rr:   case X86::SUB8rr:   case X86::SUB64rm:
  case X86::SUB32rm:   case X86::SUB16rm:  case X86::SUB8rm:
  case X86::DEC64r:    case X86::DEC32r:   case X86::DEC16r: case X86::DEC8r:
  case X86::ADD64ri32: case X86::ADD64ri8: case X86::ADD32ri:
  case X86::ADD32ri8:  case X86::ADD16ri:  case X86::ADD16ri8:
  case X86::ADD8ri:    case X86::ADD64rr:  case X86::ADD32rr:
  case X86::ADD16rr:   case X86::ADD8rr:   case X86::ADD64rm:
  case X86::ADD32rm:   case X86::ADD16rm:  case X86::ADD8rm:
  case X86::INC64r:    case X86::INC32r:   case X86::INC16r: case X86::INC8r:
  case X86::AND64ri32: case X86::AND64ri8: case X86::AND32ri:
  case X86::AND32ri8:  case X86::AND16ri:  case X86::AND16ri8:
  case X86::AND8ri:    case X86::AND64rr:  case X86::AND32rr:
  case X86::AND16rr:   case X86::AND8rr:   case X86::AND64rm:
  case X86::AND32rm:   case X86::AND16rm:  case X86::AND8rm:
  case X86::XOR64ri32: case X86::XOR64ri8: case X86::XOR32ri:
  case X86::XOR32ri8:  case X86::XOR16ri:  case X86::XOR16ri8:
  case X86::XOR8ri:    case X86::XOR64rr:  case X86::XOR32rr:
  case X86::XOR16rr:   case X86::XOR8rr:   case X86::XOR64rm:
  case X86::XOR32rm:   case X86::XOR16rm:  case X86::XOR8rm:
  case X86::OR64ri32:  case X86::OR64ri8:  case X86::OR32ri:
  case X86::OR32ri8:   case X86::OR16ri:   case X86::OR16ri8:
  case X86::OR8ri:     case X86::OR64rr:   case X86::OR32rr:
  case X86::OR16rr:    case X86::OR8rr:    case X86::OR64rm:
  case X86::OR32rm:    case X86::OR16rm:   case X86::OR8rm:
  case X86::NEG8r:     case X86::NEG16r:   case X86::NEG32r: case X86::NEG64r:
  case X86::SAR8r1:    case X86::SAR16r1:  case X86::SAR32r1:case X86::SAR64r1:
  case X86::SHR8r1:    case X86::SHR16r1:  case X86::SHR32r1:case X86::SHR64r1:
  case X86::SHL8r1:    case X86::SHL16r1:  case X86::SHL32r1:case X86::SHL64r1:
  case X86::ADC32ri:   case X86::ADC32ri8:
  case X86::ADC32rr:   case X86::ADC64ri32:
  case X86::ADC64ri8:  case X86::ADC64rr:
  case X86::SBB32ri:   case X86::SBB32ri8:
  case X86::SBB32rr:   case X86::SBB64ri32:
  case X86::SBB64ri8:  case X86::SBB64rr:
  case X86::ANDN32rr:  case X86::ANDN32rm:
  case X86::ANDN64rr:  case X86::ANDN64rm:
  case X86::BEXTR32rr: case X86::BEXTR64rr:
  case X86::BEXTR32rm: case X86::BEXTR64rm:
  case X86::BLSI32rr:  case X86::BLSI32rm:
  case X86::BLSI64rr:  case X86::BLSI64rm:
  case X86::BLSMSK32rr:case X86::BLSMSK32rm:
  case X86::BLSMSK64rr:case X86::BLSMSK64rm:
  case X86::BLSR32rr:  case X86::BLSR32rm:
  case X86::BLSR64rr:  case X86::BLSR64rm:
  case X86::BZHI32rr:  case X86::BZHI32rm:
  case X86::BZHI64rr:  case X86::BZHI64rm:
  case X86::LZCNT16rr: case X86::LZCNT16rm:
  case X86::LZCNT32rr: case X86::LZCNT32rm:
  case X86::LZCNT64rr: case X86::LZCNT64rm:
  case X86::POPCNT16rr:case X86::POPCNT16rm:
  case X86::POPCNT32rr:case X86::POPCNT32rm:
  case X86::POPCNT64rr:case X86::POPCNT64rm:
  case X86::TZCNT16rr: case X86::TZCNT16rm:
  case X86::TZCNT32rr: case X86::TZCNT32rm:
  case X86::TZCNT64rr: case X86::TZCNT64rm:
    return true;
  }
}

/// Check whether the use can be converted to remove a comparison against zero.
static X86::CondCode isUseDefConvertible(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: return X86::COND_INVALID;
  case X86::LZCNT16rr: case X86::LZCNT16rm:
  case X86::LZCNT32rr: case X86::LZCNT32rm:
  case X86::LZCNT64rr: case X86::LZCNT64rm:
    return X86::COND_B;
  case X86::POPCNT16rr:case X86::POPCNT16rm:
  case X86::POPCNT32rr:case X86::POPCNT32rm:
  case X86::POPCNT64rr:case X86::POPCNT64rm:
    return X86::COND_E;
  case X86::TZCNT16rr: case X86::TZCNT16rm:
  case X86::TZCNT32rr: case X86::TZCNT32rm:
  case X86::TZCNT64rr: case X86::TZCNT64rm:
    return X86::COND_B;
  }
}

/// Check if there exists an earlier instruction that
/// operates on the same source operands and sets flags in the same way as
/// Compare; remove Compare if possible.
bool X86InstrInfo::
optimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg, unsigned SrcReg2,
                     int CmpMask, int CmpValue,
                     const MachineRegisterInfo *MRI) const {
  // Check whether we can replace SUB with CMP.
  unsigned NewOpcode = 0;
  switch (CmpInstr->getOpcode()) {
  default: break;
  case X86::SUB64ri32:
  case X86::SUB64ri8:
  case X86::SUB32ri:
  case X86::SUB32ri8:
  case X86::SUB16ri:
  case X86::SUB16ri8:
  case X86::SUB8ri:
  case X86::SUB64rm:
  case X86::SUB32rm:
  case X86::SUB16rm:
  case X86::SUB8rm:
  case X86::SUB64rr:
  case X86::SUB32rr:
  case X86::SUB16rr:
  case X86::SUB8rr: {
    if (!MRI->use_nodbg_empty(CmpInstr->getOperand(0).getReg()))
      return false;
    // There is no use of the destination register, we can replace SUB with CMP.
    switch (CmpInstr->getOpcode()) {
    default: llvm_unreachable("Unreachable!");
    case X86::SUB64rm:   NewOpcode = X86::CMP64rm;   break;
    case X86::SUB32rm:   NewOpcode = X86::CMP32rm;   break;
    case X86::SUB16rm:   NewOpcode = X86::CMP16rm;   break;
    case X86::SUB8rm:    NewOpcode = X86::CMP8rm;    break;
    case X86::SUB64rr:   NewOpcode = X86::CMP64rr;   break;
    case X86::SUB32rr:   NewOpcode = X86::CMP32rr;   break;
    case X86::SUB16rr:   NewOpcode = X86::CMP16rr;   break;
    case X86::SUB8rr:    NewOpcode = X86::CMP8rr;    break;
    case X86::SUB64ri32: NewOpcode = X86::CMP64ri32; break;
    case X86::SUB64ri8:  NewOpcode = X86::CMP64ri8;  break;
    case X86::SUB32ri:   NewOpcode = X86::CMP32ri;   break;
    case X86::SUB32ri8:  NewOpcode = X86::CMP32ri8;  break;
    case X86::SUB16ri:   NewOpcode = X86::CMP16ri;   break;
    case X86::SUB16ri8:  NewOpcode = X86::CMP16ri8;  break;
    case X86::SUB8ri:    NewOpcode = X86::CMP8ri;    break;
    }
    CmpInstr->setDesc(get(NewOpcode));
    CmpInstr->RemoveOperand(0);
    // Fall through to optimize Cmp if Cmp is CMPrr or CMPri.
    if (NewOpcode == X86::CMP64rm || NewOpcode == X86::CMP32rm ||
        NewOpcode == X86::CMP16rm || NewOpcode == X86::CMP8rm)
      return false;
  }
  }

  // Get the unique definition of SrcReg.
  MachineInstr *MI = MRI->getUniqueVRegDef(SrcReg);
  if (!MI) return false;

  // CmpInstr is the first instruction of the BB.
  MachineBasicBlock::iterator I = CmpInstr, Def = MI;

  // If we are comparing against zero, check whether we can use MI to update
  // EFLAGS. If MI is not in the same BB as CmpInstr, do not optimize.
  bool IsCmpZero = (SrcReg2 == 0 && CmpValue == 0);
  if (IsCmpZero && MI->getParent() != CmpInstr->getParent())
    return false;

  // If we have a use of the source register between the def and our compare
  // instruction we can eliminate the compare iff the use sets EFLAGS in the
  // right way.
  bool ShouldUpdateCC = false;
  X86::CondCode NewCC = X86::COND_INVALID;
  if (IsCmpZero && !isDefConvertible(MI)) {
    // Scan forward from the use until we hit the use we're looking for or the
    // compare instruction.
    for (MachineBasicBlock::iterator J = MI;; ++J) {
      // Do we have a convertible instruction?
      NewCC = isUseDefConvertible(J);
      if (NewCC != X86::COND_INVALID && J->getOperand(1).isReg() &&
          J->getOperand(1).getReg() == SrcReg) {
        assert(J->definesRegister(X86::EFLAGS) && "Must be an EFLAGS def!");
        ShouldUpdateCC = true; // Update CC later on.
        // This is not a def of SrcReg, but still a def of EFLAGS. Keep going
        // with the new def.
        MI = Def = J;
        break;
      }

      if (J == I)
        return false;
    }
  }

  // We are searching for an earlier instruction that can make CmpInstr
  // redundant and that instruction will be saved in Sub.
  MachineInstr *Sub = nullptr;
  const TargetRegisterInfo *TRI = &getRegisterInfo();

  // We iterate backward, starting from the instruction before CmpInstr and
  // stop when reaching the definition of a source register or done with the BB.
  // RI points to the instruction before CmpInstr.
  // If the definition is in this basic block, RE points to the definition;
  // otherwise, RE is the rend of the basic block.
  MachineBasicBlock::reverse_iterator
      RI = MachineBasicBlock::reverse_iterator(I),
      RE = CmpInstr->getParent() == MI->getParent() ?
           MachineBasicBlock::reverse_iterator(++Def) /* points to MI */ :
           CmpInstr->getParent()->rend();
  MachineInstr *Movr0Inst = nullptr;
  for (; RI != RE; ++RI) {
    MachineInstr *Instr = &*RI;
    // Check whether CmpInstr can be made redundant by the current instruction.
    if (!IsCmpZero &&
        isRedundantFlagInstr(CmpInstr, SrcReg, SrcReg2, CmpValue, Instr)) {
      Sub = Instr;
      break;
    }

    if (Instr->modifiesRegister(X86::EFLAGS, TRI) ||
        Instr->readsRegister(X86::EFLAGS, TRI)) {
      // This instruction modifies or uses EFLAGS.

      // MOV32r0 etc. are implemented with xor which clobbers condition code.
      // They are safe to move up, if the definition to EFLAGS is dead and
      // earlier instructions do not read or write EFLAGS.
      if (!Movr0Inst && Instr->getOpcode() == X86::MOV32r0 &&
          Instr->registerDefIsDead(X86::EFLAGS, TRI)) {
        Movr0Inst = Instr;
        continue;
      }

      // We can't remove CmpInstr.
      return false;
    }
  }

  // Return false if no candidates exist.
  if (!IsCmpZero && !Sub)
    return false;

  bool IsSwapped = (SrcReg2 != 0 && Sub->getOperand(1).getReg() == SrcReg2 &&
                    Sub->getOperand(2).getReg() == SrcReg);

  // Scan forward from the instruction after CmpInstr for uses of EFLAGS.
  // It is safe to remove CmpInstr if EFLAGS is redefined or killed.
  // If we are done with the basic block, we need to check whether EFLAGS is
  // live-out.
  bool IsSafe = false;
  SmallVector<std::pair<MachineInstr*, unsigned /*NewOpc*/>, 4> OpsToUpdate;
  MachineBasicBlock::iterator E = CmpInstr->getParent()->end();
  for (++I; I != E; ++I) {
    const MachineInstr &Instr = *I;
    bool ModifyEFLAGS = Instr.modifiesRegister(X86::EFLAGS, TRI);
    bool UseEFLAGS = Instr.readsRegister(X86::EFLAGS, TRI);
    // We should check the usage if this instruction uses and updates EFLAGS.
    if (!UseEFLAGS && ModifyEFLAGS) {
      // It is safe to remove CmpInstr if EFLAGS is updated again.
      IsSafe = true;
      break;
    }
    if (!UseEFLAGS && !ModifyEFLAGS)
      continue;

    // EFLAGS is used by this instruction.
    X86::CondCode OldCC = X86::COND_INVALID;
    bool OpcIsSET = false;
    if (IsCmpZero || IsSwapped) {
      // We decode the condition code from opcode.
      if (Instr.isBranch())
        OldCC = getCondFromBranchOpc(Instr.getOpcode());
      else {
        OldCC = getCondFromSETOpc(Instr.getOpcode());
        if (OldCC != X86::COND_INVALID)
          OpcIsSET = true;
        else
          OldCC = X86::getCondFromCMovOpc(Instr.getOpcode());
      }
      if (OldCC == X86::COND_INVALID) return false;
    }
    if (IsCmpZero) {
      switch (OldCC) {
      default: break;
      case X86::COND_A: case X86::COND_AE:
      case X86::COND_B: case X86::COND_BE:
      case X86::COND_G: case X86::COND_GE:
      case X86::COND_L: case X86::COND_LE:
      case X86::COND_O: case X86::COND_NO:
        // CF and OF are used, we can't perform this optimization.
        return false;
      }

      // If we're updating the condition code check if we have to reverse the
      // condition.
      if (ShouldUpdateCC)
        switch (OldCC) {
        default:
          return false;
        case X86::COND_E:
          break;
        case X86::COND_NE:
          NewCC = GetOppositeBranchCondition(NewCC);
          break;
        }
    } else if (IsSwapped) {
      // If we have SUB(r1, r2) and CMP(r2, r1), the condition code needs
      // to be changed from r2 > r1 to r1 < r2, from r2 < r1 to r1 > r2, etc.
      // We swap the condition code and synthesize the new opcode.
      NewCC = getSwappedCondition(OldCC);
      if (NewCC == X86::COND_INVALID) return false;
    }

    if ((ShouldUpdateCC || IsSwapped) && NewCC != OldCC) {
      // Synthesize the new opcode.
      bool HasMemoryOperand = Instr.hasOneMemOperand();
      unsigned NewOpc;
      if (Instr.isBranch())
        NewOpc = GetCondBranchFromCond(NewCC);
      else if(OpcIsSET)
        NewOpc = getSETFromCond(NewCC, HasMemoryOperand);
      else {
        unsigned DstReg = Instr.getOperand(0).getReg();
        NewOpc = getCMovFromCond(NewCC, MRI->getRegClass(DstReg)->getSize(),
                                 HasMemoryOperand);
      }

      // Push the MachineInstr to OpsToUpdate.
      // If it is safe to remove CmpInstr, the condition code of these
      // instructions will be modified.
      OpsToUpdate.push_back(std::make_pair(&*I, NewOpc));
    }
    if (ModifyEFLAGS || Instr.killsRegister(X86::EFLAGS, TRI)) {
      // It is safe to remove CmpInstr if EFLAGS is updated again or killed.
      IsSafe = true;
      break;
    }
  }

  // If EFLAGS is not killed nor re-defined, we should check whether it is
  // live-out. If it is live-out, do not optimize.
  if ((IsCmpZero || IsSwapped) && !IsSafe) {
    MachineBasicBlock *MBB = CmpInstr->getParent();
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
             SE = MBB->succ_end(); SI != SE; ++SI)
      if ((*SI)->isLiveIn(X86::EFLAGS))
        return false;
  }

  // The instruction to be updated is either Sub or MI.
  Sub = IsCmpZero ? MI : Sub;
  // Move Movr0Inst to the appropriate place before Sub.
  if (Movr0Inst) {
    // Look backwards until we find a def that doesn't use the current EFLAGS.
    Def = Sub;
    MachineBasicBlock::reverse_iterator
      InsertI = MachineBasicBlock::reverse_iterator(++Def),
                InsertE = Sub->getParent()->rend();
    for (; InsertI != InsertE; ++InsertI) {
      MachineInstr *Instr = &*InsertI;
      if (!Instr->readsRegister(X86::EFLAGS, TRI) &&
          Instr->modifiesRegister(X86::EFLAGS, TRI)) {
        Sub->getParent()->remove(Movr0Inst);
        Instr->getParent()->insert(MachineBasicBlock::iterator(Instr),
                                   Movr0Inst);
        break;
      }
    }
    if (InsertI == InsertE)
      return false;
  }

  // Make sure Sub instruction defines EFLAGS and mark the def live.
  unsigned i = 0, e = Sub->getNumOperands();
  for (; i != e; ++i) {
    MachineOperand &MO = Sub->getOperand(i);
    if (MO.isReg() && MO.isDef() && MO.getReg() == X86::EFLAGS) {
      MO.setIsDead(false);
      break;
    }
  }
  assert(i != e && "Unable to locate a def EFLAGS operand");

  CmpInstr->eraseFromParent();

  // Modify the condition code of instructions in OpsToUpdate.
  for (unsigned i = 0, e = OpsToUpdate.size(); i < e; i++)
    OpsToUpdate[i].first->setDesc(get(OpsToUpdate[i].second));
  return true;
}

/// Try to remove the load by folding it to a register
/// operand at the use. We fold the load instructions if load defines a virtual
/// register, the virtual register is used once in the same BB, and the
/// instructions in-between do not load or store, and have no side effects.
MachineInstr *X86InstrInfo::optimizeLoadInstr(MachineInstr *MI,
                                              const MachineRegisterInfo *MRI,
                                              unsigned &FoldAsLoadDefReg,
                                              MachineInstr *&DefMI) const {
  if (FoldAsLoadDefReg == 0)
    return nullptr;
  // To be conservative, if there exists another load, clear the load candidate.
  if (MI->mayLoad()) {
    FoldAsLoadDefReg = 0;
    return nullptr;
  }

  // Check whether we can move DefMI here.
  DefMI = MRI->getVRegDef(FoldAsLoadDefReg);
  assert(DefMI);
  bool SawStore = false;
  if (!DefMI->isSafeToMove(nullptr, SawStore))
    return nullptr;

  // Collect information about virtual register operands of MI.
  unsigned SrcOperandId = 0;
  bool FoundSrcOperand = false;
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg != FoldAsLoadDefReg)
      continue;
    // Do not fold if we have a subreg use or a def or multiple uses.
    if (MO.getSubReg() || MO.isDef() || FoundSrcOperand)
      return nullptr;

    SrcOperandId = i;
    FoundSrcOperand = true;
  }
  if (!FoundSrcOperand)
    return nullptr;

  // Check whether we can fold the def into SrcOperandId.
  MachineInstr *FoldMI = foldMemoryOperand(MI, SrcOperandId, DefMI);
  if (FoldMI) {
    FoldAsLoadDefReg = 0;
    return FoldMI;
  }

  return nullptr;
}

/// Expand a single-def pseudo instruction to a two-addr
/// instruction with two undef reads of the register being defined.
/// This is used for mapping:
///   %xmm4 = V_SET0
/// to:
///   %xmm4 = PXORrr %xmm4<undef>, %xmm4<undef>
///
static bool Expand2AddrUndef(MachineInstrBuilder &MIB,
                             const MCInstrDesc &Desc) {
  assert(Desc.getNumOperands() == 3 && "Expected two-addr instruction.");
  unsigned Reg = MIB->getOperand(0).getReg();
  MIB->setDesc(Desc);

  // MachineInstr::addOperand() will insert explicit operands before any
  // implicit operands.
  MIB.addReg(Reg, RegState::Undef).addReg(Reg, RegState::Undef);
  // But we don't trust that.
  assert(MIB->getOperand(1).getReg() == Reg &&
         MIB->getOperand(2).getReg() == Reg && "Misplaced operand");
  return true;
}

// LoadStackGuard has so far only been implemented for 64-bit MachO. Different
// code sequence is needed for other targets.
static void expandLoadStackGuard(MachineInstrBuilder &MIB,
                                 const TargetInstrInfo &TII) {
  MachineBasicBlock &MBB = *MIB->getParent();
  DebugLoc DL = MIB->getDebugLoc();
  unsigned Reg = MIB->getOperand(0).getReg();
  const GlobalValue *GV =
      cast<GlobalValue>((*MIB->memoperands_begin())->getValue());
  unsigned Flag = MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant;
  MachineMemOperand *MMO = MBB.getParent()->getMachineMemOperand(
      MachinePointerInfo::getGOT(*MBB.getParent()), Flag, 8, 8);
  MachineBasicBlock::iterator I = MIB.getInstr();

  BuildMI(MBB, I, DL, TII.get(X86::MOV64rm), Reg).addReg(X86::RIP).addImm(1)
      .addReg(0).addGlobalAddress(GV, 0, X86II::MO_GOTPCREL).addReg(0)
      .addMemOperand(MMO);
  MIB->setDebugLoc(DL);
  MIB->setDesc(TII.get(X86::MOV64rm));
  MIB.addReg(Reg, RegState::Kill).addImm(1).addReg(0).addImm(0).addReg(0);
}

bool X86InstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  bool HasAVX = Subtarget.hasAVX();
  MachineInstrBuilder MIB(*MI->getParent()->getParent(), MI);
  switch (MI->getOpcode()) {
  case X86::MOV32r0:
    return Expand2AddrUndef(MIB, get(X86::XOR32rr));
  case X86::SETB_C8r:
    return Expand2AddrUndef(MIB, get(X86::SBB8rr));
  case X86::SETB_C16r:
    return Expand2AddrUndef(MIB, get(X86::SBB16rr));
  case X86::SETB_C32r:
    return Expand2AddrUndef(MIB, get(X86::SBB32rr));
  case X86::SETB_C64r:
    return Expand2AddrUndef(MIB, get(X86::SBB64rr));
  case X86::V_SET0:
  case X86::FsFLD0SS:
  case X86::FsFLD0SD:
    return Expand2AddrUndef(MIB, get(HasAVX ? X86::VXORPSrr : X86::XORPSrr));
  case X86::AVX_SET0:
    assert(HasAVX && "AVX not supported");
    return Expand2AddrUndef(MIB, get(X86::VXORPSYrr));
  case X86::AVX512_512_SET0:
    return Expand2AddrUndef(MIB, get(X86::VPXORDZrr));
  case X86::V_SETALLONES:
    return Expand2AddrUndef(MIB, get(HasAVX ? X86::VPCMPEQDrr : X86::PCMPEQDrr));
  case X86::AVX2_SETALLONES:
    return Expand2AddrUndef(MIB, get(X86::VPCMPEQDYrr));
  case X86::TEST8ri_NOREX:
    MI->setDesc(get(X86::TEST8ri));
    return true;
  case X86::KSET0B:
  case X86::KSET0W: return Expand2AddrUndef(MIB, get(X86::KXORWrr));
  case X86::KSET0D: return Expand2AddrUndef(MIB, get(X86::KXORDrr));
  case X86::KSET0Q: return Expand2AddrUndef(MIB, get(X86::KXORQrr));
  case X86::KSET1B:
  case X86::KSET1W: return Expand2AddrUndef(MIB, get(X86::KXNORWrr));
  case X86::KSET1D: return Expand2AddrUndef(MIB, get(X86::KXNORDrr));
  case X86::KSET1Q: return Expand2AddrUndef(MIB, get(X86::KXNORQrr));
  case TargetOpcode::LOAD_STACK_GUARD:
    expandLoadStackGuard(MIB, *this);
    return true;
  }
  return false;
}

static void addOperands(MachineInstrBuilder &MIB, ArrayRef<MachineOperand> MOs,
                        int PtrOffset = 0) {
  unsigned NumAddrOps = MOs.size();

  if (NumAddrOps < 4) {
    // FrameIndex only - add an immediate offset (whether its zero or not).
    for (unsigned i = 0; i != NumAddrOps; ++i)
      MIB.addOperand(MOs[i]);
    addOffset(MIB, PtrOffset);
  } else {
    // General Memory Addressing - we need to add any offset to an existing
    // offset.
    assert(MOs.size() == 5 && "Unexpected memory operand list length");
    for (unsigned i = 0; i != NumAddrOps; ++i) {
      const MachineOperand &MO = MOs[i];
      if (i == 3 && PtrOffset != 0) {
        MIB.addDisp(MO, PtrOffset);
      } else {
        MIB.addOperand(MO);
      }
    }
  }
}

static MachineInstr *FuseTwoAddrInst(MachineFunction &MF, unsigned Opcode,
                                     ArrayRef<MachineOperand> MOs,
                                     MachineBasicBlock::iterator InsertPt,
                                     MachineInstr *MI,
                                     const TargetInstrInfo &TII) {
  // Create the base instruction with the memory operand as the first part.
  // Omit the implicit operands, something BuildMI can't do.
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);
  addOperands(MIB, MOs);

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

  MachineBasicBlock *MBB = InsertPt->getParent();
  MBB->insert(InsertPt, NewMI);

  return MIB;
}

static MachineInstr *FuseInst(MachineFunction &MF, unsigned Opcode,
                              unsigned OpNo, ArrayRef<MachineOperand> MOs,
                              MachineBasicBlock::iterator InsertPt,
                              MachineInstr *MI, const TargetInstrInfo &TII,
                              int PtrOffset = 0) {
  // Omit the implicit operands, something BuildMI can't do.
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (i == OpNo) {
      assert(MO.isReg() && "Expected to fold into reg operand!");
      addOperands(MIB, MOs, PtrOffset);
    } else {
      MIB.addOperand(MO);
    }
  }

  MachineBasicBlock *MBB = InsertPt->getParent();
  MBB->insert(InsertPt, NewMI);

  return MIB;
}

static MachineInstr *MakeM0Inst(const TargetInstrInfo &TII, unsigned Opcode,
                                ArrayRef<MachineOperand> MOs,
                                MachineBasicBlock::iterator InsertPt,
                                MachineInstr *MI) {
  MachineInstrBuilder MIB = BuildMI(*InsertPt->getParent(), InsertPt,
                                    MI->getDebugLoc(), TII.get(Opcode));
  addOperands(MIB, MOs);
  return MIB.addImm(0);
}

MachineInstr *X86InstrInfo::foldMemoryOperandCustom(
    MachineFunction &MF, MachineInstr *MI, unsigned OpNum,
    ArrayRef<MachineOperand> MOs, MachineBasicBlock::iterator InsertPt,
    unsigned Size, unsigned Align) const {
  switch (MI->getOpcode()) {
  case X86::INSERTPSrr:
  case X86::VINSERTPSrr:
    // Attempt to convert the load of inserted vector into a fold load
    // of a single float.
    if (OpNum == 2) {
      unsigned Imm = MI->getOperand(MI->getNumOperands() - 1).getImm();
      unsigned ZMask = Imm & 15;
      unsigned DstIdx = (Imm >> 4) & 3;
      unsigned SrcIdx = (Imm >> 6) & 3;

      unsigned RCSize = getRegClass(MI->getDesc(), OpNum, &RI, MF)->getSize();
      if (Size <= RCSize && 4 <= Align) {
        int PtrOffset = SrcIdx * 4;
        unsigned NewImm = (DstIdx << 4) | ZMask;
        unsigned NewOpCode =
            (MI->getOpcode() == X86::VINSERTPSrr ? X86::VINSERTPSrm
                                                 : X86::INSERTPSrm);
        MachineInstr *NewMI =
            FuseInst(MF, NewOpCode, OpNum, MOs, InsertPt, MI, *this, PtrOffset);
        NewMI->getOperand(NewMI->getNumOperands() - 1).setImm(NewImm);
        return NewMI;
      }
    }
    break;
  };

  return nullptr;
}

MachineInstr *X86InstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr *MI, unsigned OpNum,
    ArrayRef<MachineOperand> MOs, MachineBasicBlock::iterator InsertPt,
    unsigned Size, unsigned Align, bool AllowCommute) const {
  const DenseMap<unsigned,
                 std::pair<unsigned,unsigned> > *OpcodeTablePtr = nullptr;
  bool isCallRegIndirect = Subtarget.callRegIndirect();
  bool isTwoAddrFold = false;

  // For CPUs that favor the register form of a call or push,
  // do not fold loads into calls or pushes, unless optimizing for size
  // aggressively.
  if (isCallRegIndirect && !MF.getFunction()->optForMinSize() &&
      (MI->getOpcode() == X86::CALL32r || MI->getOpcode() == X86::CALL64r ||
       MI->getOpcode() == X86::PUSH16r || MI->getOpcode() == X86::PUSH32r ||
       MI->getOpcode() == X86::PUSH64r))
    return nullptr;

  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // FIXME: AsmPrinter doesn't know how to handle
  // X86II::MO_GOT_ABSOLUTE_ADDRESS after folding.
  if (MI->getOpcode() == X86::ADD32ri &&
      MI->getOperand(2).getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS)
    return nullptr;

  MachineInstr *NewMI = nullptr;

  // Attempt to fold any custom cases we have.
  if (MachineInstr *CustomMI =
          foldMemoryOperandCustom(MF, MI, OpNum, MOs, InsertPt, Size, Align))
    return CustomMI;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (isTwoAddr && NumOps >= 2 && OpNum < 2 &&
      MI->getOperand(0).isReg() &&
      MI->getOperand(1).isReg() &&
      MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) {
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
    isTwoAddrFold = true;
  } else if (OpNum == 0) {
    if (MI->getOpcode() == X86::MOV32r0) {
      NewMI = MakeM0Inst(*this, X86::MOV32mi, MOs, InsertPt, MI);
      if (NewMI)
        return NewMI;
    }

    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (OpNum == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (OpNum == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  } else if (OpNum == 3) {
    OpcodeTablePtr = &RegOp2MemOpTable3;
  } else if (OpNum == 4) {
    OpcodeTablePtr = &RegOp2MemOpTable4;
  }

  // If table selected...
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
      OpcodeTablePtr->find(MI->getOpcode());
    if (I != OpcodeTablePtr->end()) {
      unsigned Opcode = I->second.first;
      unsigned MinAlign = (I->second.second & TB_ALIGN_MASK) >> TB_ALIGN_SHIFT;
      if (Align < MinAlign)
        return nullptr;
      bool NarrowToMOV32rm = false;
      if (Size) {
        unsigned RCSize = getRegClass(MI->getDesc(), OpNum, &RI, MF)->getSize();
        if (Size < RCSize) {
          // Check if it's safe to fold the load. If the size of the object is
          // narrower than the load width, then it's not.
          if (Opcode != X86::MOV64rm || RCSize != 8 || Size != 4)
            return nullptr;
          // If this is a 64-bit load, but the spill slot is 32, then we can do
          // a 32-bit load which is implicitly zero-extended. This likely is
          // due to live interval analysis remat'ing a load from stack slot.
          if (MI->getOperand(0).getSubReg() || MI->getOperand(1).getSubReg())
            return nullptr;
          Opcode = X86::MOV32rm;
          NarrowToMOV32rm = true;
        }
      }

      if (isTwoAddrFold)
        NewMI = FuseTwoAddrInst(MF, Opcode, MOs, InsertPt, MI, *this);
      else
        NewMI = FuseInst(MF, Opcode, OpNum, MOs, InsertPt, MI, *this);

      if (NarrowToMOV32rm) {
        // If this is the special case where we use a MOV32rm to load a 32-bit
        // value and zero-extend the top bits. Change the destination register
        // to a 32-bit one.
        unsigned DstReg = NewMI->getOperand(0).getReg();
        if (TargetRegisterInfo::isPhysicalRegister(DstReg))
          NewMI->getOperand(0).setReg(RI.getSubReg(DstReg, X86::sub_32bit));
        else
          NewMI->getOperand(0).setSubReg(X86::sub_32bit);
      }
      return NewMI;
    }
  }

  // If the instruction and target operand are commutable, commute the
  // instruction and try again.
  if (AllowCommute) {
    unsigned CommuteOpIdx1 = OpNum, CommuteOpIdx2 = CommuteAnyOperandIndex;
    if (findCommutedOpIndices(MI, CommuteOpIdx1, CommuteOpIdx2)) {
      bool HasDef = MI->getDesc().getNumDefs();
      unsigned Reg0 = HasDef ? MI->getOperand(0).getReg() : 0;
      unsigned Reg1 = MI->getOperand(CommuteOpIdx1).getReg();
      unsigned Reg2 = MI->getOperand(CommuteOpIdx2).getReg();
      bool Tied1 =
          0 == MI->getDesc().getOperandConstraint(CommuteOpIdx1, MCOI::TIED_TO);
      bool Tied2 =
          0 == MI->getDesc().getOperandConstraint(CommuteOpIdx2, MCOI::TIED_TO);

      // If either of the commutable operands are tied to the destination
      // then we can not commute + fold.
      if ((HasDef && Reg0 == Reg1 && Tied1) ||
          (HasDef && Reg0 == Reg2 && Tied2))
        return nullptr;

      MachineInstr *CommutedMI =
          commuteInstruction(MI, false, CommuteOpIdx1, CommuteOpIdx2);
      if (!CommutedMI) {
        // Unable to commute.
        return nullptr;
      }
      if (CommutedMI != MI) {
        // New instruction. We can't fold from this.
        CommutedMI->eraseFromParent();
        return nullptr;
      }

      // Attempt to fold with the commuted version of the instruction.
      NewMI = foldMemoryOperandImpl(MF, MI, CommuteOpIdx2, MOs, InsertPt,
                                    Size, Align, /*AllowCommute=*/false);
      if (NewMI)
        return NewMI;

      // Folding failed again - undo the commute before returning.
      MachineInstr *UncommutedMI =
          commuteInstruction(MI, false, CommuteOpIdx1, CommuteOpIdx2);
      if (!UncommutedMI) {
        // Unable to commute.
        return nullptr;
      }
      if (UncommutedMI != MI) {
        // New instruction. It doesn't need to be kept.
        UncommutedMI->eraseFromParent();
        return nullptr;
      }

      // Return here to prevent duplicate fuse failure report.
      return nullptr;
    }
  }

  // No fusion
  if (PrintFailedFusing && !MI->isCopy())
    dbgs() << "We failed to fuse operand " << OpNum << " in " << *MI;
  return nullptr;
}

/// Return true for all instructions that only update
/// the first 32 or 64-bits of the destination register and leave the rest
/// unmodified. This can be used to avoid folding loads if the instructions
/// only update part of the destination register, and the non-updated part is
/// not needed. e.g. cvtss2sd, sqrtss. Unfolding the load from these
/// instructions breaks the partial register dependency and it can improve
/// performance. e.g.:
///
///   movss (%rdi), %xmm0
///   cvtss2sd %xmm0, %xmm0
///
/// Instead of
///   cvtss2sd (%rdi), %xmm0
///
/// FIXME: This should be turned into a TSFlags.
///
static bool hasPartialRegUpdate(unsigned Opcode) {
  switch (Opcode) {
  case X86::CVTSI2SSrr:
  case X86::CVTSI2SSrm:
  case X86::CVTSI2SS64rr:
  case X86::CVTSI2SS64rm:
  case X86::CVTSI2SDrr:
  case X86::CVTSI2SDrm:
  case X86::CVTSI2SD64rr:
  case X86::CVTSI2SD64rm:
  case X86::CVTSD2SSrr:
  case X86::CVTSD2SSrm:
  case X86::Int_CVTSD2SSrr:
  case X86::Int_CVTSD2SSrm:
  case X86::CVTSS2SDrr:
  case X86::CVTSS2SDrm:
  case X86::Int_CVTSS2SDrr:
  case X86::Int_CVTSS2SDrm:
  case X86::RCPSSr:
  case X86::RCPSSm:
  case X86::RCPSSr_Int:
  case X86::RCPSSm_Int:
  case X86::ROUNDSDr:
  case X86::ROUNDSDm:
  case X86::ROUNDSDr_Int:
  case X86::ROUNDSSr:
  case X86::ROUNDSSm:
  case X86::ROUNDSSr_Int:
  case X86::RSQRTSSr:
  case X86::RSQRTSSm:
  case X86::RSQRTSSr_Int:
  case X86::RSQRTSSm_Int:
  case X86::SQRTSSr:
  case X86::SQRTSSm:
  case X86::SQRTSSr_Int:
  case X86::SQRTSSm_Int:
  case X86::SQRTSDr:
  case X86::SQRTSDm:
  case X86::SQRTSDr_Int:
  case X86::SQRTSDm_Int:
    return true;
  }

  return false;
}

/// Inform the ExeDepsFix pass how many idle
/// instructions we would like before a partial register update.
unsigned X86InstrInfo::
getPartialRegUpdateClearance(const MachineInstr *MI, unsigned OpNum,
                             const TargetRegisterInfo *TRI) const {
  if (OpNum != 0 || !hasPartialRegUpdate(MI->getOpcode()))
    return 0;

  // If MI is marked as reading Reg, the partial register update is wanted.
  const MachineOperand &MO = MI->getOperand(0);
  unsigned Reg = MO.getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    if (MO.readsReg() || MI->readsVirtualRegister(Reg))
      return 0;
  } else {
    if (MI->readsRegister(Reg, TRI))
      return 0;
  }

  // If any of the preceding 16 instructions are reading Reg, insert a
  // dependency breaking instruction.  The magic number is based on a few
  // Nehalem experiments.
  return 16;
}

// Return true for any instruction the copies the high bits of the first source
// operand into the unused high bits of the destination operand.
static bool hasUndefRegUpdate(unsigned Opcode) {
  switch (Opcode) {
  case X86::VCVTSI2SSrr:
  case X86::VCVTSI2SSrm:
  case X86::Int_VCVTSI2SSrr:
  case X86::Int_VCVTSI2SSrm:
  case X86::VCVTSI2SS64rr:
  case X86::VCVTSI2SS64rm:
  case X86::Int_VCVTSI2SS64rr:
  case X86::Int_VCVTSI2SS64rm:
  case X86::VCVTSI2SDrr:
  case X86::VCVTSI2SDrm:
  case X86::Int_VCVTSI2SDrr:
  case X86::Int_VCVTSI2SDrm:
  case X86::VCVTSI2SD64rr:
  case X86::VCVTSI2SD64rm:
  case X86::Int_VCVTSI2SD64rr:
  case X86::Int_VCVTSI2SD64rm:
  case X86::VCVTSD2SSrr:
  case X86::VCVTSD2SSrm:
  case X86::Int_VCVTSD2SSrr:
  case X86::Int_VCVTSD2SSrm:
  case X86::VCVTSS2SDrr:
  case X86::VCVTSS2SDrm:
  case X86::Int_VCVTSS2SDrr:
  case X86::Int_VCVTSS2SDrm:
  case X86::VRCPSSr:
  case X86::VRCPSSm:
  case X86::VRCPSSm_Int:
  case X86::VROUNDSDr:
  case X86::VROUNDSDm:
  case X86::VROUNDSDr_Int:
  case X86::VROUNDSSr:
  case X86::VROUNDSSm:
  case X86::VROUNDSSr_Int:
  case X86::VRSQRTSSr:
  case X86::VRSQRTSSm:
  case X86::VRSQRTSSm_Int:
  case X86::VSQRTSSr:
  case X86::VSQRTSSm:
  case X86::VSQRTSSm_Int:
  case X86::VSQRTSDr:
  case X86::VSQRTSDm:
  case X86::VSQRTSDm_Int:
    // AVX-512
  case X86::VCVTSD2SSZrr:
  case X86::VCVTSD2SSZrm:
  case X86::VCVTSS2SDZrr:
  case X86::VCVTSS2SDZrm:
    return true;
  }

  return false;
}

/// Inform the ExeDepsFix pass how many idle instructions we would like before
/// certain undef register reads.
///
/// This catches the VCVTSI2SD family of instructions:
///
/// vcvtsi2sdq %rax, %xmm0<undef>, %xmm14
///
/// We should to be careful *not* to catch VXOR idioms which are presumably
/// handled specially in the pipeline:
///
/// vxorps %xmm1<undef>, %xmm1<undef>, %xmm1
///
/// Like getPartialRegUpdateClearance, this makes a strong assumption that the
/// high bits that are passed-through are not live.
unsigned X86InstrInfo::
getUndefRegClearance(const MachineInstr *MI, unsigned &OpNum,
                     const TargetRegisterInfo *TRI) const {
  if (!hasUndefRegUpdate(MI->getOpcode()))
    return 0;

  // Set the OpNum parameter to the first source operand.
  OpNum = 1;

  const MachineOperand &MO = MI->getOperand(OpNum);
  if (MO.isUndef() && TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
    // Use the same magic number as getPartialRegUpdateClearance.
    return 16;
  }
  return 0;
}

void X86InstrInfo::
breakPartialRegDependency(MachineBasicBlock::iterator MI, unsigned OpNum,
                          const TargetRegisterInfo *TRI) const {
  unsigned Reg = MI->getOperand(OpNum).getReg();
  // If MI kills this register, the false dependence is already broken.
  if (MI->killsRegister(Reg, TRI))
    return;
  if (X86::VR128RegClass.contains(Reg)) {
    // These instructions are all floating point domain, so xorps is the best
    // choice.
    bool HasAVX = Subtarget.hasAVX();
    unsigned Opc = HasAVX ? X86::VXORPSrr : X86::XORPSrr;
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), get(Opc), Reg)
      .addReg(Reg, RegState::Undef).addReg(Reg, RegState::Undef);
  } else if (X86::VR256RegClass.contains(Reg)) {
    // Use vxorps to clear the full ymm register.
    // It wants to read and write the xmm sub-register.
    unsigned XReg = TRI->getSubReg(Reg, X86::sub_xmm);
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), get(X86::VXORPSrr), XReg)
      .addReg(XReg, RegState::Undef).addReg(XReg, RegState::Undef)
      .addReg(Reg, RegState::ImplicitDefine);
  } else
    return;
  MI->addRegisterKilled(Reg, TRI, true);
}

MachineInstr *X86InstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr *MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, int FrameIndex) const {
  // Check switch flag
  if (NoFusing) return nullptr;

  // Unless optimizing for size, don't fold to avoid partial
  // register update stalls
  if (!MF.getFunction()->optForSize() && hasPartialRegUpdate(MI->getOpcode()))
    return nullptr;

  const MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned Size = MFI->getObjectSize(FrameIndex);
  unsigned Alignment = MFI->getObjectAlignment(FrameIndex);
  // If the function stack isn't realigned we don't want to fold instructions
  // that need increased alignment.
  if (!RI.needsStackRealignment(MF))
    Alignment =
        std::min(Alignment, Subtarget.getFrameLowering()->getStackAlignment());
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    unsigned RCSize = 0;
    switch (MI->getOpcode()) {
    default: return nullptr;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; RCSize = 1; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; RCSize = 2; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; RCSize = 4; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; RCSize = 8; break;
    }
    // Check if it's safe to fold the load. If the size of the object is
    // narrower than the load width, then it's not.
    if (Size < RCSize)
      return nullptr;
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return nullptr;

  return foldMemoryOperandImpl(MF, MI, Ops[0],
                               MachineOperand::CreateFI(FrameIndex), InsertPt,
                               Size, Alignment, /*AllowCommute=*/true);
}

/// Check if \p LoadMI is a partial register load that we can't fold into \p MI
/// because the latter uses contents that wouldn't be defined in the folded
/// version.  For instance, this transformation isn't legal:
///   movss (%rdi), %xmm0
///   addps %xmm0, %xmm0
/// ->
///   addps (%rdi), %xmm0
///
/// But this one is:
///   movss (%rdi), %xmm0
///   addss %xmm0, %xmm0
/// ->
///   addss (%rdi), %xmm0
///
static bool isNonFoldablePartialRegisterLoad(const MachineInstr &LoadMI,
                                             const MachineInstr &UserMI,
                                             const MachineFunction &MF) {
  unsigned Opc = LoadMI.getOpcode();
  unsigned UserOpc = UserMI.getOpcode();
  unsigned RegSize =
      MF.getRegInfo().getRegClass(LoadMI.getOperand(0).getReg())->getSize();

  if ((Opc == X86::MOVSSrm || Opc == X86::VMOVSSrm) && RegSize > 4) {
    // These instructions only load 32 bits, we can't fold them if the
    // destination register is wider than 32 bits (4 bytes), and its user
    // instruction isn't scalar (SS).
    switch (UserOpc) {
    case X86::ADDSSrr_Int: case X86::VADDSSrr_Int:
    case X86::DIVSSrr_Int: case X86::VDIVSSrr_Int:
    case X86::MULSSrr_Int: case X86::VMULSSrr_Int:
    case X86::SUBSSrr_Int: case X86::VSUBSSrr_Int:
    case X86::VFMADDSSr132r_Int: case X86::VFNMADDSSr132r_Int:
    case X86::VFMADDSSr213r_Int: case X86::VFNMADDSSr213r_Int:
    case X86::VFMADDSSr231r_Int: case X86::VFNMADDSSr231r_Int:
    case X86::VFMSUBSSr132r_Int: case X86::VFNMSUBSSr132r_Int:
    case X86::VFMSUBSSr213r_Int: case X86::VFNMSUBSSr213r_Int:
    case X86::VFMSUBSSr231r_Int: case X86::VFNMSUBSSr231r_Int:
      return false;
    default:
      return true;
    }
  }

  if ((Opc == X86::MOVSDrm || Opc == X86::VMOVSDrm) && RegSize > 8) {
    // These instructions only load 64 bits, we can't fold them if the
    // destination register is wider than 64 bits (8 bytes), and its user
    // instruction isn't scalar (SD).
    switch (UserOpc) {
    case X86::ADDSDrr_Int: case X86::VADDSDrr_Int:
    case X86::DIVSDrr_Int: case X86::VDIVSDrr_Int:
    case X86::MULSDrr_Int: case X86::VMULSDrr_Int:
    case X86::SUBSDrr_Int: case X86::VSUBSDrr_Int:
    case X86::VFMADDSDr132r_Int: case X86::VFNMADDSDr132r_Int:
    case X86::VFMADDSDr213r_Int: case X86::VFNMADDSDr213r_Int:
    case X86::VFMADDSDr231r_Int: case X86::VFNMADDSDr231r_Int:
    case X86::VFMSUBSDr132r_Int: case X86::VFNMSUBSDr132r_Int:
    case X86::VFMSUBSDr213r_Int: case X86::VFNMSUBSDr213r_Int:
    case X86::VFMSUBSDr231r_Int: case X86::VFNMSUBSDr231r_Int:
      return false;
    default:
      return true;
    }
  }

  return false;
}

MachineInstr *X86InstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr *MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, MachineInstr *LoadMI) const {
  // If loading from a FrameIndex, fold directly from the FrameIndex.
  unsigned NumOps = LoadMI->getDesc().getNumOperands();
  int FrameIndex;
  if (isLoadFromStackSlot(LoadMI, FrameIndex)) {
    if (isNonFoldablePartialRegisterLoad(*LoadMI, *MI, MF))
      return nullptr;
    return foldMemoryOperandImpl(MF, MI, Ops, InsertPt, FrameIndex);
  }

  // Check switch flag
  if (NoFusing) return nullptr;

  // Avoid partial register update stalls unless optimizing for size.
  if (!MF.getFunction()->optForSize() && hasPartialRegUpdate(MI->getOpcode()))
    return nullptr;

  // Determine the alignment of the load.
  unsigned Alignment = 0;
  if (LoadMI->hasOneMemOperand())
    Alignment = (*LoadMI->memoperands_begin())->getAlignment();
  else
    switch (LoadMI->getOpcode()) {
    case X86::AVX2_SETALLONES:
    case X86::AVX_SET0:
      Alignment = 32;
      break;
    case X86::V_SET0:
    case X86::V_SETALLONES:
      Alignment = 16;
      break;
    case X86::FsFLD0SD:
      Alignment = 8;
      break;
    case X86::FsFLD0SS:
      Alignment = 4;
      break;
    default:
      return nullptr;
    }
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return nullptr;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return nullptr;

  // Make sure the subregisters match.
  // Otherwise we risk changing the size of the load.
  if (LoadMI->getOperand(0).getSubReg() != MI->getOperand(Ops[0]).getSubReg())
    return nullptr;

  SmallVector<MachineOperand,X86::AddrNumOperands> MOs;
  switch (LoadMI->getOpcode()) {
  case X86::V_SET0:
  case X86::V_SETALLONES:
  case X86::AVX2_SETALLONES:
  case X86::AVX_SET0:
  case X86::FsFLD0SD:
  case X86::FsFLD0SS: {
    // Folding a V_SET0 or V_SETALLONES as a load, to ease register pressure.
    // Create a constant-pool entry and operands to load from it.

    // Medium and large mode can't fold loads this way.
    if (MF.getTarget().getCodeModel() != CodeModel::Small &&
        MF.getTarget().getCodeModel() != CodeModel::Kernel)
      return nullptr;

    // x86-32 PIC requires a PIC base register for constant pools.
    unsigned PICBase = 0;
    if (MF.getTarget().getRelocationModel() == Reloc::PIC_) {
      if (Subtarget.is64Bit())
        PICBase = X86::RIP;
      else
        // FIXME: PICBase = getGlobalBaseReg(&MF);
        // This doesn't work for several reasons.
        // 1. GlobalBaseReg may have been spilled.
        // 2. It may not be live at MI.
        return nullptr;
    }

    // Create a constant-pool entry.
    MachineConstantPool &MCP = *MF.getConstantPool();
    Type *Ty;
    unsigned Opc = LoadMI->getOpcode();
    if (Opc == X86::FsFLD0SS)
      Ty = Type::getFloatTy(MF.getFunction()->getContext());
    else if (Opc == X86::FsFLD0SD)
      Ty = Type::getDoubleTy(MF.getFunction()->getContext());
    else if (Opc == X86::AVX2_SETALLONES || Opc == X86::AVX_SET0)
      Ty = VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()), 8);
    else
      Ty = VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()), 4);

    bool IsAllOnes = (Opc == X86::V_SETALLONES || Opc == X86::AVX2_SETALLONES);
    const Constant *C = IsAllOnes ? Constant::getAllOnesValue(Ty) :
                                    Constant::getNullValue(Ty);
    unsigned CPI = MCP.getConstantPoolIndex(C, Alignment);

    // Create operands to load from the constant pool entry.
    MOs.push_back(MachineOperand::CreateReg(PICBase, false));
    MOs.push_back(MachineOperand::CreateImm(1));
    MOs.push_back(MachineOperand::CreateReg(0, false));
    MOs.push_back(MachineOperand::CreateCPI(CPI, 0));
    MOs.push_back(MachineOperand::CreateReg(0, false));
    break;
  }
  default: {
    if (isNonFoldablePartialRegisterLoad(*LoadMI, *MI, MF))
      return nullptr;

    // Folding a normal load. Just copy the load's address operands.
    MOs.append(LoadMI->operands_begin() + NumOps - X86::AddrNumOperands,
               LoadMI->operands_begin() + NumOps);
    break;
  }
  }
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, InsertPt,
                               /*Size=*/0, Alignment, /*AllowCommute=*/true);
}

bool X86InstrInfo::unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
    MemOp2RegOpTable.find(MI->getOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & TB_INDEX_MASK;
  bool FoldedLoad = I->second.second & TB_FOLDED_LOAD;
  bool FoldedStore = I->second.second & TB_FOLDED_STORE;
  if (UnfoldLoad && !FoldedLoad)
    return false;
  UnfoldLoad &= FoldedLoad;
  if (UnfoldStore && !FoldedStore)
    return false;
  UnfoldStore &= FoldedStore;

  const MCInstrDesc &MCID = get(Opc);
  const TargetRegisterClass *RC = getRegClass(MCID, Index, &RI, MF);
  // TODO: Check if 32-byte or greater accesses are slow too?
  if (!MI->hasOneMemOperand() &&
      RC == &X86::VR128RegClass &&
      Subtarget.isUnalignedMem16Slow())
    // Without memoperands, loadRegFromAddr and storeRegToStackSlot will
    // conservatively assume the address is unaligned. That's bad for
    // performance.
    return false;
  SmallVector<MachineOperand, X86::AddrNumOperands> AddrOps;
  SmallVector<MachineOperand,2> BeforeOps;
  SmallVector<MachineOperand,2> AfterOps;
  SmallVector<MachineOperand,4> ImpOps;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &Op = MI->getOperand(i);
    if (i >= Index && i < Index + X86::AddrNumOperands)
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
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractLoadMemRefs(MI->memoperands_begin(),
                            MI->memoperands_end());
    loadRegFromAddr(MF, Reg, AddrOps, RC, MMOs.first, MMOs.second, NewMIs);
    if (UnfoldStore) {
      // Address operands cannot be marked isKill.
      for (unsigned i = 1; i != 1 + X86::AddrNumOperands; ++i) {
        MachineOperand &MO = NewMIs[0]->getOperand(i);
        if (MO.isReg())
          MO.setIsKill(false);
      }
    }
  }

  // Emit the data processing instruction.
  MachineInstr *DataMI = MF.CreateMachineInstr(MCID, MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, DataMI);

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
  switch (DataMI->getOpcode()) {
  default: break;
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP8ri: {
    MachineOperand &MO0 = DataMI->getOperand(0);
    MachineOperand &MO1 = DataMI->getOperand(1);
    if (MO1.getImm() == 0) {
      unsigned NewOpc;
      switch (DataMI->getOpcode()) {
      default: llvm_unreachable("Unreachable!");
      case X86::CMP64ri8:
      case X86::CMP64ri32: NewOpc = X86::TEST64rr; break;
      case X86::CMP32ri8:
      case X86::CMP32ri:   NewOpc = X86::TEST32rr; break;
      case X86::CMP16ri8:
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
    const TargetRegisterClass *DstRC = getRegClass(MCID, 0, &RI, MF);
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractStoreMemRefs(MI->memoperands_begin(),
                             MI->memoperands_end());
    storeRegToAddr(MF, Reg, true, AddrOps, DstRC, MMOs.first, MMOs.second, NewMIs);
  }

  return true;
}

bool
X86InstrInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                  SmallVectorImpl<SDNode*> &NewNodes) const {
  if (!N->isMachineOpcode())
    return false;

  DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
    MemOp2RegOpTable.find(N->getMachineOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & TB_INDEX_MASK;
  bool FoldedLoad = I->second.second & TB_FOLDED_LOAD;
  bool FoldedStore = I->second.second & TB_FOLDED_STORE;
  const MCInstrDesc &MCID = get(Opc);
  MachineFunction &MF = DAG.getMachineFunction();
  const TargetRegisterClass *RC = getRegClass(MCID, Index, &RI, MF);
  unsigned NumDefs = MCID.NumDefs;
  std::vector<SDValue> AddrOps;
  std::vector<SDValue> BeforeOps;
  std::vector<SDValue> AfterOps;
  SDLoc dl(N);
  unsigned NumOps = N->getNumOperands();
  for (unsigned i = 0; i != NumOps-1; ++i) {
    SDValue Op = N->getOperand(i);
    if (i >= Index-NumDefs && i < Index-NumDefs + X86::AddrNumOperands)
      AddrOps.push_back(Op);
    else if (i < Index-NumDefs)
      BeforeOps.push_back(Op);
    else if (i > Index-NumDefs)
      AfterOps.push_back(Op);
  }
  SDValue Chain = N->getOperand(NumOps-1);
  AddrOps.push_back(Chain);

  // Emit the load instruction.
  SDNode *Load = nullptr;
  if (FoldedLoad) {
    EVT VT = *RC->vt_begin();
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractLoadMemRefs(cast<MachineSDNode>(N)->memoperands_begin(),
                            cast<MachineSDNode>(N)->memoperands_end());
    if (!(*MMOs.first) &&
        RC == &X86::VR128RegClass &&
        Subtarget.isUnalignedMem16Slow())
      // Do not introduce a slow unaligned load.
      return false;
    // FIXME: If a VR128 can have size 32, we should be checking if a 32-byte
    // memory access is slow above.
    unsigned Alignment = RC->getSize() == 32 ? 32 : 16;
    bool isAligned = (*MMOs.first) &&
                     (*MMOs.first)->getAlignment() >= Alignment;
    Load = DAG.getMachineNode(getLoadRegOpcode(0, RC, isAligned, Subtarget), dl,
                              VT, MVT::Other, AddrOps);
    NewNodes.push_back(Load);

    // Preserve memory reference information.
    cast<MachineSDNode>(Load)->setMemRefs(MMOs.first, MMOs.second);
  }

  // Emit the data processing instruction.
  std::vector<EVT> VTs;
  const TargetRegisterClass *DstRC = nullptr;
  if (MCID.getNumDefs() > 0) {
    DstRC = getRegClass(MCID, 0, &RI, MF);
    VTs.push_back(*DstRC->vt_begin());
  }
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    EVT VT = N->getValueType(i);
    if (VT != MVT::Other && i >= (unsigned)MCID.getNumDefs())
      VTs.push_back(VT);
  }
  if (Load)
    BeforeOps.push_back(SDValue(Load, 0));
  BeforeOps.insert(BeforeOps.end(), AfterOps.begin(), AfterOps.end());
  SDNode *NewNode= DAG.getMachineNode(Opc, dl, VTs, BeforeOps);
  NewNodes.push_back(NewNode);

  // Emit the store instruction.
  if (FoldedStore) {
    AddrOps.pop_back();
    AddrOps.push_back(SDValue(NewNode, 0));
    AddrOps.push_back(Chain);
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractStoreMemRefs(cast<MachineSDNode>(N)->memoperands_begin(),
                             cast<MachineSDNode>(N)->memoperands_end());
    if (!(*MMOs.first) &&
        RC == &X86::VR128RegClass &&
        Subtarget.isUnalignedMem16Slow())
      // Do not introduce a slow unaligned store.
      return false;
    // FIXME: If a VR128 can have size 32, we should be checking if a 32-byte
    // memory access is slow above.
    unsigned Alignment = RC->getSize() == 32 ? 32 : 16;
    bool isAligned = (*MMOs.first) &&
                     (*MMOs.first)->getAlignment() >= Alignment;
    SDNode *Store =
        DAG.getMachineNode(getStoreRegOpcode(0, DstRC, isAligned, Subtarget),
                           dl, MVT::Other, AddrOps);
    NewNodes.push_back(Store);

    // Preserve memory reference information.
    cast<MachineSDNode>(Store)->setMemRefs(MMOs.first, MMOs.second);
  }

  return true;
}

unsigned X86InstrInfo::getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore,
                                      unsigned *LoadRegIndex) const {
  DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
    MemOp2RegOpTable.find(Opc);
  if (I == MemOp2RegOpTable.end())
    return 0;
  bool FoldedLoad = I->second.second & TB_FOLDED_LOAD;
  bool FoldedStore = I->second.second & TB_FOLDED_STORE;
  if (UnfoldLoad && !FoldedLoad)
    return 0;
  if (UnfoldStore && !FoldedStore)
    return 0;
  if (LoadRegIndex)
    *LoadRegIndex = I->second.second & TB_INDEX_MASK;
  return I->second.first;
}

bool
X86InstrInfo::areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                     int64_t &Offset1, int64_t &Offset2) const {
  if (!Load1->isMachineOpcode() || !Load2->isMachineOpcode())
    return false;
  unsigned Opc1 = Load1->getMachineOpcode();
  unsigned Opc2 = Load2->getMachineOpcode();
  switch (Opc1) {
  default: return false;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp32m:
  case X86::LD_Fp64m:
  case X86::LD_Fp80m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::FsMOVAPSrm:
  case X86::FsMOVAPDrm:
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  // AVX load instructions
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::FsVMOVAPSrm:
  case X86::FsVMOVAPDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVDQAYrm:
  case X86::VMOVDQUYrm:
    break;
  }
  switch (Opc2) {
  default: return false;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp32m:
  case X86::LD_Fp64m:
  case X86::LD_Fp80m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::FsMOVAPSrm:
  case X86::FsMOVAPDrm:
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  // AVX load instructions
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::FsVMOVAPSrm:
  case X86::FsVMOVAPDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVDQAYrm:
  case X86::VMOVDQUYrm:
    break;
  }

  // Check if chain operands and base addresses match.
  if (Load1->getOperand(0) != Load2->getOperand(0) ||
      Load1->getOperand(5) != Load2->getOperand(5))
    return false;
  // Segment operands should match as well.
  if (Load1->getOperand(4) != Load2->getOperand(4))
    return false;
  // Scale should be 1, Index should be Reg0.
  if (Load1->getOperand(1) == Load2->getOperand(1) &&
      Load1->getOperand(2) == Load2->getOperand(2)) {
    if (cast<ConstantSDNode>(Load1->getOperand(1))->getZExtValue() != 1)
      return false;

    // Now let's examine the displacements.
    if (isa<ConstantSDNode>(Load1->getOperand(3)) &&
        isa<ConstantSDNode>(Load2->getOperand(3))) {
      Offset1 = cast<ConstantSDNode>(Load1->getOperand(3))->getSExtValue();
      Offset2 = cast<ConstantSDNode>(Load2->getOperand(3))->getSExtValue();
      return true;
    }
  }
  return false;
}

bool X86InstrInfo::shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                           int64_t Offset1, int64_t Offset2,
                                           unsigned NumLoads) const {
  assert(Offset2 > Offset1);
  if ((Offset2 - Offset1) / 8 > 64)
    return false;

  unsigned Opc1 = Load1->getMachineOpcode();
  unsigned Opc2 = Load2->getMachineOpcode();
  if (Opc1 != Opc2)
    return false;  // FIXME: overly conservative?

  switch (Opc1) {
  default: break;
  case X86::LD_Fp32m:
  case X86::LD_Fp64m:
  case X86::LD_Fp80m:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    return false;
  }

  EVT VT = Load1->getValueType(0);
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    // XMM registers. In 64-bit mode we can be a bit more aggressive since we
    // have 16 of them to play with.
    if (Subtarget.is64Bit()) {
      if (NumLoads >= 3)
        return false;
    } else if (NumLoads) {
      return false;
    }
    break;
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
  case MVT::f32:
  case MVT::f64:
    if (NumLoads)
      return false;
    break;
  }

  return true;
}

bool X86InstrInfo::shouldScheduleAdjacent(MachineInstr* First,
                                          MachineInstr *Second) const {
  // Check if this processor supports macro-fusion. Since this is a minor
  // heuristic, we haven't specifically reserved a feature. hasAVX is a decent
  // proxy for SandyBridge+.
  if (!Subtarget.hasAVX())
    return false;

  enum {
    FuseTest,
    FuseCmp,
    FuseInc
  } FuseKind;

  switch(Second->getOpcode()) {
  default:
    return false;
  case X86::JE_1:
  case X86::JNE_1:
  case X86::JL_1:
  case X86::JLE_1:
  case X86::JG_1:
  case X86::JGE_1:
    FuseKind = FuseInc;
    break;
  case X86::JB_1:
  case X86::JBE_1:
  case X86::JA_1:
  case X86::JAE_1:
    FuseKind = FuseCmp;
    break;
  case X86::JS_1:
  case X86::JNS_1:
  case X86::JP_1:
  case X86::JNP_1:
  case X86::JO_1:
  case X86::JNO_1:
    FuseKind = FuseTest;
    break;
  }
  switch (First->getOpcode()) {
  default:
    return false;
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
  case X86::TEST8ri:
  case X86::TEST16ri:
  case X86::TEST32ri:
  case X86::TEST32i32:
  case X86::TEST64i32:
  case X86::TEST64ri32:
  case X86::TEST8rm:
  case X86::TEST16rm:
  case X86::TEST32rm:
  case X86::TEST64rm:
  case X86::TEST8ri_NOREX:
  case X86::AND16i16:
  case X86::AND16ri:
  case X86::AND16ri8:
  case X86::AND16rm:
  case X86::AND16rr:
  case X86::AND32i32:
  case X86::AND32ri:
  case X86::AND32ri8:
  case X86::AND32rm:
  case X86::AND32rr:
  case X86::AND64i32:
  case X86::AND64ri32:
  case X86::AND64ri8:
  case X86::AND64rm:
  case X86::AND64rr:
  case X86::AND8i8:
  case X86::AND8ri:
  case X86::AND8rm:
  case X86::AND8rr:
    return true;
  case X86::CMP16i16:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP16rm:
  case X86::CMP16rr:
  case X86::CMP32i32:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP32rm:
  case X86::CMP32rr:
  case X86::CMP64i32:
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP64rm:
  case X86::CMP64rr:
  case X86::CMP8i8:
  case X86::CMP8ri:
  case X86::CMP8rm:
  case X86::CMP8rr:
  case X86::ADD16i16:
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri8_DB:
  case X86::ADD16ri_DB:
  case X86::ADD16rm:
  case X86::ADD16rr:
  case X86::ADD16rr_DB:
  case X86::ADD32i32:
  case X86::ADD32ri:
  case X86::ADD32ri8:
  case X86::ADD32ri8_DB:
  case X86::ADD32ri_DB:
  case X86::ADD32rm:
  case X86::ADD32rr:
  case X86::ADD32rr_DB:
  case X86::ADD64i32:
  case X86::ADD64ri32:
  case X86::ADD64ri32_DB:
  case X86::ADD64ri8:
  case X86::ADD64ri8_DB:
  case X86::ADD64rm:
  case X86::ADD64rr:
  case X86::ADD64rr_DB:
  case X86::ADD8i8:
  case X86::ADD8mi:
  case X86::ADD8mr:
  case X86::ADD8ri:
  case X86::ADD8rm:
  case X86::ADD8rr:
  case X86::SUB16i16:
  case X86::SUB16ri:
  case X86::SUB16ri8:
  case X86::SUB16rm:
  case X86::SUB16rr:
  case X86::SUB32i32:
  case X86::SUB32ri:
  case X86::SUB32ri8:
  case X86::SUB32rm:
  case X86::SUB32rr:
  case X86::SUB64i32:
  case X86::SUB64ri32:
  case X86::SUB64ri8:
  case X86::SUB64rm:
  case X86::SUB64rr:
  case X86::SUB8i8:
  case X86::SUB8ri:
  case X86::SUB8rm:
  case X86::SUB8rr:
    return FuseKind == FuseCmp || FuseKind == FuseInc;
  case X86::INC16r:
  case X86::INC32r:
  case X86::INC64r:
  case X86::INC8r:
  case X86::DEC16r:
  case X86::DEC32r:
  case X86::DEC64r:
  case X86::DEC8r:
    return FuseKind == FuseInc;
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

/// Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
/// TODO: Eliminate this and move the code to X86MachineFunctionInfo.
///
unsigned X86InstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  assert(!Subtarget.is64Bit() &&
         "X86-64 PIC uses RIP relative addressing");

  X86MachineFunctionInfo *X86FI = MF->getInfo<X86MachineFunctionInfo>();
  unsigned GlobalBaseReg = X86FI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Create the register. The code to initialize it is inserted
  // later, by the CGBR pass (below).
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  GlobalBaseReg = RegInfo.createVirtualRegister(&X86::GR32_NOSPRegClass);
  X86FI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}

// These are the replaceable SSE instructions. Some of these have Int variants
// that we don't include here. We don't want to replace instructions selected
// by intrinsics.
static const uint16_t ReplaceableInstrs[][3] = {
  //PackedSingle     PackedDouble    PackedInt
  { X86::MOVAPSmr,   X86::MOVAPDmr,  X86::MOVDQAmr  },
  { X86::MOVAPSrm,   X86::MOVAPDrm,  X86::MOVDQArm  },
  { X86::MOVAPSrr,   X86::MOVAPDrr,  X86::MOVDQArr  },
  { X86::MOVUPSmr,   X86::MOVUPDmr,  X86::MOVDQUmr  },
  { X86::MOVUPSrm,   X86::MOVUPDrm,  X86::MOVDQUrm  },
  { X86::MOVLPSmr,   X86::MOVLPDmr,  X86::MOVPQI2QImr  },
  { X86::MOVNTPSmr,  X86::MOVNTPDmr, X86::MOVNTDQmr },
  { X86::ANDNPSrm,   X86::ANDNPDrm,  X86::PANDNrm   },
  { X86::ANDNPSrr,   X86::ANDNPDrr,  X86::PANDNrr   },
  { X86::ANDPSrm,    X86::ANDPDrm,   X86::PANDrm    },
  { X86::ANDPSrr,    X86::ANDPDrr,   X86::PANDrr    },
  { X86::ORPSrm,     X86::ORPDrm,    X86::PORrm     },
  { X86::ORPSrr,     X86::ORPDrr,    X86::PORrr     },
  { X86::XORPSrm,    X86::XORPDrm,   X86::PXORrm    },
  { X86::XORPSrr,    X86::XORPDrr,   X86::PXORrr    },
  // AVX 128-bit support
  { X86::VMOVAPSmr,  X86::VMOVAPDmr,  X86::VMOVDQAmr  },
  { X86::VMOVAPSrm,  X86::VMOVAPDrm,  X86::VMOVDQArm  },
  { X86::VMOVAPSrr,  X86::VMOVAPDrr,  X86::VMOVDQArr  },
  { X86::VMOVUPSmr,  X86::VMOVUPDmr,  X86::VMOVDQUmr  },
  { X86::VMOVUPSrm,  X86::VMOVUPDrm,  X86::VMOVDQUrm  },
  { X86::VMOVLPSmr,  X86::VMOVLPDmr,  X86::VMOVPQI2QImr  },
  { X86::VMOVNTPSmr, X86::VMOVNTPDmr, X86::VMOVNTDQmr },
  { X86::VANDNPSrm,  X86::VANDNPDrm,  X86::VPANDNrm   },
  { X86::VANDNPSrr,  X86::VANDNPDrr,  X86::VPANDNrr   },
  { X86::VANDPSrm,   X86::VANDPDrm,   X86::VPANDrm    },
  { X86::VANDPSrr,   X86::VANDPDrr,   X86::VPANDrr    },
  { X86::VORPSrm,    X86::VORPDrm,    X86::VPORrm     },
  { X86::VORPSrr,    X86::VORPDrr,    X86::VPORrr     },
  { X86::VXORPSrm,   X86::VXORPDrm,   X86::VPXORrm    },
  { X86::VXORPSrr,   X86::VXORPDrr,   X86::VPXORrr    },
  // AVX 256-bit support
  { X86::VMOVAPSYmr,   X86::VMOVAPDYmr,   X86::VMOVDQAYmr  },
  { X86::VMOVAPSYrm,   X86::VMOVAPDYrm,   X86::VMOVDQAYrm  },
  { X86::VMOVAPSYrr,   X86::VMOVAPDYrr,   X86::VMOVDQAYrr  },
  { X86::VMOVUPSYmr,   X86::VMOVUPDYmr,   X86::VMOVDQUYmr  },
  { X86::VMOVUPSYrm,   X86::VMOVUPDYrm,   X86::VMOVDQUYrm  },
  { X86::VMOVNTPSYmr,  X86::VMOVNTPDYmr,  X86::VMOVNTDQYmr }
};

static const uint16_t ReplaceableInstrsAVX2[][3] = {
  //PackedSingle       PackedDouble       PackedInt
  { X86::VANDNPSYrm,   X86::VANDNPDYrm,   X86::VPANDNYrm   },
  { X86::VANDNPSYrr,   X86::VANDNPDYrr,   X86::VPANDNYrr   },
  { X86::VANDPSYrm,    X86::VANDPDYrm,    X86::VPANDYrm    },
  { X86::VANDPSYrr,    X86::VANDPDYrr,    X86::VPANDYrr    },
  { X86::VORPSYrm,     X86::VORPDYrm,     X86::VPORYrm     },
  { X86::VORPSYrr,     X86::VORPDYrr,     X86::VPORYrr     },
  { X86::VXORPSYrm,    X86::VXORPDYrm,    X86::VPXORYrm    },
  { X86::VXORPSYrr,    X86::VXORPDYrr,    X86::VPXORYrr    },
  { X86::VEXTRACTF128mr, X86::VEXTRACTF128mr, X86::VEXTRACTI128mr },
  { X86::VEXTRACTF128rr, X86::VEXTRACTF128rr, X86::VEXTRACTI128rr },
  { X86::VINSERTF128rm,  X86::VINSERTF128rm,  X86::VINSERTI128rm },
  { X86::VINSERTF128rr,  X86::VINSERTF128rr,  X86::VINSERTI128rr },
  { X86::VPERM2F128rm,   X86::VPERM2F128rm,   X86::VPERM2I128rm },
  { X86::VPERM2F128rr,   X86::VPERM2F128rr,   X86::VPERM2I128rr },
  { X86::VBROADCASTSSrm, X86::VBROADCASTSSrm, X86::VPBROADCASTDrm},
  { X86::VBROADCASTSSrr, X86::VBROADCASTSSrr, X86::VPBROADCASTDrr},
  { X86::VBROADCASTSSYrr, X86::VBROADCASTSSYrr, X86::VPBROADCASTDYrr},
  { X86::VBROADCASTSSYrm, X86::VBROADCASTSSYrm, X86::VPBROADCASTDYrm},
  { X86::VBROADCASTSDYrr, X86::VBROADCASTSDYrr, X86::VPBROADCASTQYrr},
  { X86::VBROADCASTSDYrm, X86::VBROADCASTSDYrm, X86::VPBROADCASTQYrm}
};

// FIXME: Some shuffle and unpack instructions have equivalents in different
// domains, but they require a bit more work than just switching opcodes.

static const uint16_t *lookup(unsigned opcode, unsigned domain) {
  for (unsigned i = 0, e = array_lengthof(ReplaceableInstrs); i != e; ++i)
    if (ReplaceableInstrs[i][domain-1] == opcode)
      return ReplaceableInstrs[i];
  return nullptr;
}

static const uint16_t *lookupAVX2(unsigned opcode, unsigned domain) {
  for (unsigned i = 0, e = array_lengthof(ReplaceableInstrsAVX2); i != e; ++i)
    if (ReplaceableInstrsAVX2[i][domain-1] == opcode)
      return ReplaceableInstrsAVX2[i];
  return nullptr;
}

std::pair<uint16_t, uint16_t>
X86InstrInfo::getExecutionDomain(const MachineInstr *MI) const {
  uint16_t domain = (MI->getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  bool hasAVX2 = Subtarget.hasAVX2();
  uint16_t validDomains = 0;
  if (domain && lookup(MI->getOpcode(), domain))
    validDomains = 0xe;
  else if (domain && lookupAVX2(MI->getOpcode(), domain))
    validDomains = hasAVX2 ? 0xe : 0x6;
  return std::make_pair(domain, validDomains);
}

void X86InstrInfo::setExecutionDomain(MachineInstr *MI, unsigned Domain) const {
  assert(Domain>0 && Domain<4 && "Invalid execution domain");
  uint16_t dom = (MI->getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  assert(dom && "Not an SSE instruction");
  const uint16_t *table = lookup(MI->getOpcode(), dom);
  if (!table) { // try the other table
    assert((Subtarget.hasAVX2() || Domain < 3) &&
           "256-bit vector operations only available in AVX2");
    table = lookupAVX2(MI->getOpcode(), dom);
  }
  assert(table && "Cannot change domain");
  MI->setDesc(get(table[Domain-1]));
}

/// Return the noop instruction to use for a noop.
void X86InstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  NopInst.setOpcode(X86::NOOP);
}

// This code must remain in sync with getJumpInstrTableEntryBound in this class!
// In particular, getJumpInstrTableEntryBound must always return an upper bound
// on the encoding lengths of the instructions generated by
// getUnconditionalBranch and getTrap.
void X86InstrInfo::getUnconditionalBranch(
    MCInst &Branch, const MCSymbolRefExpr *BranchTarget) const {
  Branch.setOpcode(X86::JMP_1);
  Branch.addOperand(MCOperand::createExpr(BranchTarget));
}

// This code must remain in sync with getJumpInstrTableEntryBound in this class!
// In particular, getJumpInstrTableEntryBound must always return an upper bound
// on the encoding lengths of the instructions generated by
// getUnconditionalBranch and getTrap.
void X86InstrInfo::getTrap(MCInst &MI) const {
  MI.setOpcode(X86::TRAP);
}

// See getTrap and getUnconditionalBranch for conditions on the value returned
// by this function.
unsigned X86InstrInfo::getJumpInstrTableEntryBound() const {
  // 5 bytes suffice: JMP_4 Symbol@PLT is uses 1 byte (E9) for the JMP_4 and 4
  // bytes for the symbol offset. And TRAP is ud2, which is two bytes (0F 0B).
  return 5;
}

bool X86InstrInfo::isHighLatencyDef(int opc) const {
  switch (opc) {
  default: return false;
  case X86::DIVSDrm:
  case X86::DIVSDrm_Int:
  case X86::DIVSDrr:
  case X86::DIVSDrr_Int:
  case X86::DIVSSrm:
  case X86::DIVSSrm_Int:
  case X86::DIVSSrr:
  case X86::DIVSSrr_Int:
  case X86::SQRTPDm:
  case X86::SQRTPDr:
  case X86::SQRTPSm:
  case X86::SQRTPSr:
  case X86::SQRTSDm:
  case X86::SQRTSDm_Int:
  case X86::SQRTSDr:
  case X86::SQRTSDr_Int:
  case X86::SQRTSSm:
  case X86::SQRTSSm_Int:
  case X86::SQRTSSr:
  case X86::SQRTSSr_Int:
  // AVX instructions with high latency
  case X86::VDIVSDrm:
  case X86::VDIVSDrm_Int:
  case X86::VDIVSDrr:
  case X86::VDIVSDrr_Int:
  case X86::VDIVSSrm:
  case X86::VDIVSSrm_Int:
  case X86::VDIVSSrr:
  case X86::VDIVSSrr_Int:
  case X86::VSQRTPDm:
  case X86::VSQRTPDr:
  case X86::VSQRTPSm:
  case X86::VSQRTPSr:
  case X86::VSQRTSDm:
  case X86::VSQRTSDm_Int:
  case X86::VSQRTSDr:
  case X86::VSQRTSSm:
  case X86::VSQRTSSm_Int:
  case X86::VSQRTSSr:
  case X86::VSQRTPDZm:
  case X86::VSQRTPDZr:
  case X86::VSQRTPSZm:
  case X86::VSQRTPSZr:
  case X86::VSQRTSDZm:
  case X86::VSQRTSDZm_Int:
  case X86::VSQRTSDZr:
  case X86::VSQRTSSZm_Int:
  case X86::VSQRTSSZr:
  case X86::VSQRTSSZm:
  case X86::VDIVSDZrm:
  case X86::VDIVSDZrr:
  case X86::VDIVSSZrm:
  case X86::VDIVSSZrr:

  case X86::VGATHERQPSZrm:
  case X86::VGATHERQPDZrm:
  case X86::VGATHERDPDZrm:
  case X86::VGATHERDPSZrm:
  case X86::VPGATHERQDZrm:
  case X86::VPGATHERQQZrm:
  case X86::VPGATHERDDZrm:
  case X86::VPGATHERDQZrm:
  case X86::VSCATTERQPDZmr:
  case X86::VSCATTERQPSZmr:
  case X86::VSCATTERDPDZmr:
  case X86::VSCATTERDPSZmr:
  case X86::VPSCATTERQDZmr:
  case X86::VPSCATTERQQZmr:
  case X86::VPSCATTERDDZmr:
  case X86::VPSCATTERDQZmr:
    return true;
  }
}

bool X86InstrInfo::
hasHighOperandLatency(const TargetSchedModel &SchedModel,
                      const MachineRegisterInfo *MRI,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx) const {
  return isHighLatencyDef(DefMI->getOpcode());
}

bool X86InstrInfo::hasReassociableOperands(const MachineInstr &Inst,
                                           const MachineBasicBlock *MBB) const {
  assert((Inst.getNumOperands() == 3 || Inst.getNumOperands() == 4) &&
         "Reassociation needs binary operators");

  // Integer binary math/logic instructions have a third source operand:
  // the EFLAGS register. That operand must be both defined here and never
  // used; ie, it must be dead. If the EFLAGS operand is live, then we can
  // not change anything because rearranging the operands could affect other
  // instructions that depend on the exact status flags (zero, sign, etc.)
  // that are set by using these particular operands with this operation.
  if (Inst.getNumOperands() == 4) {
    assert(Inst.getOperand(3).isReg() &&
           Inst.getOperand(3).getReg() == X86::EFLAGS &&
           "Unexpected operand in reassociable instruction");
    if (!Inst.getOperand(3).isDead())
      return false;
  }

  return TargetInstrInfo::hasReassociableOperands(Inst, MBB);
}

// TODO: There are many more machine instruction opcodes to match:
//       1. Other data types (integer, vectors)
//       2. Other math / logic operations (xor, or)
//       3. Other forms of the same operation (intrinsics and other variants)
bool X86InstrInfo::isAssociativeAndCommutative(const MachineInstr &Inst) const {
  switch (Inst.getOpcode()) {
  case X86::AND8rr:
  case X86::AND16rr:
  case X86::AND32rr:
  case X86::AND64rr:
  case X86::OR8rr:
  case X86::OR16rr:
  case X86::OR32rr:
  case X86::OR64rr:
  case X86::XOR8rr:
  case X86::XOR16rr:
  case X86::XOR32rr:
  case X86::XOR64rr:
  case X86::IMUL16rr:
  case X86::IMUL32rr:
  case X86::IMUL64rr:
  case X86::PANDrr:
  case X86::PORrr:
  case X86::PXORrr:
  case X86::VPANDrr:
  case X86::VPANDYrr:
  case X86::VPORrr:
  case X86::VPORYrr:
  case X86::VPXORrr:
  case X86::VPXORYrr:
  // Normal min/max instructions are not commutative because of NaN and signed
  // zero semantics, but these are. Thus, there's no need to check for global
  // relaxed math; the instructions themselves have the properties we need.
  case X86::MAXCPDrr:
  case X86::MAXCPSrr:
  case X86::MAXCSDrr:
  case X86::MAXCSSrr:
  case X86::MINCPDrr:
  case X86::MINCPSrr:
  case X86::MINCSDrr:
  case X86::MINCSSrr:
  case X86::VMAXCPDrr:
  case X86::VMAXCPSrr:
  case X86::VMAXCPDYrr:
  case X86::VMAXCPSYrr:
  case X86::VMAXCSDrr:
  case X86::VMAXCSSrr:
  case X86::VMINCPDrr:
  case X86::VMINCPSrr:
  case X86::VMINCPDYrr:
  case X86::VMINCPSYrr:
  case X86::VMINCSDrr:
  case X86::VMINCSSrr:
    return true;
  case X86::ADDPDrr:
  case X86::ADDPSrr:
  case X86::ADDSDrr:
  case X86::ADDSSrr:
  case X86::MULPDrr:
  case X86::MULPSrr:
  case X86::MULSDrr:
  case X86::MULSSrr:
  case X86::VADDPDrr:
  case X86::VADDPSrr:
  case X86::VADDPDYrr:
  case X86::VADDPSYrr:
  case X86::VADDSDrr:
  case X86::VADDSSrr:
  case X86::VMULPDrr:
  case X86::VMULPSrr:
  case X86::VMULPDYrr:
  case X86::VMULPSYrr:
  case X86::VMULSDrr:
  case X86::VMULSSrr:
    return Inst.getParent()->getParent()->getTarget().Options.UnsafeFPMath;
  default:
    return false;
  }
}

/// This is an architecture-specific helper function of reassociateOps.
/// Set special operand attributes for new instructions after reassociation.
void X86InstrInfo::setSpecialOperandAttr(MachineInstr &OldMI1,
                                         MachineInstr &OldMI2,
                                         MachineInstr &NewMI1,
                                         MachineInstr &NewMI2) const {
  // Integer instructions define an implicit EFLAGS source register operand as
  // the third source (fourth total) operand.
  if (OldMI1.getNumOperands() != 4 || OldMI2.getNumOperands() != 4)
    return;

  assert(NewMI1.getNumOperands() == 4 && NewMI2.getNumOperands() == 4 &&
         "Unexpected instruction type for reassociation");

  MachineOperand &OldOp1 = OldMI1.getOperand(3);
  MachineOperand &OldOp2 = OldMI2.getOperand(3);
  MachineOperand &NewOp1 = NewMI1.getOperand(3);
  MachineOperand &NewOp2 = NewMI2.getOperand(3);

  assert(OldOp1.isReg() && OldOp1.getReg() == X86::EFLAGS && OldOp1.isDead() &&
         "Must have dead EFLAGS operand in reassociable instruction");
  assert(OldOp2.isReg() && OldOp2.getReg() == X86::EFLAGS && OldOp2.isDead() &&
         "Must have dead EFLAGS operand in reassociable instruction");

  (void)OldOp1;
  (void)OldOp2;

  assert(NewOp1.isReg() && NewOp1.getReg() == X86::EFLAGS &&
         "Unexpected operand in reassociable instruction");
  assert(NewOp2.isReg() && NewOp2.getReg() == X86::EFLAGS &&
         "Unexpected operand in reassociable instruction");

  // Mark the new EFLAGS operands as dead to be helpful to subsequent iterations
  // of this pass or other passes. The EFLAGS operands must be dead in these new
  // instructions because the EFLAGS operands in the original instructions must
  // be dead in order for reassociation to occur.
  NewOp1.setIsDead();
  NewOp2.setIsDead();
}

std::pair<unsigned, unsigned>
X86InstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  return std::make_pair(TF, 0u);
}

ArrayRef<std::pair<unsigned, const char *>>
X86InstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace X86II;
  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_GOT_ABSOLUTE_ADDRESS, "x86-got-absolute-address"},
      {MO_PIC_BASE_OFFSET, "x86-pic-base-offset"},
      {MO_GOT, "x86-got"},
      {MO_GOTOFF, "x86-gotoff"},
      {MO_GOTPCREL, "x86-gotpcrel"},
      {MO_PLT, "x86-plt"},
      {MO_TLSGD, "x86-tlsgd"},
      {MO_TLSLD, "x86-tlsld"},
      {MO_TLSLDM, "x86-tlsldm"},
      {MO_GOTTPOFF, "x86-gottpoff"},
      {MO_INDNTPOFF, "x86-indntpoff"},
      {MO_TPOFF, "x86-tpoff"},
      {MO_DTPOFF, "x86-dtpoff"},
      {MO_NTPOFF, "x86-ntpoff"},
      {MO_GOTNTPOFF, "x86-gotntpoff"},
      {MO_DLLIMPORT, "x86-dllimport"},
      {MO_DARWIN_STUB, "x86-darwin-stub"},
      {MO_DARWIN_NONLAZY, "x86-darwin-nonlazy"},
      {MO_DARWIN_NONLAZY_PIC_BASE, "x86-darwin-nonlazy-pic-base"},
      {MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE, "x86-darwin-hidden-nonlazy-pic-base"},
      {MO_TLVP, "x86-tlvp"},
      {MO_TLVP_PIC_BASE, "x86-tlvp-pic-base"},
      {MO_SECREL, "x86-secrel"}};
  return makeArrayRef(TargetFlags);
}

namespace {
  /// Create Global Base Reg pass. This initializes the PIC
  /// global base register for x86-32.
  struct CGBR : public MachineFunctionPass {
    static char ID;
    CGBR() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override {
      const X86TargetMachine *TM =
        static_cast<const X86TargetMachine *>(&MF.getTarget());
      const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();

      // Don't do anything if this is 64-bit as 64-bit PIC
      // uses RIP relative addressing.
      if (STI.is64Bit())
        return false;

      // Only emit a global base reg in PIC mode.
      if (TM->getRelocationModel() != Reloc::PIC_)
        return false;

      X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
      unsigned GlobalBaseReg = X86FI->getGlobalBaseReg();

      // If we didn't need a GlobalBaseReg, don't insert code.
      if (GlobalBaseReg == 0)
        return false;

      // Insert the set of GlobalBaseReg into the first MBB of the function
      MachineBasicBlock &FirstMBB = MF.front();
      MachineBasicBlock::iterator MBBI = FirstMBB.begin();
      DebugLoc DL = FirstMBB.findDebugLoc(MBBI);
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      const X86InstrInfo *TII = STI.getInstrInfo();

      unsigned PC;
      if (STI.isPICStyleGOT())
        PC = RegInfo.createVirtualRegister(&X86::GR32RegClass);
      else
        PC = GlobalBaseReg;

      // Operand of MovePCtoStack is completely ignored by asm printer. It's
      // only used in JIT code emission as displacement to pc.
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::MOVPC32r), PC).addImm(0);

      // If we're using vanilla 'GOT' PIC style, we should use relative addressing
      // not to pc, but to _GLOBAL_OFFSET_TABLE_ external.
      if (STI.isPICStyleGOT()) {
        // Generate addl $__GLOBAL_OFFSET_TABLE_ + [.-piclabel], %some_register
        BuildMI(FirstMBB, MBBI, DL, TII->get(X86::ADD32ri), GlobalBaseReg)
          .addReg(PC).addExternalSymbol("_GLOBAL_OFFSET_TABLE_",
                                        X86II::MO_GOT_ABSOLUTE_ADDRESS);
      }

      return true;
    }

    const char *getPassName() const override {
      return "X86 PIC Global Base Reg Initialization";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char CGBR::ID = 0;
FunctionPass*
llvm::createX86GlobalBaseRegPass() { return new CGBR(); }

namespace {
  struct LDTLSCleanup : public MachineFunctionPass {
    static char ID;
    LDTLSCleanup() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override {
      X86MachineFunctionInfo* MFI = MF.getInfo<X86MachineFunctionInfo>();
      if (MFI->getNumLocalDynamicTLSAccesses() < 2) {
        // No point folding accesses if there isn't at least two.
        return false;
      }

      MachineDominatorTree *DT = &getAnalysis<MachineDominatorTree>();
      return VisitNode(DT->getRootNode(), 0);
    }

    // Visit the dominator subtree rooted at Node in pre-order.
    // If TLSBaseAddrReg is non-null, then use that to replace any
    // TLS_base_addr instructions. Otherwise, create the register
    // when the first such instruction is seen, and then use it
    // as we encounter more instructions.
    bool VisitNode(MachineDomTreeNode *Node, unsigned TLSBaseAddrReg) {
      MachineBasicBlock *BB = Node->getBlock();
      bool Changed = false;

      // Traverse the current block.
      for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;
           ++I) {
        switch (I->getOpcode()) {
          case X86::TLS_base_addr32:
          case X86::TLS_base_addr64:
            if (TLSBaseAddrReg)
              I = ReplaceTLSBaseAddrCall(I, TLSBaseAddrReg);
            else
              I = SetRegister(I, &TLSBaseAddrReg);
            Changed = true;
            break;
          default:
            break;
        }
      }

      // Visit the children of this block in the dominator tree.
      for (MachineDomTreeNode::iterator I = Node->begin(), E = Node->end();
           I != E; ++I) {
        Changed |= VisitNode(*I, TLSBaseAddrReg);
      }

      return Changed;
    }

    // Replace the TLS_base_addr instruction I with a copy from
    // TLSBaseAddrReg, returning the new instruction.
    MachineInstr *ReplaceTLSBaseAddrCall(MachineInstr *I,
                                         unsigned TLSBaseAddrReg) {
      MachineFunction *MF = I->getParent()->getParent();
      const X86Subtarget &STI = MF->getSubtarget<X86Subtarget>();
      const bool is64Bit = STI.is64Bit();
      const X86InstrInfo *TII = STI.getInstrInfo();

      // Insert a Copy from TLSBaseAddrReg to RAX/EAX.
      MachineInstr *Copy = BuildMI(*I->getParent(), I, I->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY),
                                   is64Bit ? X86::RAX : X86::EAX)
                                   .addReg(TLSBaseAddrReg);

      // Erase the TLS_base_addr instruction.
      I->eraseFromParent();

      return Copy;
    }

    // Create a virtal register in *TLSBaseAddrReg, and populate it by
    // inserting a copy instruction after I. Returns the new instruction.
    MachineInstr *SetRegister(MachineInstr *I, unsigned *TLSBaseAddrReg) {
      MachineFunction *MF = I->getParent()->getParent();
      const X86Subtarget &STI = MF->getSubtarget<X86Subtarget>();
      const bool is64Bit = STI.is64Bit();
      const X86InstrInfo *TII = STI.getInstrInfo();

      // Create a virtual register for the TLS base address.
      MachineRegisterInfo &RegInfo = MF->getRegInfo();
      *TLSBaseAddrReg = RegInfo.createVirtualRegister(is64Bit
                                                      ? &X86::GR64RegClass
                                                      : &X86::GR32RegClass);

      // Insert a copy from RAX/EAX to TLSBaseAddrReg.
      MachineInstr *Next = I->getNextNode();
      MachineInstr *Copy = BuildMI(*I->getParent(), Next, I->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY),
                                   *TLSBaseAddrReg)
                                   .addReg(is64Bit ? X86::RAX : X86::EAX);

      return Copy;
    }

    const char *getPassName() const override {
      return "Local Dynamic TLS Access Clean-up";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char LDTLSCleanup::ID = 0;
FunctionPass*
llvm::createCleanupLocalDynamicTLSPass() { return new LDTLSCleanup(); }
