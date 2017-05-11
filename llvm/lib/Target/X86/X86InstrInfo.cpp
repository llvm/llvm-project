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
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
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
static cl::opt<unsigned>
PartialRegUpdateClearance("partial-reg-update-clearance",
                          cl::desc("Clearance between two register writes "
                                   "for inserting XOR to avoid partial "
                                   "register update"),
                          cl::init(64), cl::Hidden);
static cl::opt<unsigned>
UndefRegClearance("undef-reg-clearance",
                  cl::desc("How many idle instructions we would like before "
                           "certain undef register reads"),
                  cl::init(128), cl::Hidden);

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
                      X86::CATCHRET,
                      (STI.is64Bit() ? X86::RETQ : X86::RETL)),
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
    { X86::SHL16r1,     X86::SHL16m1,    0 },
    { X86::SHL16rCL,    X86::SHL16mCL,   0 },
    { X86::SHL16ri,     X86::SHL16mi,    0 },
    { X86::SHL32r1,     X86::SHL32m1,    0 },
    { X86::SHL32rCL,    X86::SHL32mCL,   0 },
    { X86::SHL32ri,     X86::SHL32mi,    0 },
    { X86::SHL64r1,     X86::SHL64m1,    0 },
    { X86::SHL64rCL,    X86::SHL64mCL,   0 },
    { X86::SHL64ri,     X86::SHL64mi,    0 },
    { X86::SHL8r1,      X86::SHL8m1,     0 },
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
    { X86::MOVDQUrr,    X86::MOVDQUmr,      TB_FOLDED_STORE },
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
    { X86::VMOVDQUrr,   X86::VMOVDQUmr,     TB_FOLDED_STORE },
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
    { X86::VMOVDQUYrr,  X86::VMOVDQUYmr,    TB_FOLDED_STORE },
    { X86::VMOVUPDYrr,  X86::VMOVUPDYmr,    TB_FOLDED_STORE },
    { X86::VMOVUPSYrr,  X86::VMOVUPSYmr,    TB_FOLDED_STORE },

    // AVX-512 foldable instructions
    { X86::VEXTRACTF32x4Zrr,X86::VEXTRACTF32x4Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTF32x8Zrr,X86::VEXTRACTF32x8Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTF64x2Zrr,X86::VEXTRACTF64x2Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTF64x4Zrr,X86::VEXTRACTF64x4Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTI32x4Zrr,X86::VEXTRACTI32x4Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTI32x8Zrr,X86::VEXTRACTI32x8Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTI64x2Zrr,X86::VEXTRACTI64x2Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTI64x4Zrr,X86::VEXTRACTI64x4Zmr, TB_FOLDED_STORE },
    { X86::VEXTRACTPSZrr,   X86::VEXTRACTPSZmr,    TB_FOLDED_STORE },
    { X86::VMOVAPDZrr,      X86::VMOVAPDZmr,    TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVAPSZrr,      X86::VMOVAPSZmr,    TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVDQA32Zrr,    X86::VMOVDQA32Zmr,  TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVDQA64Zrr,    X86::VMOVDQA64Zmr,  TB_FOLDED_STORE | TB_ALIGN_64 },
    { X86::VMOVDQU8Zrr,     X86::VMOVDQU8Zmr,   TB_FOLDED_STORE },
    { X86::VMOVDQU16Zrr,    X86::VMOVDQU16Zmr,  TB_FOLDED_STORE },
    { X86::VMOVDQU32Zrr,    X86::VMOVDQU32Zmr,  TB_FOLDED_STORE },
    { X86::VMOVDQU64Zrr,    X86::VMOVDQU64Zmr,  TB_FOLDED_STORE },
    { X86::VMOVPDI2DIZrr,   X86::VMOVPDI2DIZmr, TB_FOLDED_STORE },
    { X86::VMOVPQIto64Zrr,  X86::VMOVPQI2QIZmr, TB_FOLDED_STORE },
    { X86::VMOVSDto64Zrr,   X86::VMOVSDto64Zmr, TB_FOLDED_STORE },
    { X86::VMOVSS2DIZrr,    X86::VMOVSS2DIZmr,  TB_FOLDED_STORE },
    { X86::VMOVUPDZrr,      X86::VMOVUPDZmr,    TB_FOLDED_STORE },
    { X86::VMOVUPSZrr,      X86::VMOVUPSZmr,    TB_FOLDED_STORE },
    { X86::VPEXTRDZrr,      X86::VPEXTRDZmr,    TB_FOLDED_STORE },
    { X86::VPEXTRQZrr,      X86::VPEXTRQZmr,    TB_FOLDED_STORE },
    { X86::VPMOVDBZrr,      X86::VPMOVDBZmr,    TB_FOLDED_STORE },
    { X86::VPMOVDWZrr,      X86::VPMOVDWZmr,    TB_FOLDED_STORE },
    { X86::VPMOVQDZrr,      X86::VPMOVQDZmr,    TB_FOLDED_STORE },
    { X86::VPMOVQWZrr,      X86::VPMOVQWZmr,    TB_FOLDED_STORE },
    { X86::VPMOVWBZrr,      X86::VPMOVWBZmr,    TB_FOLDED_STORE },
    { X86::VPMOVSDBZrr,     X86::VPMOVSDBZmr,   TB_FOLDED_STORE },
    { X86::VPMOVSDWZrr,     X86::VPMOVSDWZmr,   TB_FOLDED_STORE },
    { X86::VPMOVSQDZrr,     X86::VPMOVSQDZmr,   TB_FOLDED_STORE },
    { X86::VPMOVSQWZrr,     X86::VPMOVSQWZmr,   TB_FOLDED_STORE },
    { X86::VPMOVSWBZrr,     X86::VPMOVSWBZmr,   TB_FOLDED_STORE },
    { X86::VPMOVUSDBZrr,    X86::VPMOVUSDBZmr,  TB_FOLDED_STORE },
    { X86::VPMOVUSDWZrr,    X86::VPMOVUSDWZmr,  TB_FOLDED_STORE },
    { X86::VPMOVUSQDZrr,    X86::VPMOVUSQDZmr,  TB_FOLDED_STORE },
    { X86::VPMOVUSQWZrr,    X86::VPMOVUSQWZmr,  TB_FOLDED_STORE },
    { X86::VPMOVUSWBZrr,    X86::VPMOVUSWBZmr,  TB_FOLDED_STORE },

    // AVX-512 foldable instructions (256-bit versions)
    { X86::VEXTRACTF32x4Z256rr,X86::VEXTRACTF32x4Z256mr, TB_FOLDED_STORE },
    { X86::VEXTRACTF64x2Z256rr,X86::VEXTRACTF64x2Z256mr, TB_FOLDED_STORE },
    { X86::VEXTRACTI32x4Z256rr,X86::VEXTRACTI32x4Z256mr, TB_FOLDED_STORE },
    { X86::VEXTRACTI64x2Z256rr,X86::VEXTRACTI64x2Z256mr, TB_FOLDED_STORE },
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
    { X86::VPMOVDWZ256rr,      X86::VPMOVDWZ256mr,    TB_FOLDED_STORE },
    { X86::VPMOVQDZ256rr,      X86::VPMOVQDZ256mr,    TB_FOLDED_STORE },
    { X86::VPMOVWBZ256rr,      X86::VPMOVWBZ256mr,    TB_FOLDED_STORE },
    { X86::VPMOVSDWZ256rr,     X86::VPMOVSDWZ256mr,   TB_FOLDED_STORE },
    { X86::VPMOVSQDZ256rr,     X86::VPMOVSQDZ256mr,   TB_FOLDED_STORE },
    { X86::VPMOVSWBZ256rr,     X86::VPMOVSWBZ256mr,   TB_FOLDED_STORE },
    { X86::VPMOVUSDWZ256rr,    X86::VPMOVUSDWZ256mr,  TB_FOLDED_STORE },
    { X86::VPMOVUSQDZ256rr,    X86::VPMOVUSQDZ256mr,  TB_FOLDED_STORE },
    { X86::VPMOVUSWBZ256rr,    X86::VPMOVUSWBZ256mr,  TB_FOLDED_STORE },

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
    { X86::Int_COMISDrr,    X86::Int_COMISDrm,        TB_NO_REVERSE },
    { X86::Int_COMISSrr,    X86::Int_COMISSrm,        TB_NO_REVERSE },
    { X86::CVTSD2SI64rr,    X86::CVTSD2SI64rm,        TB_NO_REVERSE },
    { X86::CVTSD2SIrr,      X86::CVTSD2SIrm,          TB_NO_REVERSE },
    { X86::CVTSS2SI64rr,    X86::CVTSS2SI64rm,        TB_NO_REVERSE },
    { X86::CVTSS2SIrr,      X86::CVTSS2SIrm,          TB_NO_REVERSE },
    { X86::CVTDQ2PDrr,      X86::CVTDQ2PDrm,          TB_NO_REVERSE },
    { X86::CVTDQ2PSrr,      X86::CVTDQ2PSrm,          TB_ALIGN_16 },
    { X86::CVTPD2DQrr,      X86::CVTPD2DQrm,          TB_ALIGN_16 },
    { X86::CVTPD2PSrr,      X86::CVTPD2PSrm,          TB_ALIGN_16 },
    { X86::CVTPS2DQrr,      X86::CVTPS2DQrm,          TB_ALIGN_16 },
    { X86::CVTPS2PDrr,      X86::CVTPS2PDrm,          TB_NO_REVERSE },
    { X86::CVTTPD2DQrr,     X86::CVTTPD2DQrm,         TB_ALIGN_16 },
    { X86::CVTTPS2DQrr,     X86::CVTTPS2DQrm,         TB_ALIGN_16 },
    { X86::Int_CVTTSD2SI64rr,X86::Int_CVTTSD2SI64rm,  TB_NO_REVERSE },
    { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm,     TB_NO_REVERSE },
    { X86::Int_CVTTSS2SI64rr,X86::Int_CVTTSS2SI64rm,  TB_NO_REVERSE },
    { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm,     TB_NO_REVERSE },
    { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm,       TB_NO_REVERSE },
    { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm,       TB_NO_REVERSE },
    { X86::MOV16rr,         X86::MOV16rm,             0 },
    { X86::MOV32rr,         X86::MOV32rm,             0 },
    { X86::MOV64rr,         X86::MOV64rm,             0 },
    { X86::MOV64toPQIrr,    X86::MOVQI2PQIrm,         0 },
    { X86::MOV64toSDrr,     X86::MOV64toSDrm,         0 },
    { X86::MOV8rr,          X86::MOV8rm,              0 },
    { X86::MOVAPDrr,        X86::MOVAPDrm,            TB_ALIGN_16 },
    { X86::MOVAPSrr,        X86::MOVAPSrm,            TB_ALIGN_16 },
    { X86::MOVDDUPrr,       X86::MOVDDUPrm,           TB_NO_REVERSE },
    { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm,         0 },
    { X86::MOVDI2SSrr,      X86::MOVDI2SSrm,          0 },
    { X86::MOVDQArr,        X86::MOVDQArm,            TB_ALIGN_16 },
    { X86::MOVDQUrr,        X86::MOVDQUrm,            0 },
    { X86::MOVSHDUPrr,      X86::MOVSHDUPrm,          TB_ALIGN_16 },
    { X86::MOVSLDUPrr,      X86::MOVSLDUPrm,          TB_ALIGN_16 },
    { X86::MOVSX16rr8,      X86::MOVSX16rm8,          0 },
    { X86::MOVSX32rr16,     X86::MOVSX32rm16,         0 },
    { X86::MOVSX32rr8,      X86::MOVSX32rm8,          0 },
    { X86::MOVSX64rr16,     X86::MOVSX64rm16,         0 },
    { X86::MOVSX64rr32,     X86::MOVSX64rm32,         0 },
    { X86::MOVSX64rr8,      X86::MOVSX64rm8,          0 },
    { X86::MOVUPDrr,        X86::MOVUPDrm,            0 },
    { X86::MOVUPSrr,        X86::MOVUPSrm,            0 },
    { X86::MOVZPQILo2PQIrr, X86::MOVQI2PQIrm,         TB_NO_REVERSE },
    { X86::MOVZX16rr8,      X86::MOVZX16rm8,          0 },
    { X86::MOVZX32rr16,     X86::MOVZX32rm16,         0 },
    { X86::MOVZX32_NOREXrr8, X86::MOVZX32_NOREXrm8,   0 },
    { X86::MOVZX32rr8,      X86::MOVZX32rm8,          0 },
    { X86::PABSBrr,         X86::PABSBrm,             TB_ALIGN_16 },
    { X86::PABSDrr,         X86::PABSDrm,             TB_ALIGN_16 },
    { X86::PABSWrr,         X86::PABSWrm,             TB_ALIGN_16 },
    { X86::PCMPESTRIrr,     X86::PCMPESTRIrm,         TB_ALIGN_16 },
    { X86::PCMPESTRM128rr,  X86::PCMPESTRM128rm,      TB_ALIGN_16 },
    { X86::PCMPISTRIrr,     X86::PCMPISTRIrm,         TB_ALIGN_16 },
    { X86::PCMPISTRM128rr,  X86::PCMPISTRM128rm,      TB_ALIGN_16 },
    { X86::PHMINPOSUWrr128, X86::PHMINPOSUWrm128,     TB_ALIGN_16 },
    { X86::PMOVSXBDrr,      X86::PMOVSXBDrm,          TB_NO_REVERSE },
    { X86::PMOVSXBQrr,      X86::PMOVSXBQrm,          TB_NO_REVERSE },
    { X86::PMOVSXBWrr,      X86::PMOVSXBWrm,          TB_NO_REVERSE },
    { X86::PMOVSXDQrr,      X86::PMOVSXDQrm,          TB_NO_REVERSE },
    { X86::PMOVSXWDrr,      X86::PMOVSXWDrm,          TB_NO_REVERSE },
    { X86::PMOVSXWQrr,      X86::PMOVSXWQrm,          TB_NO_REVERSE },
    { X86::PMOVZXBDrr,      X86::PMOVZXBDrm,          TB_NO_REVERSE },
    { X86::PMOVZXBQrr,      X86::PMOVZXBQrm,          TB_NO_REVERSE },
    { X86::PMOVZXBWrr,      X86::PMOVZXBWrm,          TB_NO_REVERSE },
    { X86::PMOVZXDQrr,      X86::PMOVZXDQrm,          TB_NO_REVERSE },
    { X86::PMOVZXWDrr,      X86::PMOVZXWDrm,          TB_NO_REVERSE },
    { X86::PMOVZXWQrr,      X86::PMOVZXWQrm,          TB_NO_REVERSE },
    { X86::PSHUFDri,        X86::PSHUFDmi,            TB_ALIGN_16 },
    { X86::PSHUFHWri,       X86::PSHUFHWmi,           TB_ALIGN_16 },
    { X86::PSHUFLWri,       X86::PSHUFLWmi,           TB_ALIGN_16 },
    { X86::PTESTrr,         X86::PTESTrm,             TB_ALIGN_16 },
    { X86::RCPPSr,          X86::RCPPSm,              TB_ALIGN_16 },
    { X86::RCPSSr,          X86::RCPSSm,              0 },
    { X86::RCPSSr_Int,      X86::RCPSSm_Int,          TB_NO_REVERSE },
    { X86::ROUNDPDr,        X86::ROUNDPDm,            TB_ALIGN_16 },
    { X86::ROUNDPSr,        X86::ROUNDPSm,            TB_ALIGN_16 },
    { X86::ROUNDSDr,        X86::ROUNDSDm,            0 },
    { X86::ROUNDSSr,        X86::ROUNDSSm,            0 },
    { X86::RSQRTPSr,        X86::RSQRTPSm,            TB_ALIGN_16 },
    { X86::RSQRTSSr,        X86::RSQRTSSm,            0 },
    { X86::RSQRTSSr_Int,    X86::RSQRTSSm_Int,        TB_NO_REVERSE },
    { X86::SQRTPDr,         X86::SQRTPDm,             TB_ALIGN_16 },
    { X86::SQRTPSr,         X86::SQRTPSm,             TB_ALIGN_16 },
    { X86::SQRTSDr,         X86::SQRTSDm,             0 },
    { X86::SQRTSDr_Int,     X86::SQRTSDm_Int,         TB_NO_REVERSE },
    { X86::SQRTSSr,         X86::SQRTSSm,             0 },
    { X86::SQRTSSr_Int,     X86::SQRTSSm_Int,         TB_NO_REVERSE },
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
    { X86::Int_VCOMISDrr,   X86::Int_VCOMISDrm,       TB_NO_REVERSE },
    { X86::Int_VCOMISSrr,   X86::Int_VCOMISSrm,       TB_NO_REVERSE },
    { X86::Int_VUCOMISDrr,  X86::Int_VUCOMISDrm,      TB_NO_REVERSE },
    { X86::Int_VUCOMISSrr,  X86::Int_VUCOMISSrm,      TB_NO_REVERSE },
    { X86::VCVTTSD2SI64rr,  X86::VCVTTSD2SI64rm,      0 },
    { X86::Int_VCVTTSD2SI64rr,X86::Int_VCVTTSD2SI64rm,TB_NO_REVERSE },
    { X86::VCVTTSD2SIrr,    X86::VCVTTSD2SIrm,        0 },
    { X86::Int_VCVTTSD2SIrr,X86::Int_VCVTTSD2SIrm,    TB_NO_REVERSE },
    { X86::VCVTTSS2SI64rr,  X86::VCVTTSS2SI64rm,      0 },
    { X86::Int_VCVTTSS2SI64rr,X86::Int_VCVTTSS2SI64rm,TB_NO_REVERSE },
    { X86::VCVTTSS2SIrr,    X86::VCVTTSS2SIrm,        0 },
    { X86::Int_VCVTTSS2SIrr,X86::Int_VCVTTSS2SIrm,    TB_NO_REVERSE },
    { X86::VCVTSD2SI64rr,   X86::VCVTSD2SI64rm,       TB_NO_REVERSE },
    { X86::VCVTSD2SIrr,     X86::VCVTSD2SIrm,         TB_NO_REVERSE },
    { X86::VCVTSS2SI64rr,   X86::VCVTSS2SI64rm,       TB_NO_REVERSE },
    { X86::VCVTSS2SIrr,     X86::VCVTSS2SIrm,         TB_NO_REVERSE },
    { X86::VCVTDQ2PDrr,     X86::VCVTDQ2PDrm,         TB_NO_REVERSE },
    { X86::VCVTDQ2PSrr,     X86::VCVTDQ2PSrm,         0 },
    { X86::VCVTPD2DQrr,     X86::VCVTPD2DQrm,         0 },
    { X86::VCVTPD2PSrr,     X86::VCVTPD2PSrm,         0 },
    { X86::VCVTPS2DQrr,     X86::VCVTPS2DQrm,         0 },
    { X86::VCVTPS2PDrr,     X86::VCVTPS2PDrm,         TB_NO_REVERSE },
    { X86::VCVTTPD2DQrr,    X86::VCVTTPD2DQrm,        0 },
    { X86::VCVTTPS2DQrr,    X86::VCVTTPS2DQrm,        0 },
    { X86::VMOV64toPQIrr,   X86::VMOVQI2PQIrm,        0 },
    { X86::VMOV64toSDrr,    X86::VMOV64toSDrm,        0 },
    { X86::VMOVAPDrr,       X86::VMOVAPDrm,           TB_ALIGN_16 },
    { X86::VMOVAPSrr,       X86::VMOVAPSrm,           TB_ALIGN_16 },
    { X86::VMOVDDUPrr,      X86::VMOVDDUPrm,          TB_NO_REVERSE },
    { X86::VMOVDI2PDIrr,    X86::VMOVDI2PDIrm,        0 },
    { X86::VMOVDI2SSrr,     X86::VMOVDI2SSrm,         0 },
    { X86::VMOVDQArr,       X86::VMOVDQArm,           TB_ALIGN_16 },
    { X86::VMOVDQUrr,       X86::VMOVDQUrm,           0 },
    { X86::VMOVSLDUPrr,     X86::VMOVSLDUPrm,         0 },
    { X86::VMOVSHDUPrr,     X86::VMOVSHDUPrm,         0 },
    { X86::VMOVUPDrr,       X86::VMOVUPDrm,           0 },
    { X86::VMOVUPSrr,       X86::VMOVUPSrm,           0 },
    { X86::VMOVZPQILo2PQIrr,X86::VMOVQI2PQIrm,        TB_NO_REVERSE },
    { X86::VPABSBrr,        X86::VPABSBrm,            0 },
    { X86::VPABSDrr,        X86::VPABSDrm,            0 },
    { X86::VPABSWrr,        X86::VPABSWrm,            0 },
    { X86::VPCMPESTRIrr,    X86::VPCMPESTRIrm,        0 },
    { X86::VPCMPESTRM128rr, X86::VPCMPESTRM128rm,     0 },
    { X86::VPCMPISTRIrr,    X86::VPCMPISTRIrm,        0 },
    { X86::VPCMPISTRM128rr, X86::VPCMPISTRM128rm,     0 },
    { X86::VPHMINPOSUWrr128, X86::VPHMINPOSUWrm128,   0 },
    { X86::VPERMILPDri,     X86::VPERMILPDmi,         0 },
    { X86::VPERMILPSri,     X86::VPERMILPSmi,         0 },
    { X86::VPMOVSXBDrr,     X86::VPMOVSXBDrm,         TB_NO_REVERSE },
    { X86::VPMOVSXBQrr,     X86::VPMOVSXBQrm,         TB_NO_REVERSE },
    { X86::VPMOVSXBWrr,     X86::VPMOVSXBWrm,         TB_NO_REVERSE },
    { X86::VPMOVSXDQrr,     X86::VPMOVSXDQrm,         TB_NO_REVERSE },
    { X86::VPMOVSXWDrr,     X86::VPMOVSXWDrm,         TB_NO_REVERSE },
    { X86::VPMOVSXWQrr,     X86::VPMOVSXWQrm,         TB_NO_REVERSE },
    { X86::VPMOVZXBDrr,     X86::VPMOVZXBDrm,         TB_NO_REVERSE },
    { X86::VPMOVZXBQrr,     X86::VPMOVZXBQrm,         TB_NO_REVERSE },
    { X86::VPMOVZXBWrr,     X86::VPMOVZXBWrm,         TB_NO_REVERSE },
    { X86::VPMOVZXDQrr,     X86::VPMOVZXDQrm,         TB_NO_REVERSE },
    { X86::VPMOVZXWDrr,     X86::VPMOVZXWDrm,         TB_NO_REVERSE },
    { X86::VPMOVZXWQrr,     X86::VPMOVZXWQrm,         TB_NO_REVERSE },
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
    { X86::VCVTDQ2PDYrr,    X86::VCVTDQ2PDYrm,        TB_NO_REVERSE },
    { X86::VCVTDQ2PSYrr,    X86::VCVTDQ2PSYrm,        0 },
    { X86::VCVTPD2DQYrr,    X86::VCVTPD2DQYrm,        0 },
    { X86::VCVTPD2PSYrr,    X86::VCVTPD2PSYrm,        0 },
    { X86::VCVTPS2DQYrr,    X86::VCVTPS2DQYrm,        0 },
    { X86::VCVTPS2PDYrr,    X86::VCVTPS2PDYrm,        TB_NO_REVERSE },
    { X86::VCVTTPD2DQYrr,   X86::VCVTTPD2DQYrm,       0 },
    { X86::VCVTTPS2DQYrr,   X86::VCVTTPS2DQYrm,       0 },
    { X86::VMOVAPDYrr,      X86::VMOVAPDYrm,          TB_ALIGN_32 },
    { X86::VMOVAPSYrr,      X86::VMOVAPSYrm,          TB_ALIGN_32 },
    { X86::VMOVDDUPYrr,     X86::VMOVDDUPYrm,         0 },
    { X86::VMOVDQAYrr,      X86::VMOVDQAYrm,          TB_ALIGN_32 },
    { X86::VMOVDQUYrr,      X86::VMOVDQUYrm,          0 },
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
    { X86::VPABSBYrr,       X86::VPABSBYrm,           0 },
    { X86::VPABSDYrr,       X86::VPABSDYrm,           0 },
    { X86::VPABSWYrr,       X86::VPABSWYrm,           0 },
    { X86::VPBROADCASTBrr,  X86::VPBROADCASTBrm,      TB_NO_REVERSE },
    { X86::VPBROADCASTBYrr, X86::VPBROADCASTBYrm,     TB_NO_REVERSE },
    { X86::VPBROADCASTDrr,  X86::VPBROADCASTDrm,      TB_NO_REVERSE },
    { X86::VPBROADCASTDYrr, X86::VPBROADCASTDYrm,     TB_NO_REVERSE },
    { X86::VPBROADCASTQrr,  X86::VPBROADCASTQrm,      TB_NO_REVERSE },
    { X86::VPBROADCASTQYrr, X86::VPBROADCASTQYrm,     TB_NO_REVERSE },
    { X86::VPBROADCASTWrr,  X86::VPBROADCASTWrm,      TB_NO_REVERSE },
    { X86::VPBROADCASTWYrr, X86::VPBROADCASTWYrm,     TB_NO_REVERSE },
    { X86::VPERMPDYri,      X86::VPERMPDYmi,          0 },
    { X86::VPERMQYri,       X86::VPERMQYmi,           0 },
    { X86::VPMOVSXBDYrr,    X86::VPMOVSXBDYrm,        TB_NO_REVERSE },
    { X86::VPMOVSXBQYrr,    X86::VPMOVSXBQYrm,        TB_NO_REVERSE },
    { X86::VPMOVSXBWYrr,    X86::VPMOVSXBWYrm,        0 },
    { X86::VPMOVSXDQYrr,    X86::VPMOVSXDQYrm,        0 },
    { X86::VPMOVSXWDYrr,    X86::VPMOVSXWDYrm,        0 },
    { X86::VPMOVSXWQYrr,    X86::VPMOVSXWQYrm,        TB_NO_REVERSE },
    { X86::VPMOVZXBDYrr,    X86::VPMOVZXBDYrm,        TB_NO_REVERSE },
    { X86::VPMOVZXBQYrr,    X86::VPMOVZXBQYrm,        TB_NO_REVERSE },
    { X86::VPMOVZXBWYrr,    X86::VPMOVZXBWYrm,        0 },
    { X86::VPMOVZXDQYrr,    X86::VPMOVZXDQYrm,        0 },
    { X86::VPMOVZXWDYrr,    X86::VPMOVZXWDYrm,        0 },
    { X86::VPMOVZXWQYrr,    X86::VPMOVZXWQYrm,        TB_NO_REVERSE },
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

    // LWP foldable instructions
    { X86::LWPINS32rri,        X86::LWPINS32rmi,      0 },
    { X86::LWPINS64rri,        X86::LWPINS64rmi,      0 },
    { X86::LWPVAL32rri,        X86::LWPVAL32rmi,      0 },
    { X86::LWPVAL64rri,        X86::LWPVAL64rmi,      0 },

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
    { X86::VBROADCASTSSZr,   X86::VBROADCASTSSZm,     TB_NO_REVERSE },
    { X86::VBROADCASTSDZr,   X86::VBROADCASTSDZm,     TB_NO_REVERSE },
    { X86::VMOV64toPQIZrr,   X86::VMOVQI2PQIZrm,      0 },
    { X86::VMOV64toSDZrr,    X86::VMOV64toSDZrm,      0 },
    { X86::VMOVDI2PDIZrr,    X86::VMOVDI2PDIZrm,      0 },
    { X86::VMOVDI2SSZrr,     X86::VMOVDI2SSZrm,       0 },
    { X86::VMOVAPDZrr,       X86::VMOVAPDZrm,         TB_ALIGN_64 },
    { X86::VMOVAPSZrr,       X86::VMOVAPSZrm,         TB_ALIGN_64 },
    { X86::VMOVDQA32Zrr,     X86::VMOVDQA32Zrm,       TB_ALIGN_64 },
    { X86::VMOVDQA64Zrr,     X86::VMOVDQA64Zrm,       TB_ALIGN_64 },
    { X86::VMOVDQU8Zrr,      X86::VMOVDQU8Zrm,        0 },
    { X86::VMOVDQU16Zrr,     X86::VMOVDQU16Zrm,       0 },
    { X86::VMOVDQU32Zrr,     X86::VMOVDQU32Zrm,       0 },
    { X86::VMOVDQU64Zrr,     X86::VMOVDQU64Zrm,       0 },
    { X86::VMOVUPDZrr,       X86::VMOVUPDZrm,         0 },
    { X86::VMOVUPSZrr,       X86::VMOVUPSZrm,         0 },
    { X86::VMOVZPQILo2PQIZrr,X86::VMOVQI2PQIZrm,      TB_NO_REVERSE },
    { X86::VPABSBZrr,        X86::VPABSBZrm,          0 },
    { X86::VPABSDZrr,        X86::VPABSDZrm,          0 },
    { X86::VPABSQZrr,        X86::VPABSQZrm,          0 },
    { X86::VPABSWZrr,        X86::VPABSWZrm,          0 },
    { X86::VPERMILPDZri,     X86::VPERMILPDZmi,       0 },
    { X86::VPERMILPSZri,     X86::VPERMILPSZmi,       0 },
    { X86::VPERMPDZri,       X86::VPERMPDZmi,         0 },
    { X86::VPERMQZri,        X86::VPERMQZmi,          0 },
    { X86::VPMOVSXBDZrr,     X86::VPMOVSXBDZrm,       0 },
    { X86::VPMOVSXBQZrr,     X86::VPMOVSXBQZrm,       TB_NO_REVERSE },
    { X86::VPMOVSXBWZrr,     X86::VPMOVSXBWZrm,       0 },
    { X86::VPMOVSXDQZrr,     X86::VPMOVSXDQZrm,       0 },
    { X86::VPMOVSXWDZrr,     X86::VPMOVSXWDZrm,       0 },
    { X86::VPMOVSXWQZrr,     X86::VPMOVSXWQZrm,       0 },
    { X86::VPMOVZXBDZrr,     X86::VPMOVZXBDZrm,       0 },
    { X86::VPMOVZXBQZrr,     X86::VPMOVZXBQZrm,       TB_NO_REVERSE },
    { X86::VPMOVZXBWZrr,     X86::VPMOVZXBWZrm,       0 },
    { X86::VPMOVZXDQZrr,     X86::VPMOVZXDQZrm,       0 },
    { X86::VPMOVZXWDZrr,     X86::VPMOVZXWDZrm,       0 },
    { X86::VPMOVZXWQZrr,     X86::VPMOVZXWQZrm,       0 },
    { X86::VPSHUFDZri,       X86::VPSHUFDZmi,         0 },
    { X86::VPSHUFHWZri,      X86::VPSHUFHWZmi,        0 },
    { X86::VPSHUFLWZri,      X86::VPSHUFLWZmi,        0 },
    { X86::VPSLLDQZ512rr,    X86::VPSLLDQZ512rm,      0 },
    { X86::VPSLLDZri,        X86::VPSLLDZmi,          0 },
    { X86::VPSLLQZri,        X86::VPSLLQZmi,          0 },
    { X86::VPSLLWZri,        X86::VPSLLWZmi,          0 },
    { X86::VPSRADZri,        X86::VPSRADZmi,          0 },
    { X86::VPSRAQZri,        X86::VPSRAQZmi,          0 },
    { X86::VPSRAWZri,        X86::VPSRAWZmi,          0 },
    { X86::VPSRLDQZ512rr,    X86::VPSRLDQZ512rm,      0 },
    { X86::VPSRLDZri,        X86::VPSRLDZmi,          0 },
    { X86::VPSRLQZri,        X86::VPSRLQZmi,          0 },
    { X86::VPSRLWZri,        X86::VPSRLWZmi,          0 },

    // AVX-512 foldable instructions (256-bit versions)
    { X86::VBROADCASTSSZ256r,    X86::VBROADCASTSSZ256m,    TB_NO_REVERSE },
    { X86::VBROADCASTSDZ256r,    X86::VBROADCASTSDZ256m,    TB_NO_REVERSE },
    { X86::VMOVAPDZ256rr,        X86::VMOVAPDZ256rm,        TB_ALIGN_32 },
    { X86::VMOVAPSZ256rr,        X86::VMOVAPSZ256rm,        TB_ALIGN_32 },
    { X86::VMOVDQA32Z256rr,      X86::VMOVDQA32Z256rm,      TB_ALIGN_32 },
    { X86::VMOVDQA64Z256rr,      X86::VMOVDQA64Z256rm,      TB_ALIGN_32 },
    { X86::VMOVDQU8Z256rr,       X86::VMOVDQU8Z256rm,       0 },
    { X86::VMOVDQU16Z256rr,      X86::VMOVDQU16Z256rm,      0 },
    { X86::VMOVDQU32Z256rr,      X86::VMOVDQU32Z256rm,      0 },
    { X86::VMOVDQU64Z256rr,      X86::VMOVDQU64Z256rm,      0 },
    { X86::VMOVUPDZ256rr,        X86::VMOVUPDZ256rm,        0 },
    { X86::VMOVUPSZ256rr,        X86::VMOVUPSZ256rm,        0 },
    { X86::VPABSBZ256rr,         X86::VPABSBZ256rm,         0 },
    { X86::VPABSDZ256rr,         X86::VPABSDZ256rm,         0 },
    { X86::VPABSQZ256rr,         X86::VPABSQZ256rm,         0 },
    { X86::VPABSWZ256rr,         X86::VPABSWZ256rm,         0 },
    { X86::VPERMILPDZ256ri,      X86::VPERMILPDZ256mi,      0 },
    { X86::VPERMILPSZ256ri,      X86::VPERMILPSZ256mi,      0 },
    { X86::VPERMPDZ256ri,        X86::VPERMPDZ256mi,        0 },
    { X86::VPERMQZ256ri,         X86::VPERMQZ256mi,         0 },
    { X86::VPMOVSXBDZ256rr,      X86::VPMOVSXBDZ256rm,      TB_NO_REVERSE },
    { X86::VPMOVSXBQZ256rr,      X86::VPMOVSXBQZ256rm,      TB_NO_REVERSE },
    { X86::VPMOVSXBWZ256rr,      X86::VPMOVSXBWZ256rm,      0 },
    { X86::VPMOVSXDQZ256rr,      X86::VPMOVSXDQZ256rm,      0 },
    { X86::VPMOVSXWDZ256rr,      X86::VPMOVSXWDZ256rm,      0 },
    { X86::VPMOVSXWQZ256rr,      X86::VPMOVSXWQZ256rm,      TB_NO_REVERSE },
    { X86::VPMOVZXBDZ256rr,      X86::VPMOVZXBDZ256rm,      TB_NO_REVERSE },
    { X86::VPMOVZXBQZ256rr,      X86::VPMOVZXBQZ256rm,      TB_NO_REVERSE },
    { X86::VPMOVZXBWZ256rr,      X86::VPMOVZXBWZ256rm,      0 },
    { X86::VPMOVZXDQZ256rr,      X86::VPMOVZXDQZ256rm,      0 },
    { X86::VPMOVZXWDZ256rr,      X86::VPMOVZXWDZ256rm,      0 },
    { X86::VPMOVZXWQZ256rr,      X86::VPMOVZXWQZ256rm,      TB_NO_REVERSE },
    { X86::VPSHUFDZ256ri,        X86::VPSHUFDZ256mi,        0 },
    { X86::VPSHUFHWZ256ri,       X86::VPSHUFHWZ256mi,       0 },
    { X86::VPSHUFLWZ256ri,       X86::VPSHUFLWZ256mi,       0 },
    { X86::VPSLLDQZ256rr,        X86::VPSLLDQZ256rm,        0 },
    { X86::VPSLLDZ256ri,         X86::VPSLLDZ256mi,         0 },
    { X86::VPSLLQZ256ri,         X86::VPSLLQZ256mi,         0 },
    { X86::VPSLLWZ256ri,         X86::VPSLLWZ256mi,         0 },
    { X86::VPSRADZ256ri,         X86::VPSRADZ256mi,         0 },
    { X86::VPSRAQZ256ri,         X86::VPSRAQZ256mi,         0 },
    { X86::VPSRAWZ256ri,         X86::VPSRAWZ256mi,         0 },
    { X86::VPSRLDQZ256rr,        X86::VPSRLDQZ256rm,        0 },
    { X86::VPSRLDZ256ri,         X86::VPSRLDZ256mi,         0 },
    { X86::VPSRLQZ256ri,         X86::VPSRLQZ256mi,         0 },
    { X86::VPSRLWZ256ri,         X86::VPSRLWZ256mi,         0 },

    // AVX-512 foldable instructions (128-bit versions)
    { X86::VBROADCASTSSZ128r,    X86::VBROADCASTSSZ128m,    TB_NO_REVERSE },
    { X86::VMOVAPDZ128rr,        X86::VMOVAPDZ128rm,        TB_ALIGN_16 },
    { X86::VMOVAPSZ128rr,        X86::VMOVAPSZ128rm,        TB_ALIGN_16 },
    { X86::VMOVDQA32Z128rr,      X86::VMOVDQA32Z128rm,      TB_ALIGN_16 },
    { X86::VMOVDQA64Z128rr,      X86::VMOVDQA64Z128rm,      TB_ALIGN_16 },
    { X86::VMOVDQU8Z128rr,       X86::VMOVDQU8Z128rm,       0 },
    { X86::VMOVDQU16Z128rr,      X86::VMOVDQU16Z128rm,      0 },
    { X86::VMOVDQU32Z128rr,      X86::VMOVDQU32Z128rm,      0 },
    { X86::VMOVDQU64Z128rr,      X86::VMOVDQU64Z128rm,      0 },
    { X86::VMOVUPDZ128rr,        X86::VMOVUPDZ128rm,        0 },
    { X86::VMOVUPSZ128rr,        X86::VMOVUPSZ128rm,        0 },
    { X86::VPABSBZ128rr,         X86::VPABSBZ128rm,         0 },
    { X86::VPABSDZ128rr,         X86::VPABSDZ128rm,         0 },
    { X86::VPABSQZ128rr,         X86::VPABSQZ128rm,         0 },
    { X86::VPABSWZ128rr,         X86::VPABSWZ128rm,         0 },
    { X86::VPERMILPDZ128ri,      X86::VPERMILPDZ128mi,      0 },
    { X86::VPERMILPSZ128ri,      X86::VPERMILPSZ128mi,      0 },
    { X86::VPMOVSXBDZ128rr,      X86::VPMOVSXBDZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVSXBQZ128rr,      X86::VPMOVSXBQZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVSXBWZ128rr,      X86::VPMOVSXBWZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVSXDQZ128rr,      X86::VPMOVSXDQZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVSXWDZ128rr,      X86::VPMOVSXWDZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVSXWQZ128rr,      X86::VPMOVSXWQZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVZXBDZ128rr,      X86::VPMOVZXBDZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVZXBQZ128rr,      X86::VPMOVZXBQZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVZXBWZ128rr,      X86::VPMOVZXBWZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVZXDQZ128rr,      X86::VPMOVZXDQZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVZXWDZ128rr,      X86::VPMOVZXWDZ128rm,      TB_NO_REVERSE },
    { X86::VPMOVZXWQZ128rr,      X86::VPMOVZXWQZ128rm,      TB_NO_REVERSE },
    { X86::VPSHUFDZ128ri,        X86::VPSHUFDZ128mi,        0 },
    { X86::VPSHUFHWZ128ri,       X86::VPSHUFHWZ128mi,       0 },
    { X86::VPSHUFLWZ128ri,       X86::VPSHUFLWZ128mi,       0 },
    { X86::VPSLLDQZ128rr,        X86::VPSLLDQZ128rm,        0 },
    { X86::VPSLLDZ128ri,         X86::VPSLLDZ128mi,         0 },
    { X86::VPSLLQZ128ri,         X86::VPSLLQZ128mi,         0 },
    { X86::VPSLLWZ128ri,         X86::VPSLLWZ128mi,         0 },
    { X86::VPSRADZ128ri,         X86::VPSRADZ128mi,         0 },
    { X86::VPSRAQZ128ri,         X86::VPSRAQZ128mi,         0 },
    { X86::VPSRAWZ128ri,         X86::VPSRAWZ128mi,         0 },
    { X86::VPSRLDQZ128rr,        X86::VPSRLDQZ128rm,        0 },
    { X86::VPSRLDZ128ri,         X86::VPSRLDZ128mi,         0 },
    { X86::VPSRLQZ128ri,         X86::VPSRLQZ128mi,         0 },
    { X86::VPSRLWZ128ri,         X86::VPSRLWZ128mi,         0 },

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
    { X86::ADDSDrr_Int,     X86::ADDSDrm_Int,   TB_NO_REVERSE },
    { X86::ADDSSrr,         X86::ADDSSrm,       0 },
    { X86::ADDSSrr_Int,     X86::ADDSSrm_Int,   TB_NO_REVERSE },
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
    { X86::DIVSDrr_Int,     X86::DIVSDrm_Int,   TB_NO_REVERSE },
    { X86::DIVSSrr,         X86::DIVSSrm,       0 },
    { X86::DIVSSrr_Int,     X86::DIVSSrm_Int,   TB_NO_REVERSE },
    { X86::DPPDrri,         X86::DPPDrmi,       TB_ALIGN_16 },
    { X86::DPPSrri,         X86::DPPSrmi,       TB_ALIGN_16 },
    { X86::HADDPDrr,        X86::HADDPDrm,      TB_ALIGN_16 },
    { X86::HADDPSrr,        X86::HADDPSrm,      TB_ALIGN_16 },
    { X86::HSUBPDrr,        X86::HSUBPDrm,      TB_ALIGN_16 },
    { X86::HSUBPSrr,        X86::HSUBPSrm,      TB_ALIGN_16 },
    { X86::IMUL16rr,        X86::IMUL16rm,      0 },
    { X86::IMUL32rr,        X86::IMUL32rm,      0 },
    { X86::IMUL64rr,        X86::IMUL64rm,      0 },
    { X86::Int_CMPSDrr,     X86::Int_CMPSDrm,   TB_NO_REVERSE },
    { X86::Int_CMPSSrr,     X86::Int_CMPSSrm,   TB_NO_REVERSE },
    { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm,      TB_NO_REVERSE },
    { X86::Int_CVTSI2SD64rr,X86::Int_CVTSI2SD64rm,    0 },
    { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm,      0 },
    { X86::Int_CVTSI2SS64rr,X86::Int_CVTSI2SS64rm,    0 },
    { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm,      0 },
    { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm,      TB_NO_REVERSE },
    { X86::MAXPDrr,         X86::MAXPDrm,       TB_ALIGN_16 },
    { X86::MAXCPDrr,        X86::MAXCPDrm,      TB_ALIGN_16 },
    { X86::MAXPSrr,         X86::MAXPSrm,       TB_ALIGN_16 },
    { X86::MAXCPSrr,        X86::MAXCPSrm,      TB_ALIGN_16 },
    { X86::MAXSDrr,         X86::MAXSDrm,       0 },
    { X86::MAXCSDrr,        X86::MAXCSDrm,      0 },
    { X86::MAXSDrr_Int,     X86::MAXSDrm_Int,   TB_NO_REVERSE },
    { X86::MAXSSrr,         X86::MAXSSrm,       0 },
    { X86::MAXCSSrr,        X86::MAXCSSrm,      0 },
    { X86::MAXSSrr_Int,     X86::MAXSSrm_Int,   TB_NO_REVERSE },
    { X86::MINPDrr,         X86::MINPDrm,       TB_ALIGN_16 },
    { X86::MINCPDrr,        X86::MINCPDrm,      TB_ALIGN_16 },
    { X86::MINPSrr,         X86::MINPSrm,       TB_ALIGN_16 },
    { X86::MINCPSrr,        X86::MINCPSrm,      TB_ALIGN_16 },
    { X86::MINSDrr,         X86::MINSDrm,       0 },
    { X86::MINCSDrr,        X86::MINCSDrm,      0 },
    { X86::MINSDrr_Int,     X86::MINSDrm_Int,   TB_NO_REVERSE },
    { X86::MINSSrr,         X86::MINSSrm,       0 },
    { X86::MINCSSrr,        X86::MINCSSrm,      0 },
    { X86::MINSSrr_Int,     X86::MINSSrm_Int,   TB_NO_REVERSE },
    { X86::MOVLHPSrr,       X86::MOVHPSrm,      TB_NO_REVERSE },
    { X86::MPSADBWrri,      X86::MPSADBWrmi,    TB_ALIGN_16 },
    { X86::MULPDrr,         X86::MULPDrm,       TB_ALIGN_16 },
    { X86::MULPSrr,         X86::MULPSrm,       TB_ALIGN_16 },
    { X86::MULSDrr,         X86::MULSDrm,       0 },
    { X86::MULSDrr_Int,     X86::MULSDrm_Int,   TB_NO_REVERSE },
    { X86::MULSSrr,         X86::MULSSrm,       0 },
    { X86::MULSSrr_Int,     X86::MULSSrm_Int,   TB_NO_REVERSE },
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
    { X86::PALIGNRrri,      X86::PALIGNRrmi,    TB_ALIGN_16 },
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
    { X86::PMADDUBSWrr,     X86::PMADDUBSWrm,   TB_ALIGN_16 },
    { X86::PMADDWDrr,       X86::PMADDWDrm,     TB_ALIGN_16 },
    { X86::PMAXSBrr,        X86::PMAXSBrm,      TB_ALIGN_16 },
    { X86::PMAXSDrr,        X86::PMAXSDrm,      TB_ALIGN_16 },
    { X86::PMAXSWrr,        X86::PMAXSWrm,      TB_ALIGN_16 },
    { X86::PMAXUBrr,        X86::PMAXUBrm,      TB_ALIGN_16 },
    { X86::PMAXUDrr,        X86::PMAXUDrm,      TB_ALIGN_16 },
    { X86::PMAXUWrr,        X86::PMAXUWrm,      TB_ALIGN_16 },
    { X86::PMINSBrr,        X86::PMINSBrm,      TB_ALIGN_16 },
    { X86::PMINSDrr,        X86::PMINSDrm,      TB_ALIGN_16 },
    { X86::PMINSWrr,        X86::PMINSWrm,      TB_ALIGN_16 },
    { X86::PMINUBrr,        X86::PMINUBrm,      TB_ALIGN_16 },
    { X86::PMINUDrr,        X86::PMINUDrm,      TB_ALIGN_16 },
    { X86::PMINUWrr,        X86::PMINUWrm,      TB_ALIGN_16 },
    { X86::PMULDQrr,        X86::PMULDQrm,      TB_ALIGN_16 },
    { X86::PMULHRSWrr,      X86::PMULHRSWrm,    TB_ALIGN_16 },
    { X86::PMULHUWrr,       X86::PMULHUWrm,     TB_ALIGN_16 },
    { X86::PMULHWrr,        X86::PMULHWrm,      TB_ALIGN_16 },
    { X86::PMULLDrr,        X86::PMULLDrm,      TB_ALIGN_16 },
    { X86::PMULLWrr,        X86::PMULLWrm,      TB_ALIGN_16 },
    { X86::PMULUDQrr,       X86::PMULUDQrm,     TB_ALIGN_16 },
    { X86::PORrr,           X86::PORrm,         TB_ALIGN_16 },
    { X86::PSADBWrr,        X86::PSADBWrm,      TB_ALIGN_16 },
    { X86::PSHUFBrr,        X86::PSHUFBrm,      TB_ALIGN_16 },
    { X86::PSIGNBrr128,     X86::PSIGNBrm128,   TB_ALIGN_16 },
    { X86::PSIGNWrr128,     X86::PSIGNWrm128,   TB_ALIGN_16 },
    { X86::PSIGNDrr128,     X86::PSIGNDrm128,   TB_ALIGN_16 },
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
    { X86::ROUNDSDr_Int,    X86::ROUNDSDm_Int,  TB_NO_REVERSE },
    { X86::ROUNDSSr_Int,    X86::ROUNDSSm_Int,  TB_NO_REVERSE },
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
    { X86::SUBSDrr_Int,     X86::SUBSDrm_Int,   TB_NO_REVERSE },
    { X86::SUBSSrr,         X86::SUBSSrm,       0 },
    { X86::SUBSSrr_Int,     X86::SUBSSrm_Int,   TB_NO_REVERSE },
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
    { X86::VCVTSI2SD64rr,     X86::VCVTSI2SD64rm,      0 },
    { X86::Int_VCVTSI2SD64rr, X86::Int_VCVTSI2SD64rm,  0 },
    { X86::VCVTSI2SDrr,       X86::VCVTSI2SDrm,        0 },
    { X86::Int_VCVTSI2SDrr,   X86::Int_VCVTSI2SDrm,    0 },
    { X86::VCVTSI2SS64rr,     X86::VCVTSI2SS64rm,      0 },
    { X86::Int_VCVTSI2SS64rr, X86::Int_VCVTSI2SS64rm,  0 },
    { X86::VCVTSI2SSrr,       X86::VCVTSI2SSrm,        0 },
    { X86::Int_VCVTSI2SSrr,   X86::Int_VCVTSI2SSrm,    0 },
    { X86::VADDPDrr,          X86::VADDPDrm,           0 },
    { X86::VADDPSrr,          X86::VADDPSrm,           0 },
    { X86::VADDSDrr,          X86::VADDSDrm,           0 },
    { X86::VADDSDrr_Int,      X86::VADDSDrm_Int,       TB_NO_REVERSE },
    { X86::VADDSSrr,          X86::VADDSSrm,           0 },
    { X86::VADDSSrr_Int,      X86::VADDSSrm_Int,       TB_NO_REVERSE },
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
    { X86::VDIVSDrr_Int,      X86::VDIVSDrm_Int,       TB_NO_REVERSE },
    { X86::VDIVSSrr,          X86::VDIVSSrm,           0 },
    { X86::VDIVSSrr_Int,      X86::VDIVSSrm_Int,       TB_NO_REVERSE },
    { X86::VDPPDrri,          X86::VDPPDrmi,           0 },
    { X86::VDPPSrri,          X86::VDPPSrmi,           0 },
    { X86::VHADDPDrr,         X86::VHADDPDrm,          0 },
    { X86::VHADDPSrr,         X86::VHADDPSrm,          0 },
    { X86::VHSUBPDrr,         X86::VHSUBPDrm,          0 },
    { X86::VHSUBPSrr,         X86::VHSUBPSrm,          0 },
    { X86::Int_VCMPSDrr,      X86::Int_VCMPSDrm,       TB_NO_REVERSE },
    { X86::Int_VCMPSSrr,      X86::Int_VCMPSSrm,       TB_NO_REVERSE },
    { X86::VMAXCPDrr,         X86::VMAXCPDrm,          0 },
    { X86::VMAXCPSrr,         X86::VMAXCPSrm,          0 },
    { X86::VMAXCSDrr,         X86::VMAXCSDrm,          0 },
    { X86::VMAXCSSrr,         X86::VMAXCSSrm,          0 },
    { X86::VMAXPDrr,          X86::VMAXPDrm,           0 },
    { X86::VMAXPSrr,          X86::VMAXPSrm,           0 },
    { X86::VMAXSDrr,          X86::VMAXSDrm,           0 },
    { X86::VMAXSDrr_Int,      X86::VMAXSDrm_Int,       TB_NO_REVERSE },
    { X86::VMAXSSrr,          X86::VMAXSSrm,           0 },
    { X86::VMAXSSrr_Int,      X86::VMAXSSrm_Int,       TB_NO_REVERSE },
    { X86::VMINCPDrr,         X86::VMINCPDrm,          0 },
    { X86::VMINCPSrr,         X86::VMINCPSrm,          0 },
    { X86::VMINCSDrr,         X86::VMINCSDrm,          0 },
    { X86::VMINCSSrr,         X86::VMINCSSrm,          0 },
    { X86::VMINPDrr,          X86::VMINPDrm,           0 },
    { X86::VMINPSrr,          X86::VMINPSrm,           0 },
    { X86::VMINSDrr,          X86::VMINSDrm,           0 },
    { X86::VMINSDrr_Int,      X86::VMINSDrm_Int,       TB_NO_REVERSE },
    { X86::VMINSSrr,          X86::VMINSSrm,           0 },
    { X86::VMINSSrr_Int,      X86::VMINSSrm_Int,       TB_NO_REVERSE },
    { X86::VMOVLHPSrr,        X86::VMOVHPSrm,          TB_NO_REVERSE },
    { X86::VMPSADBWrri,       X86::VMPSADBWrmi,        0 },
    { X86::VMULPDrr,          X86::VMULPDrm,           0 },
    { X86::VMULPSrr,          X86::VMULPSrm,           0 },
    { X86::VMULSDrr,          X86::VMULSDrm,           0 },
    { X86::VMULSDrr_Int,      X86::VMULSDrm_Int,       TB_NO_REVERSE },
    { X86::VMULSSrr,          X86::VMULSSrm,           0 },
    { X86::VMULSSrr_Int,      X86::VMULSSrm_Int,       TB_NO_REVERSE },
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
    { X86::VPALIGNRrri,       X86::VPALIGNRrmi,        0 },
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
    { X86::VPMADDUBSWrr,      X86::VPMADDUBSWrm,       0 },
    { X86::VPMADDWDrr,        X86::VPMADDWDrm,         0 },
    { X86::VPMAXSBrr,         X86::VPMAXSBrm,          0 },
    { X86::VPMAXSDrr,         X86::VPMAXSDrm,          0 },
    { X86::VPMAXSWrr,         X86::VPMAXSWrm,          0 },
    { X86::VPMAXUBrr,         X86::VPMAXUBrm,          0 },
    { X86::VPMAXUDrr,         X86::VPMAXUDrm,          0 },
    { X86::VPMAXUWrr,         X86::VPMAXUWrm,          0 },
    { X86::VPMINSBrr,         X86::VPMINSBrm,          0 },
    { X86::VPMINSDrr,         X86::VPMINSDrm,          0 },
    { X86::VPMINSWrr,         X86::VPMINSWrm,          0 },
    { X86::VPMINUBrr,         X86::VPMINUBrm,          0 },
    { X86::VPMINUDrr,         X86::VPMINUDrm,          0 },
    { X86::VPMINUWrr,         X86::VPMINUWrm,          0 },
    { X86::VPMULDQrr,         X86::VPMULDQrm,          0 },
    { X86::VPMULHRSWrr,       X86::VPMULHRSWrm,        0 },
    { X86::VPMULHUWrr,        X86::VPMULHUWrm,         0 },
    { X86::VPMULHWrr,         X86::VPMULHWrm,          0 },
    { X86::VPMULLDrr,         X86::VPMULLDrm,          0 },
    { X86::VPMULLWrr,         X86::VPMULLWrm,          0 },
    { X86::VPMULUDQrr,        X86::VPMULUDQrm,         0 },
    { X86::VPORrr,            X86::VPORrm,             0 },
    { X86::VPSADBWrr,         X86::VPSADBWrm,          0 },
    { X86::VPSHUFBrr,         X86::VPSHUFBrm,          0 },
    { X86::VPSIGNBrr128,      X86::VPSIGNBrm128,       0 },
    { X86::VPSIGNWrr128,      X86::VPSIGNWrm128,       0 },
    { X86::VPSIGNDrr128,      X86::VPSIGNDrm128,       0 },
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
    { X86::VRCPSSr,           X86::VRCPSSm,            0 },
    { X86::VRCPSSr_Int,       X86::VRCPSSm_Int,        TB_NO_REVERSE },
    { X86::VRSQRTSSr,         X86::VRSQRTSSm,          0 },
    { X86::VRSQRTSSr_Int,     X86::VRSQRTSSm_Int,      TB_NO_REVERSE },
    { X86::VROUNDSDr,         X86::VROUNDSDm,          0 },
    { X86::VROUNDSDr_Int,     X86::VROUNDSDm_Int,      TB_NO_REVERSE },
    { X86::VROUNDSSr,         X86::VROUNDSSm,          0 },
    { X86::VROUNDSSr_Int,     X86::VROUNDSSm_Int,      TB_NO_REVERSE },
    { X86::VSHUFPDrri,        X86::VSHUFPDrmi,         0 },
    { X86::VSHUFPSrri,        X86::VSHUFPSrmi,         0 },
    { X86::VSQRTSDr,          X86::VSQRTSDm,           0 },
    { X86::VSQRTSDr_Int,      X86::VSQRTSDm_Int,       TB_NO_REVERSE },
    { X86::VSQRTSSr,          X86::VSQRTSSm,           0 },
    { X86::VSQRTSSr_Int,      X86::VSQRTSSm_Int,       TB_NO_REVERSE },
    { X86::VSUBPDrr,          X86::VSUBPDrm,           0 },
    { X86::VSUBPSrr,          X86::VSUBPSrm,           0 },
    { X86::VSUBSDrr,          X86::VSUBSDrm,           0 },
    { X86::VSUBSDrr_Int,      X86::VSUBSDrm_Int,       TB_NO_REVERSE },
    { X86::VSUBSSrr,          X86::VSUBSSrm,           0 },
    { X86::VSUBSSrr_Int,      X86::VSUBSSrm_Int,       TB_NO_REVERSE },
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
    { X86::VMAXCPDYrr,        X86::VMAXCPDYrm,         0 },
    { X86::VMAXCPSYrr,        X86::VMAXCPSYrm,         0 },
    { X86::VMAXPDYrr,         X86::VMAXPDYrm,          0 },
    { X86::VMAXPSYrr,         X86::VMAXPSYrm,          0 },
    { X86::VMINCPDYrr,        X86::VMINCPDYrm,         0 },
    { X86::VMINCPSYrr,        X86::VMINCPSYrm,         0 },
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
    { X86::VPALIGNRYrri,      X86::VPALIGNRYrmi,       0 },
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
    { X86::VPMADDUBSWYrr,     X86::VPMADDUBSWYrm,      0 },
    { X86::VPMADDWDYrr,       X86::VPMADDWDYrm,        0 },
    { X86::VPMAXSBYrr,        X86::VPMAXSBYrm,         0 },
    { X86::VPMAXSDYrr,        X86::VPMAXSDYrm,         0 },
    { X86::VPMAXSWYrr,        X86::VPMAXSWYrm,         0 },
    { X86::VPMAXUBYrr,        X86::VPMAXUBYrm,         0 },
    { X86::VPMAXUDYrr,        X86::VPMAXUDYrm,         0 },
    { X86::VPMAXUWYrr,        X86::VPMAXUWYrm,         0 },
    { X86::VPMINSBYrr,        X86::VPMINSBYrm,         0 },
    { X86::VPMINSDYrr,        X86::VPMINSDYrm,         0 },
    { X86::VPMINSWYrr,        X86::VPMINSWYrm,         0 },
    { X86::VPMINUBYrr,        X86::VPMINUBYrm,         0 },
    { X86::VPMINUDYrr,        X86::VPMINUDYrm,         0 },
    { X86::VPMINUWYrr,        X86::VPMINUWYrm,         0 },
    { X86::VMPSADBWYrri,      X86::VMPSADBWYrmi,       0 },
    { X86::VPMULDQYrr,        X86::VPMULDQYrm,         0 },
    { X86::VPMULHRSWYrr,      X86::VPMULHRSWYrm,       0 },
    { X86::VPMULHUWYrr,       X86::VPMULHUWYrm,        0 },
    { X86::VPMULHWYrr,        X86::VPMULHWYrm,         0 },
    { X86::VPMULLDYrr,        X86::VPMULLDYrm,         0 },
    { X86::VPMULLWYrr,        X86::VPMULLWYrm,         0 },
    { X86::VPMULUDQYrr,       X86::VPMULUDQYrm,        0 },
    { X86::VPORYrr,           X86::VPORYrm,            0 },
    { X86::VPSADBWYrr,        X86::VPSADBWYrm,         0 },
    { X86::VPSHUFBYrr,        X86::VPSHUFBYrm,         0 },
    { X86::VPSIGNBYrr256,     X86::VPSIGNBYrm256,      0 },
    { X86::VPSIGNWYrr256,     X86::VPSIGNWYrm256,      0 },
    { X86::VPSIGNDYrr256,     X86::VPSIGNDYrm256,      0 },
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
    { X86::VFMADDSS4rr_Int,   X86::VFMADDSS4mr_Int,    TB_NO_REVERSE },
    { X86::VFMADDSD4rr,       X86::VFMADDSD4mr,        TB_ALIGN_NONE },
    { X86::VFMADDSD4rr_Int,   X86::VFMADDSD4mr_Int,    TB_NO_REVERSE },
    { X86::VFMADDPS4rr,       X86::VFMADDPS4mr,        TB_ALIGN_NONE },
    { X86::VFMADDPD4rr,       X86::VFMADDPD4mr,        TB_ALIGN_NONE },
    { X86::VFMADDPS4Yrr,      X86::VFMADDPS4Ymr,       TB_ALIGN_NONE },
    { X86::VFMADDPD4Yrr,      X86::VFMADDPD4Ymr,       TB_ALIGN_NONE },
    { X86::VFNMADDSS4rr,      X86::VFNMADDSS4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDSS4rr_Int,  X86::VFNMADDSS4mr_Int,   TB_NO_REVERSE },
    { X86::VFNMADDSD4rr,      X86::VFNMADDSD4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDSD4rr_Int,  X86::VFNMADDSD4mr_Int,   TB_NO_REVERSE },
    { X86::VFNMADDPS4rr,      X86::VFNMADDPS4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDPD4rr,      X86::VFNMADDPD4mr,       TB_ALIGN_NONE },
    { X86::VFNMADDPS4Yrr,     X86::VFNMADDPS4Ymr,      TB_ALIGN_NONE },
    { X86::VFNMADDPD4Yrr,     X86::VFNMADDPD4Ymr,      TB_ALIGN_NONE },
    { X86::VFMSUBSS4rr,       X86::VFMSUBSS4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBSS4rr_Int,   X86::VFMSUBSS4mr_Int,    TB_NO_REVERSE },
    { X86::VFMSUBSD4rr,       X86::VFMSUBSD4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBSD4rr_Int,   X86::VFMSUBSD4mr_Int,    TB_NO_REVERSE },
    { X86::VFMSUBPS4rr,       X86::VFMSUBPS4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBPD4rr,       X86::VFMSUBPD4mr,        TB_ALIGN_NONE },
    { X86::VFMSUBPS4Yrr,      X86::VFMSUBPS4Ymr,       TB_ALIGN_NONE },
    { X86::VFMSUBPD4Yrr,      X86::VFMSUBPD4Ymr,       TB_ALIGN_NONE },
    { X86::VFNMSUBSS4rr,      X86::VFNMSUBSS4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBSS4rr_Int,  X86::VFNMSUBSS4mr_Int,   TB_NO_REVERSE },
    { X86::VFNMSUBSD4rr,      X86::VFNMSUBSD4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBSD4rr_Int,  X86::VFNMSUBSD4mr_Int,   TB_NO_REVERSE },
    { X86::VFNMSUBPS4rr,      X86::VFNMSUBPS4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBPD4rr,      X86::VFNMSUBPD4mr,       TB_ALIGN_NONE },
    { X86::VFNMSUBPS4Yrr,     X86::VFNMSUBPS4Ymr,      TB_ALIGN_NONE },
    { X86::VFNMSUBPD4Yrr,     X86::VFNMSUBPD4Ymr,      TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4rr,    X86::VFMADDSUBPS4mr,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4rr,    X86::VFMADDSUBPD4mr,     TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4Yrr,   X86::VFMADDSUBPS4Ymr,    TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4Yrr,   X86::VFMADDSUBPD4Ymr,    TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4rr,    X86::VFMSUBADDPS4mr,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4rr,    X86::VFMSUBADDPD4mr,     TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4Yrr,   X86::VFMSUBADDPS4Ymr,    TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4Yrr,   X86::VFMSUBADDPD4Ymr,    TB_ALIGN_NONE },

    // XOP foldable instructions
    { X86::VPCMOVrrr,         X86::VPCMOVrmr,           0 },
    { X86::VPCMOVYrrr,        X86::VPCMOVYrmr,          0 },
    { X86::VPCOMBri,          X86::VPCOMBmi,            0 },
    { X86::VPCOMDri,          X86::VPCOMDmi,            0 },
    { X86::VPCOMQri,          X86::VPCOMQmi,            0 },
    { X86::VPCOMWri,          X86::VPCOMWmi,            0 },
    { X86::VPCOMUBri,         X86::VPCOMUBmi,           0 },
    { X86::VPCOMUDri,         X86::VPCOMUDmi,           0 },
    { X86::VPCOMUQri,         X86::VPCOMUQmi,           0 },
    { X86::VPCOMUWri,         X86::VPCOMUWmi,           0 },
    { X86::VPERMIL2PDrr,      X86::VPERMIL2PDmr,        0 },
    { X86::VPERMIL2PDYrr,     X86::VPERMIL2PDYmr,       0 },
    { X86::VPERMIL2PSrr,      X86::VPERMIL2PSmr,        0 },
    { X86::VPERMIL2PSYrr,     X86::VPERMIL2PSYmr,       0 },
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
    { X86::VPPERMrrr,         X86::VPPERMrmr,           0 },
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

    // ADX foldable instructions
    { X86::ADCX32rr,          X86::ADCX32rm,            0 },
    { X86::ADCX64rr,          X86::ADCX64rm,            0 },
    { X86::ADOX32rr,          X86::ADOX32rm,            0 },
    { X86::ADOX64rr,          X86::ADOX64rm,            0 },

    // AVX-512 foldable instructions
    { X86::VADDPDZrr,         X86::VADDPDZrm,           0 },
    { X86::VADDPSZrr,         X86::VADDPSZrm,           0 },
    { X86::VADDSDZrr,         X86::VADDSDZrm,           0 },
    { X86::VADDSDZrr_Int,     X86::VADDSDZrm_Int,       TB_NO_REVERSE },
    { X86::VADDSSZrr,         X86::VADDSSZrm,           0 },
    { X86::VADDSSZrr_Int,     X86::VADDSSZrm_Int,       TB_NO_REVERSE },
    { X86::VALIGNDZrri,       X86::VALIGNDZrmi,         0 },
    { X86::VALIGNQZrri,       X86::VALIGNQZrmi,         0 },
    { X86::VANDNPDZrr,        X86::VANDNPDZrm,          0 },
    { X86::VANDNPSZrr,        X86::VANDNPSZrm,          0 },
    { X86::VANDPDZrr,         X86::VANDPDZrm,           0 },
    { X86::VANDPSZrr,         X86::VANDPSZrm,           0 },
    { X86::VCMPPDZrri,        X86::VCMPPDZrmi,          0 },
    { X86::VCMPPSZrri,        X86::VCMPPSZrmi,          0 },
    { X86::VCMPSDZrr,         X86::VCMPSDZrm,           0 },
    { X86::VCMPSDZrr_Int,     X86::VCMPSDZrm_Int,       TB_NO_REVERSE },
    { X86::VCMPSSZrr,         X86::VCMPSSZrm,           0 },
    { X86::VCMPSSZrr_Int,     X86::VCMPSSZrm_Int,       TB_NO_REVERSE },
    { X86::VDIVPDZrr,         X86::VDIVPDZrm,           0 },
    { X86::VDIVPSZrr,         X86::VDIVPSZrm,           0 },
    { X86::VDIVSDZrr,         X86::VDIVSDZrm,           0 },
    { X86::VDIVSDZrr_Int,     X86::VDIVSDZrm_Int,       TB_NO_REVERSE },
    { X86::VDIVSSZrr,         X86::VDIVSSZrm,           0 },
    { X86::VDIVSSZrr_Int,     X86::VDIVSSZrm_Int,       TB_NO_REVERSE },
    { X86::VINSERTF32x4Zrr,   X86::VINSERTF32x4Zrm,     0 },
    { X86::VINSERTF32x8Zrr,   X86::VINSERTF32x8Zrm,     0 },
    { X86::VINSERTF64x2Zrr,   X86::VINSERTF64x2Zrm,     0 },
    { X86::VINSERTF64x4Zrr,   X86::VINSERTF64x4Zrm,     0 },
    { X86::VINSERTI32x4Zrr,   X86::VINSERTI32x4Zrm,     0 },
    { X86::VINSERTI32x8Zrr,   X86::VINSERTI32x8Zrm,     0 },
    { X86::VINSERTI64x2Zrr,   X86::VINSERTI64x2Zrm,     0 },
    { X86::VINSERTI64x4Zrr,   X86::VINSERTI64x4Zrm,     0 },
    { X86::VMAXCPDZrr,        X86::VMAXCPDZrm,          0 },
    { X86::VMAXCPSZrr,        X86::VMAXCPSZrm,          0 },
    { X86::VMAXCSDZrr,        X86::VMAXCSDZrm,          0 },
    { X86::VMAXCSSZrr,        X86::VMAXCSSZrm,          0 },
    { X86::VMAXPDZrr,         X86::VMAXPDZrm,           0 },
    { X86::VMAXPSZrr,         X86::VMAXPSZrm,           0 },
    { X86::VMAXSDZrr,         X86::VMAXSDZrm,           0 },
    { X86::VMAXSDZrr_Int,     X86::VMAXSDZrm_Int,       TB_NO_REVERSE },
    { X86::VMAXSSZrr,         X86::VMAXSSZrm,           0 },
    { X86::VMAXSSZrr_Int,     X86::VMAXSSZrm_Int,       TB_NO_REVERSE },
    { X86::VMINCPDZrr,        X86::VMINCPDZrm,          0 },
    { X86::VMINCPSZrr,        X86::VMINCPSZrm,          0 },
    { X86::VMINCSDZrr,        X86::VMINCSDZrm,          0 },
    { X86::VMINCSSZrr,        X86::VMINCSSZrm,          0 },
    { X86::VMINPDZrr,         X86::VMINPDZrm,           0 },
    { X86::VMINPSZrr,         X86::VMINPSZrm,           0 },
    { X86::VMINSDZrr,         X86::VMINSDZrm,           0 },
    { X86::VMINSDZrr_Int,     X86::VMINSDZrm_Int,       TB_NO_REVERSE },
    { X86::VMINSSZrr,         X86::VMINSSZrm,           0 },
    { X86::VMINSSZrr_Int,     X86::VMINSSZrm_Int,       TB_NO_REVERSE },
    { X86::VMOVLHPSZrr,       X86::VMOVHPSZ128rm,       TB_NO_REVERSE },
    { X86::VMULPDZrr,         X86::VMULPDZrm,           0 },
    { X86::VMULPSZrr,         X86::VMULPSZrm,           0 },
    { X86::VMULSDZrr,         X86::VMULSDZrm,           0 },
    { X86::VMULSDZrr_Int,     X86::VMULSDZrm_Int,       TB_NO_REVERSE },
    { X86::VMULSSZrr,         X86::VMULSSZrm,           0 },
    { X86::VMULSSZrr_Int,     X86::VMULSSZrm_Int,       TB_NO_REVERSE },
    { X86::VORPDZrr,          X86::VORPDZrm,            0 },
    { X86::VORPSZrr,          X86::VORPSZrm,            0 },
    { X86::VPACKSSDWZrr,      X86::VPACKSSDWZrm,        0 },
    { X86::VPACKSSWBZrr,      X86::VPACKSSWBZrm,        0 },
    { X86::VPACKUSDWZrr,      X86::VPACKUSDWZrm,        0 },
    { X86::VPACKUSWBZrr,      X86::VPACKUSWBZrm,        0 },
    { X86::VPADDBZrr,         X86::VPADDBZrm,           0 },
    { X86::VPADDDZrr,         X86::VPADDDZrm,           0 },
    { X86::VPADDQZrr,         X86::VPADDQZrm,           0 },
    { X86::VPADDSBZrr,        X86::VPADDSBZrm,          0 },
    { X86::VPADDSWZrr,        X86::VPADDSWZrm,          0 },
    { X86::VPADDUSBZrr,       X86::VPADDUSBZrm,         0 },
    { X86::VPADDUSWZrr,       X86::VPADDUSWZrm,         0 },
    { X86::VPADDWZrr,         X86::VPADDWZrm,           0 },
    { X86::VPALIGNRZrri,      X86::VPALIGNRZrmi,        0 },
    { X86::VPANDDZrr,         X86::VPANDDZrm,           0 },
    { X86::VPANDNDZrr,        X86::VPANDNDZrm,          0 },
    { X86::VPANDNQZrr,        X86::VPANDNQZrm,          0 },
    { X86::VPANDQZrr,         X86::VPANDQZrm,           0 },
    { X86::VPAVGBZrr,         X86::VPAVGBZrm,           0 },
    { X86::VPAVGWZrr,         X86::VPAVGWZrm,           0 },
    { X86::VPCMPBZrri,        X86::VPCMPBZrmi,          0 },
    { X86::VPCMPDZrri,        X86::VPCMPDZrmi,          0 },
    { X86::VPCMPEQBZrr,       X86::VPCMPEQBZrm,         0 },
    { X86::VPCMPEQDZrr,       X86::VPCMPEQDZrm,         0 },
    { X86::VPCMPEQQZrr,       X86::VPCMPEQQZrm,         0 },
    { X86::VPCMPEQWZrr,       X86::VPCMPEQWZrm,         0 },
    { X86::VPCMPGTBZrr,       X86::VPCMPGTBZrm,         0 },
    { X86::VPCMPGTDZrr,       X86::VPCMPGTDZrm,         0 },
    { X86::VPCMPGTQZrr,       X86::VPCMPGTQZrm,         0 },
    { X86::VPCMPGTWZrr,       X86::VPCMPGTWZrm,         0 },
    { X86::VPCMPQZrri,        X86::VPCMPQZrmi,          0 },
    { X86::VPCMPUBZrri,       X86::VPCMPUBZrmi,         0 },
    { X86::VPCMPUDZrri,       X86::VPCMPUDZrmi,         0 },
    { X86::VPCMPUQZrri,       X86::VPCMPUQZrmi,         0 },
    { X86::VPCMPUWZrri,       X86::VPCMPUWZrmi,         0 },
    { X86::VPCMPWZrri,        X86::VPCMPWZrmi,          0 },
    { X86::VPERMBZrr,         X86::VPERMBZrm,           0 },
    { X86::VPERMDZrr,         X86::VPERMDZrm,           0 },
    { X86::VPERMILPDZrr,      X86::VPERMILPDZrm,        0 },
    { X86::VPERMILPSZrr,      X86::VPERMILPSZrm,        0 },
    { X86::VPERMPDZrr,        X86::VPERMPDZrm,          0 },
    { X86::VPERMPSZrr,        X86::VPERMPSZrm,          0 },
    { X86::VPERMQZrr,         X86::VPERMQZrm,           0 },
    { X86::VPERMWZrr,         X86::VPERMWZrm,           0 },
    { X86::VPINSRBZrr,        X86::VPINSRBZrm,          0 },
    { X86::VPINSRDZrr,        X86::VPINSRDZrm,          0 },
    { X86::VPINSRQZrr,        X86::VPINSRQZrm,          0 },
    { X86::VPINSRWZrr,        X86::VPINSRWZrm,          0 },
    { X86::VPMADDUBSWZrr,     X86::VPMADDUBSWZrm,       0 },
    { X86::VPMADDWDZrr,       X86::VPMADDWDZrm,         0 },
    { X86::VPMAXSBZrr,        X86::VPMAXSBZrm,          0 },
    { X86::VPMAXSDZrr,        X86::VPMAXSDZrm,          0 },
    { X86::VPMAXSQZrr,        X86::VPMAXSQZrm,          0 },
    { X86::VPMAXSWZrr,        X86::VPMAXSWZrm,          0 },
    { X86::VPMAXUBZrr,        X86::VPMAXUBZrm,          0 },
    { X86::VPMAXUDZrr,        X86::VPMAXUDZrm,          0 },
    { X86::VPMAXUQZrr,        X86::VPMAXUQZrm,          0 },
    { X86::VPMAXUWZrr,        X86::VPMAXUWZrm,          0 },
    { X86::VPMINSBZrr,        X86::VPMINSBZrm,          0 },
    { X86::VPMINSDZrr,        X86::VPMINSDZrm,          0 },
    { X86::VPMINSQZrr,        X86::VPMINSQZrm,          0 },
    { X86::VPMINSWZrr,        X86::VPMINSWZrm,          0 },
    { X86::VPMINUBZrr,        X86::VPMINUBZrm,          0 },
    { X86::VPMINUDZrr,        X86::VPMINUDZrm,          0 },
    { X86::VPMINUQZrr,        X86::VPMINUQZrm,          0 },
    { X86::VPMINUWZrr,        X86::VPMINUWZrm,          0 },
    { X86::VPMULDQZrr,        X86::VPMULDQZrm,          0 },
    { X86::VPMULLDZrr,        X86::VPMULLDZrm,          0 },
    { X86::VPMULLQZrr,        X86::VPMULLQZrm,          0 },
    { X86::VPMULLWZrr,        X86::VPMULLWZrm,          0 },
    { X86::VPMULUDQZrr,       X86::VPMULUDQZrm,         0 },
    { X86::VPORDZrr,          X86::VPORDZrm,            0 },
    { X86::VPORQZrr,          X86::VPORQZrm,            0 },
    { X86::VPSADBWZ512rr,     X86::VPSADBWZ512rm,       0 },
    { X86::VPSHUFBZrr,        X86::VPSHUFBZrm,          0 },
    { X86::VPSLLDZrr,         X86::VPSLLDZrm,           0 },
    { X86::VPSLLQZrr,         X86::VPSLLQZrm,           0 },
    { X86::VPSLLVDZrr,        X86::VPSLLVDZrm,          0 },
    { X86::VPSLLVQZrr,        X86::VPSLLVQZrm,          0 },
    { X86::VPSLLVWZrr,        X86::VPSLLVWZrm,          0 },
    { X86::VPSLLWZrr,         X86::VPSLLWZrm,           0 },
    { X86::VPSRADZrr,         X86::VPSRADZrm,           0 },
    { X86::VPSRAQZrr,         X86::VPSRAQZrm,           0 },
    { X86::VPSRAVDZrr,        X86::VPSRAVDZrm,          0 },
    { X86::VPSRAVQZrr,        X86::VPSRAVQZrm,          0 },
    { X86::VPSRAVWZrr,        X86::VPSRAVWZrm,          0 },
    { X86::VPSRAWZrr,         X86::VPSRAWZrm,           0 },
    { X86::VPSRLDZrr,         X86::VPSRLDZrm,           0 },
    { X86::VPSRLQZrr,         X86::VPSRLQZrm,           0 },
    { X86::VPSRLVDZrr,        X86::VPSRLVDZrm,          0 },
    { X86::VPSRLVQZrr,        X86::VPSRLVQZrm,          0 },
    { X86::VPSRLVWZrr,        X86::VPSRLVWZrm,          0 },
    { X86::VPSRLWZrr,         X86::VPSRLWZrm,           0 },
    { X86::VPSUBBZrr,         X86::VPSUBBZrm,           0 },
    { X86::VPSUBDZrr,         X86::VPSUBDZrm,           0 },
    { X86::VPSUBQZrr,         X86::VPSUBQZrm,           0 },
    { X86::VPSUBSBZrr,        X86::VPSUBSBZrm,          0 },
    { X86::VPSUBSWZrr,        X86::VPSUBSWZrm,          0 },
    { X86::VPSUBUSBZrr,       X86::VPSUBUSBZrm,         0 },
    { X86::VPSUBUSWZrr,       X86::VPSUBUSWZrm,         0 },
    { X86::VPSUBWZrr,         X86::VPSUBWZrm,           0 },
    { X86::VPUNPCKHBWZrr,     X86::VPUNPCKHBWZrm,       0 },
    { X86::VPUNPCKHDQZrr,     X86::VPUNPCKHDQZrm,       0 },
    { X86::VPUNPCKHQDQZrr,    X86::VPUNPCKHQDQZrm,      0 },
    { X86::VPUNPCKHWDZrr,     X86::VPUNPCKHWDZrm,       0 },
    { X86::VPUNPCKLBWZrr,     X86::VPUNPCKLBWZrm,       0 },
    { X86::VPUNPCKLDQZrr,     X86::VPUNPCKLDQZrm,       0 },
    { X86::VPUNPCKLQDQZrr,    X86::VPUNPCKLQDQZrm,      0 },
    { X86::VPUNPCKLWDZrr,     X86::VPUNPCKLWDZrm,       0 },
    { X86::VPXORDZrr,         X86::VPXORDZrm,           0 },
    { X86::VPXORQZrr,         X86::VPXORQZrm,           0 },
    { X86::VSHUFPDZrri,       X86::VSHUFPDZrmi,         0 },
    { X86::VSHUFPSZrri,       X86::VSHUFPSZrmi,         0 },
    { X86::VSUBPDZrr,         X86::VSUBPDZrm,           0 },
    { X86::VSUBPSZrr,         X86::VSUBPSZrm,           0 },
    { X86::VSUBSDZrr,         X86::VSUBSDZrm,           0 },
    { X86::VSUBSDZrr_Int,     X86::VSUBSDZrm_Int,       TB_NO_REVERSE },
    { X86::VSUBSSZrr,         X86::VSUBSSZrm,           0 },
    { X86::VSUBSSZrr_Int,     X86::VSUBSSZrm_Int,       TB_NO_REVERSE },
    { X86::VUNPCKHPDZrr,      X86::VUNPCKHPDZrm,        0 },
    { X86::VUNPCKHPSZrr,      X86::VUNPCKHPSZrm,        0 },
    { X86::VUNPCKLPDZrr,      X86::VUNPCKLPDZrm,        0 },
    { X86::VUNPCKLPSZrr,      X86::VUNPCKLPSZrm,        0 },
    { X86::VXORPDZrr,         X86::VXORPDZrm,           0 },
    { X86::VXORPSZrr,         X86::VXORPSZrm,           0 },

    // AVX-512{F,VL} foldable instructions
    { X86::VADDPDZ128rr,      X86::VADDPDZ128rm,        0 },
    { X86::VADDPDZ256rr,      X86::VADDPDZ256rm,        0 },
    { X86::VADDPSZ128rr,      X86::VADDPSZ128rm,        0 },
    { X86::VADDPSZ256rr,      X86::VADDPSZ256rm,        0 },
    { X86::VALIGNDZ128rri,    X86::VALIGNDZ128rmi,      0 },
    { X86::VALIGNDZ256rri,    X86::VALIGNDZ256rmi,      0 },
    { X86::VALIGNQZ128rri,    X86::VALIGNQZ128rmi,      0 },
    { X86::VALIGNQZ256rri,    X86::VALIGNQZ256rmi,      0 },
    { X86::VANDNPDZ128rr,     X86::VANDNPDZ128rm,       0 },
    { X86::VANDNPDZ256rr,     X86::VANDNPDZ256rm,       0 },
    { X86::VANDNPSZ128rr,     X86::VANDNPSZ128rm,       0 },
    { X86::VANDNPSZ256rr,     X86::VANDNPSZ256rm,       0 },
    { X86::VANDPDZ128rr,      X86::VANDPDZ128rm,        0 },
    { X86::VANDPDZ256rr,      X86::VANDPDZ256rm,        0 },
    { X86::VANDPSZ128rr,      X86::VANDPSZ128rm,        0 },
    { X86::VANDPSZ256rr,      X86::VANDPSZ256rm,        0 },
    { X86::VCMPPDZ128rri,     X86::VCMPPDZ128rmi,       0 },
    { X86::VCMPPDZ256rri,     X86::VCMPPDZ256rmi,       0 },
    { X86::VCMPPSZ128rri,     X86::VCMPPSZ128rmi,       0 },
    { X86::VCMPPSZ256rri,     X86::VCMPPSZ256rmi,       0 },
    { X86::VDIVPDZ128rr,      X86::VDIVPDZ128rm,        0 },
    { X86::VDIVPDZ256rr,      X86::VDIVPDZ256rm,        0 },
    { X86::VDIVPSZ128rr,      X86::VDIVPSZ128rm,        0 },
    { X86::VDIVPSZ256rr,      X86::VDIVPSZ256rm,        0 },
    { X86::VINSERTF32x4Z256rr,X86::VINSERTF32x4Z256rm,  0 },
    { X86::VINSERTF64x2Z256rr,X86::VINSERTF64x2Z256rm,  0 },
    { X86::VINSERTI32x4Z256rr,X86::VINSERTI32x4Z256rm,  0 },
    { X86::VINSERTI64x2Z256rr,X86::VINSERTI64x2Z256rm,  0 },
    { X86::VMAXCPDZ128rr,     X86::VMAXCPDZ128rm,       0 },
    { X86::VMAXCPDZ256rr,     X86::VMAXCPDZ256rm,       0 },
    { X86::VMAXCPSZ128rr,     X86::VMAXCPSZ128rm,       0 },
    { X86::VMAXCPSZ256rr,     X86::VMAXCPSZ256rm,       0 },
    { X86::VMAXPDZ128rr,      X86::VMAXPDZ128rm,        0 },
    { X86::VMAXPDZ256rr,      X86::VMAXPDZ256rm,        0 },
    { X86::VMAXPSZ128rr,      X86::VMAXPSZ128rm,        0 },
    { X86::VMAXPSZ256rr,      X86::VMAXPSZ256rm,        0 },
    { X86::VMINCPDZ128rr,     X86::VMINCPDZ128rm,       0 },
    { X86::VMINCPDZ256rr,     X86::VMINCPDZ256rm,       0 },
    { X86::VMINCPSZ128rr,     X86::VMINCPSZ128rm,       0 },
    { X86::VMINCPSZ256rr,     X86::VMINCPSZ256rm,       0 },
    { X86::VMINPDZ128rr,      X86::VMINPDZ128rm,        0 },
    { X86::VMINPDZ256rr,      X86::VMINPDZ256rm,        0 },
    { X86::VMINPSZ128rr,      X86::VMINPSZ128rm,        0 },
    { X86::VMINPSZ256rr,      X86::VMINPSZ256rm,        0 },
    { X86::VMULPDZ128rr,      X86::VMULPDZ128rm,        0 },
    { X86::VMULPDZ256rr,      X86::VMULPDZ256rm,        0 },
    { X86::VMULPSZ128rr,      X86::VMULPSZ128rm,        0 },
    { X86::VMULPSZ256rr,      X86::VMULPSZ256rm,        0 },
    { X86::VORPDZ128rr,       X86::VORPDZ128rm,         0 },
    { X86::VORPDZ256rr,       X86::VORPDZ256rm,         0 },
    { X86::VORPSZ128rr,       X86::VORPSZ128rm,         0 },
    { X86::VORPSZ256rr,       X86::VORPSZ256rm,         0 },
    { X86::VPACKSSDWZ256rr,   X86::VPACKSSDWZ256rm,     0 },
    { X86::VPACKSSDWZ128rr,   X86::VPACKSSDWZ128rm,     0 },
    { X86::VPACKSSWBZ256rr,   X86::VPACKSSWBZ256rm,     0 },
    { X86::VPACKSSWBZ128rr,   X86::VPACKSSWBZ128rm,     0 },
    { X86::VPACKUSDWZ256rr,   X86::VPACKUSDWZ256rm,     0 },
    { X86::VPACKUSDWZ128rr,   X86::VPACKUSDWZ128rm,     0 },
    { X86::VPACKUSWBZ256rr,   X86::VPACKUSWBZ256rm,     0 },
    { X86::VPACKUSWBZ128rr,   X86::VPACKUSWBZ128rm,     0 },
    { X86::VPADDBZ128rr,      X86::VPADDBZ128rm,        0 },
    { X86::VPADDBZ256rr,      X86::VPADDBZ256rm,        0 },
    { X86::VPADDDZ128rr,      X86::VPADDDZ128rm,        0 },
    { X86::VPADDDZ256rr,      X86::VPADDDZ256rm,        0 },
    { X86::VPADDQZ128rr,      X86::VPADDQZ128rm,        0 },
    { X86::VPADDQZ256rr,      X86::VPADDQZ256rm,        0 },
    { X86::VPADDSBZ128rr,     X86::VPADDSBZ128rm,       0 },
    { X86::VPADDSBZ256rr,     X86::VPADDSBZ256rm,       0 },
    { X86::VPADDSWZ128rr,     X86::VPADDSWZ128rm,       0 },
    { X86::VPADDSWZ256rr,     X86::VPADDSWZ256rm,       0 },
    { X86::VPADDUSBZ128rr,    X86::VPADDUSBZ128rm,      0 },
    { X86::VPADDUSBZ256rr,    X86::VPADDUSBZ256rm,      0 },
    { X86::VPADDUSWZ128rr,    X86::VPADDUSWZ128rm,      0 },
    { X86::VPADDUSWZ256rr,    X86::VPADDUSWZ256rm,      0 },
    { X86::VPADDWZ128rr,      X86::VPADDWZ128rm,        0 },
    { X86::VPADDWZ256rr,      X86::VPADDWZ256rm,        0 },
    { X86::VPALIGNRZ128rri,   X86::VPALIGNRZ128rmi,     0 },
    { X86::VPALIGNRZ256rri,   X86::VPALIGNRZ256rmi,     0 },
    { X86::VPANDDZ128rr,      X86::VPANDDZ128rm,        0 },
    { X86::VPANDDZ256rr,      X86::VPANDDZ256rm,        0 },
    { X86::VPANDNDZ128rr,     X86::VPANDNDZ128rm,       0 },
    { X86::VPANDNDZ256rr,     X86::VPANDNDZ256rm,       0 },
    { X86::VPANDNQZ128rr,     X86::VPANDNQZ128rm,       0 },
    { X86::VPANDNQZ256rr,     X86::VPANDNQZ256rm,       0 },
    { X86::VPANDQZ128rr,      X86::VPANDQZ128rm,        0 },
    { X86::VPANDQZ256rr,      X86::VPANDQZ256rm,        0 },
    { X86::VPAVGBZ128rr,      X86::VPAVGBZ128rm,        0 },
    { X86::VPAVGBZ256rr,      X86::VPAVGBZ256rm,        0 },
    { X86::VPAVGWZ128rr,      X86::VPAVGWZ128rm,        0 },
    { X86::VPAVGWZ256rr,      X86::VPAVGWZ256rm,        0 },
    { X86::VPCMPBZ128rri,     X86::VPCMPBZ128rmi,       0 },
    { X86::VPCMPBZ256rri,     X86::VPCMPBZ256rmi,       0 },
    { X86::VPCMPDZ128rri,     X86::VPCMPDZ128rmi,       0 },
    { X86::VPCMPDZ256rri,     X86::VPCMPDZ256rmi,       0 },
    { X86::VPCMPEQBZ128rr,    X86::VPCMPEQBZ128rm,      0 },
    { X86::VPCMPEQBZ256rr,    X86::VPCMPEQBZ256rm,      0 },
    { X86::VPCMPEQDZ128rr,    X86::VPCMPEQDZ128rm,      0 },
    { X86::VPCMPEQDZ256rr,    X86::VPCMPEQDZ256rm,      0 },
    { X86::VPCMPEQQZ128rr,    X86::VPCMPEQQZ128rm,      0 },
    { X86::VPCMPEQQZ256rr,    X86::VPCMPEQQZ256rm,      0 },
    { X86::VPCMPEQWZ128rr,    X86::VPCMPEQWZ128rm,      0 },
    { X86::VPCMPEQWZ256rr,    X86::VPCMPEQWZ256rm,      0 },
    { X86::VPCMPGTBZ128rr,    X86::VPCMPGTBZ128rm,      0 },
    { X86::VPCMPGTBZ256rr,    X86::VPCMPGTBZ256rm,      0 },
    { X86::VPCMPGTDZ128rr,    X86::VPCMPGTDZ128rm,      0 },
    { X86::VPCMPGTDZ256rr,    X86::VPCMPGTDZ256rm,      0 },
    { X86::VPCMPGTQZ128rr,    X86::VPCMPGTQZ128rm,      0 },
    { X86::VPCMPGTQZ256rr,    X86::VPCMPGTQZ256rm,      0 },
    { X86::VPCMPGTWZ128rr,    X86::VPCMPGTWZ128rm,      0 },
    { X86::VPCMPGTWZ256rr,    X86::VPCMPGTWZ256rm,      0 },
    { X86::VPCMPQZ128rri,     X86::VPCMPQZ128rmi,       0 },
    { X86::VPCMPQZ256rri,     X86::VPCMPQZ256rmi,       0 },
    { X86::VPCMPUBZ128rri,    X86::VPCMPUBZ128rmi,      0 },
    { X86::VPCMPUBZ256rri,    X86::VPCMPUBZ256rmi,      0 },
    { X86::VPCMPUDZ128rri,    X86::VPCMPUDZ128rmi,      0 },
    { X86::VPCMPUDZ256rri,    X86::VPCMPUDZ256rmi,      0 },
    { X86::VPCMPUQZ128rri,    X86::VPCMPUQZ128rmi,      0 },
    { X86::VPCMPUQZ256rri,    X86::VPCMPUQZ256rmi,      0 },
    { X86::VPCMPUWZ128rri,    X86::VPCMPUWZ128rmi,      0 },
    { X86::VPCMPUWZ256rri,    X86::VPCMPUWZ256rmi,      0 },
    { X86::VPCMPWZ128rri,     X86::VPCMPWZ128rmi,       0 },
    { X86::VPCMPWZ256rri,     X86::VPCMPWZ256rmi,       0 },
    { X86::VPERMBZ128rr,      X86::VPERMBZ128rm,        0 },
    { X86::VPERMBZ256rr,      X86::VPERMBZ256rm,        0 },
    { X86::VPERMDZ256rr,      X86::VPERMDZ256rm,        0 },
    { X86::VPERMILPDZ128rr,   X86::VPERMILPDZ128rm,     0 },
    { X86::VPERMILPDZ256rr,   X86::VPERMILPDZ256rm,     0 },
    { X86::VPERMILPSZ128rr,   X86::VPERMILPSZ128rm,     0 },
    { X86::VPERMILPSZ256rr,   X86::VPERMILPSZ256rm,     0 },
    { X86::VPERMPDZ256rr,     X86::VPERMPDZ256rm,       0 },
    { X86::VPERMPSZ256rr,     X86::VPERMPSZ256rm,       0 },
    { X86::VPERMQZ256rr,      X86::VPERMQZ256rm,        0 },
    { X86::VPERMWZ128rr,      X86::VPERMWZ128rm,        0 },
    { X86::VPERMWZ256rr,      X86::VPERMWZ256rm,        0 },
    { X86::VPMADDUBSWZ128rr,  X86::VPMADDUBSWZ128rm,    0 },
    { X86::VPMADDUBSWZ256rr,  X86::VPMADDUBSWZ256rm,    0 },
    { X86::VPMADDWDZ128rr,    X86::VPMADDWDZ128rm,      0 },
    { X86::VPMADDWDZ256rr,    X86::VPMADDWDZ256rm,      0 },
    { X86::VPMAXSBZ128rr,     X86::VPMAXSBZ128rm,       0 },
    { X86::VPMAXSBZ256rr,     X86::VPMAXSBZ256rm,       0 },
    { X86::VPMAXSDZ128rr,     X86::VPMAXSDZ128rm,       0 },
    { X86::VPMAXSDZ256rr,     X86::VPMAXSDZ256rm,       0 },
    { X86::VPMAXSQZ128rr,     X86::VPMAXSQZ128rm,       0 },
    { X86::VPMAXSQZ256rr,     X86::VPMAXSQZ256rm,       0 },
    { X86::VPMAXSWZ128rr,     X86::VPMAXSWZ128rm,       0 },
    { X86::VPMAXSWZ256rr,     X86::VPMAXSWZ256rm,       0 },
    { X86::VPMAXUBZ128rr,     X86::VPMAXUBZ128rm,       0 },
    { X86::VPMAXUBZ256rr,     X86::VPMAXUBZ256rm,       0 },
    { X86::VPMAXUDZ128rr,     X86::VPMAXUDZ128rm,       0 },
    { X86::VPMAXUDZ256rr,     X86::VPMAXUDZ256rm,       0 },
    { X86::VPMAXUQZ128rr,     X86::VPMAXUQZ128rm,       0 },
    { X86::VPMAXUQZ256rr,     X86::VPMAXUQZ256rm,       0 },
    { X86::VPMAXUWZ128rr,     X86::VPMAXUWZ128rm,       0 },
    { X86::VPMAXUWZ256rr,     X86::VPMAXUWZ256rm,       0 },
    { X86::VPMINSBZ128rr,     X86::VPMINSBZ128rm,       0 },
    { X86::VPMINSBZ256rr,     X86::VPMINSBZ256rm,       0 },
    { X86::VPMINSDZ128rr,     X86::VPMINSDZ128rm,       0 },
    { X86::VPMINSDZ256rr,     X86::VPMINSDZ256rm,       0 },
    { X86::VPMINSQZ128rr,     X86::VPMINSQZ128rm,       0 },
    { X86::VPMINSQZ256rr,     X86::VPMINSQZ256rm,       0 },
    { X86::VPMINSWZ128rr,     X86::VPMINSWZ128rm,       0 },
    { X86::VPMINSWZ256rr,     X86::VPMINSWZ256rm,       0 },
    { X86::VPMINUBZ128rr,     X86::VPMINUBZ128rm,       0 },
    { X86::VPMINUBZ256rr,     X86::VPMINUBZ256rm,       0 },
    { X86::VPMINUDZ128rr,     X86::VPMINUDZ128rm,       0 },
    { X86::VPMINUDZ256rr,     X86::VPMINUDZ256rm,       0 },
    { X86::VPMINUQZ128rr,     X86::VPMINUQZ128rm,       0 },
    { X86::VPMINUQZ256rr,     X86::VPMINUQZ256rm,       0 },
    { X86::VPMINUWZ128rr,     X86::VPMINUWZ128rm,       0 },
    { X86::VPMINUWZ256rr,     X86::VPMINUWZ256rm,       0 },
    { X86::VPMULDQZ128rr,     X86::VPMULDQZ128rm,       0 },
    { X86::VPMULDQZ256rr,     X86::VPMULDQZ256rm,       0 },
    { X86::VPMULLDZ128rr,     X86::VPMULLDZ128rm,       0 },
    { X86::VPMULLDZ256rr,     X86::VPMULLDZ256rm,       0 },
    { X86::VPMULLQZ128rr,     X86::VPMULLQZ128rm,       0 },
    { X86::VPMULLQZ256rr,     X86::VPMULLQZ256rm,       0 },
    { X86::VPMULLWZ128rr,     X86::VPMULLWZ128rm,       0 },
    { X86::VPMULLWZ256rr,     X86::VPMULLWZ256rm,       0 },
    { X86::VPMULUDQZ128rr,    X86::VPMULUDQZ128rm,      0 },
    { X86::VPMULUDQZ256rr,    X86::VPMULUDQZ256rm,      0 },
    { X86::VPORDZ128rr,       X86::VPORDZ128rm,         0 },
    { X86::VPORDZ256rr,       X86::VPORDZ256rm,         0 },
    { X86::VPORQZ128rr,       X86::VPORQZ128rm,         0 },
    { X86::VPORQZ256rr,       X86::VPORQZ256rm,         0 },
    { X86::VPSADBWZ128rr,     X86::VPSADBWZ128rm,       0 },
    { X86::VPSADBWZ256rr,     X86::VPSADBWZ256rm,       0 },
    { X86::VPSHUFBZ128rr,     X86::VPSHUFBZ128rm,       0 },
    { X86::VPSHUFBZ256rr,     X86::VPSHUFBZ256rm,       0 },
    { X86::VPSLLDZ128rr,      X86::VPSLLDZ128rm,        0 },
    { X86::VPSLLDZ256rr,      X86::VPSLLDZ256rm,        0 },
    { X86::VPSLLQZ128rr,      X86::VPSLLQZ128rm,        0 },
    { X86::VPSLLQZ256rr,      X86::VPSLLQZ256rm,        0 },
    { X86::VPSLLVDZ128rr,     X86::VPSLLVDZ128rm,       0 },
    { X86::VPSLLVDZ256rr,     X86::VPSLLVDZ256rm,       0 },
    { X86::VPSLLVQZ128rr,     X86::VPSLLVQZ128rm,       0 },
    { X86::VPSLLVQZ256rr,     X86::VPSLLVQZ256rm,       0 },
    { X86::VPSLLVWZ128rr,     X86::VPSLLVWZ128rm,       0 },
    { X86::VPSLLVWZ256rr,     X86::VPSLLVWZ256rm,       0 },
    { X86::VPSLLWZ128rr,      X86::VPSLLWZ128rm,        0 },
    { X86::VPSLLWZ256rr,      X86::VPSLLWZ256rm,        0 },
    { X86::VPSRADZ128rr,      X86::VPSRADZ128rm,        0 },
    { X86::VPSRADZ256rr,      X86::VPSRADZ256rm,        0 },
    { X86::VPSRAQZ128rr,      X86::VPSRAQZ128rm,        0 },
    { X86::VPSRAQZ256rr,      X86::VPSRAQZ256rm,        0 },
    { X86::VPSRAVDZ128rr,     X86::VPSRAVDZ128rm,       0 },
    { X86::VPSRAVDZ256rr,     X86::VPSRAVDZ256rm,       0 },
    { X86::VPSRAVQZ128rr,     X86::VPSRAVQZ128rm,       0 },
    { X86::VPSRAVQZ256rr,     X86::VPSRAVQZ256rm,       0 },
    { X86::VPSRAVWZ128rr,     X86::VPSRAVWZ128rm,       0 },
    { X86::VPSRAVWZ256rr,     X86::VPSRAVWZ256rm,       0 },
    { X86::VPSRAWZ128rr,      X86::VPSRAWZ128rm,        0 },
    { X86::VPSRAWZ256rr,      X86::VPSRAWZ256rm,        0 },
    { X86::VPSRLDZ128rr,      X86::VPSRLDZ128rm,        0 },
    { X86::VPSRLDZ256rr,      X86::VPSRLDZ256rm,        0 },
    { X86::VPSRLQZ128rr,      X86::VPSRLQZ128rm,        0 },
    { X86::VPSRLQZ256rr,      X86::VPSRLQZ256rm,        0 },
    { X86::VPSRLVDZ128rr,     X86::VPSRLVDZ128rm,       0 },
    { X86::VPSRLVDZ256rr,     X86::VPSRLVDZ256rm,       0 },
    { X86::VPSRLVQZ128rr,     X86::VPSRLVQZ128rm,       0 },
    { X86::VPSRLVQZ256rr,     X86::VPSRLVQZ256rm,       0 },
    { X86::VPSRLVWZ128rr,     X86::VPSRLVWZ128rm,       0 },
    { X86::VPSRLVWZ256rr,     X86::VPSRLVWZ256rm,       0 },
    { X86::VPSRLWZ128rr,      X86::VPSRLWZ128rm,        0 },
    { X86::VPSRLWZ256rr,      X86::VPSRLWZ256rm,        0 },
    { X86::VPSUBBZ128rr,      X86::VPSUBBZ128rm,        0 },
    { X86::VPSUBBZ256rr,      X86::VPSUBBZ256rm,        0 },
    { X86::VPSUBDZ128rr,      X86::VPSUBDZ128rm,        0 },
    { X86::VPSUBDZ256rr,      X86::VPSUBDZ256rm,        0 },
    { X86::VPSUBQZ128rr,      X86::VPSUBQZ128rm,        0 },
    { X86::VPSUBQZ256rr,      X86::VPSUBQZ256rm,        0 },
    { X86::VPSUBSBZ128rr,     X86::VPSUBSBZ128rm,       0 },
    { X86::VPSUBSBZ256rr,     X86::VPSUBSBZ256rm,       0 },
    { X86::VPSUBSWZ128rr,     X86::VPSUBSWZ128rm,       0 },
    { X86::VPSUBSWZ256rr,     X86::VPSUBSWZ256rm,       0 },
    { X86::VPSUBUSBZ128rr,    X86::VPSUBUSBZ128rm,      0 },
    { X86::VPSUBUSBZ256rr,    X86::VPSUBUSBZ256rm,      0 },
    { X86::VPSUBUSWZ128rr,    X86::VPSUBUSWZ128rm,      0 },
    { X86::VPSUBUSWZ256rr,    X86::VPSUBUSWZ256rm,      0 },
    { X86::VPSUBWZ128rr,      X86::VPSUBWZ128rm,        0 },
    { X86::VPSUBWZ256rr,      X86::VPSUBWZ256rm,        0 },
    { X86::VPUNPCKHBWZ128rr,  X86::VPUNPCKHBWZ128rm,    0 },
    { X86::VPUNPCKHBWZ256rr,  X86::VPUNPCKHBWZ256rm,    0 },
    { X86::VPUNPCKHDQZ128rr,  X86::VPUNPCKHDQZ128rm,    0 },
    { X86::VPUNPCKHDQZ256rr,  X86::VPUNPCKHDQZ256rm,    0 },
    { X86::VPUNPCKHQDQZ128rr, X86::VPUNPCKHQDQZ128rm,   0 },
    { X86::VPUNPCKHQDQZ256rr, X86::VPUNPCKHQDQZ256rm,   0 },
    { X86::VPUNPCKHWDZ128rr,  X86::VPUNPCKHWDZ128rm,    0 },
    { X86::VPUNPCKHWDZ256rr,  X86::VPUNPCKHWDZ256rm,    0 },
    { X86::VPUNPCKLBWZ128rr,  X86::VPUNPCKLBWZ128rm,    0 },
    { X86::VPUNPCKLBWZ256rr,  X86::VPUNPCKLBWZ256rm,    0 },
    { X86::VPUNPCKLDQZ128rr,  X86::VPUNPCKLDQZ128rm,    0 },
    { X86::VPUNPCKLDQZ256rr,  X86::VPUNPCKLDQZ256rm,    0 },
    { X86::VPUNPCKLQDQZ128rr, X86::VPUNPCKLQDQZ128rm,   0 },
    { X86::VPUNPCKLQDQZ256rr, X86::VPUNPCKLQDQZ256rm,   0 },
    { X86::VPUNPCKLWDZ128rr,  X86::VPUNPCKLWDZ128rm,    0 },
    { X86::VPUNPCKLWDZ256rr,  X86::VPUNPCKLWDZ256rm,    0 },
    { X86::VPXORDZ128rr,      X86::VPXORDZ128rm,        0 },
    { X86::VPXORDZ256rr,      X86::VPXORDZ256rm,        0 },
    { X86::VPXORQZ128rr,      X86::VPXORQZ128rm,        0 },
    { X86::VPXORQZ256rr,      X86::VPXORQZ256rm,        0 },
    { X86::VSHUFPDZ128rri,    X86::VSHUFPDZ128rmi,      0 },
    { X86::VSHUFPDZ256rri,    X86::VSHUFPDZ256rmi,      0 },
    { X86::VSHUFPSZ128rri,    X86::VSHUFPSZ128rmi,      0 },
    { X86::VSHUFPSZ256rri,    X86::VSHUFPSZ256rmi,      0 },
    { X86::VSUBPDZ128rr,      X86::VSUBPDZ128rm,        0 },
    { X86::VSUBPDZ256rr,      X86::VSUBPDZ256rm,        0 },
    { X86::VSUBPSZ128rr,      X86::VSUBPSZ128rm,        0 },
    { X86::VSUBPSZ256rr,      X86::VSUBPSZ256rm,        0 },
    { X86::VUNPCKHPDZ128rr,   X86::VUNPCKHPDZ128rm,     0 },
    { X86::VUNPCKHPDZ256rr,   X86::VUNPCKHPDZ256rm,     0 },
    { X86::VUNPCKHPSZ128rr,   X86::VUNPCKHPSZ128rm,     0 },
    { X86::VUNPCKHPSZ256rr,   X86::VUNPCKHPSZ256rm,     0 },
    { X86::VUNPCKLPDZ128rr,   X86::VUNPCKLPDZ128rm,     0 },
    { X86::VUNPCKLPDZ256rr,   X86::VUNPCKLPDZ256rm,     0 },
    { X86::VUNPCKLPSZ128rr,   X86::VUNPCKLPSZ128rm,     0 },
    { X86::VUNPCKLPSZ256rr,   X86::VUNPCKLPSZ256rm,     0 },
    { X86::VXORPDZ128rr,      X86::VXORPDZ128rm,        0 },
    { X86::VXORPDZ256rr,      X86::VXORPDZ256rm,        0 },
    { X86::VXORPSZ128rr,      X86::VXORPSZ128rm,        0 },
    { X86::VXORPSZ256rr,      X86::VXORPSZ256rm,        0 },

    // AVX-512 masked foldable instructions
    { X86::VBROADCASTSSZrkz,  X86::VBROADCASTSSZmkz,    TB_NO_REVERSE },
    { X86::VBROADCASTSDZrkz,  X86::VBROADCASTSDZmkz,    TB_NO_REVERSE },
    { X86::VPABSBZrrkz,       X86::VPABSBZrmkz,         0 },
    { X86::VPABSDZrrkz,       X86::VPABSDZrmkz,         0 },
    { X86::VPABSQZrrkz,       X86::VPABSQZrmkz,         0 },
    { X86::VPABSWZrrkz,       X86::VPABSWZrmkz,         0 },
    { X86::VPERMILPDZrikz,    X86::VPERMILPDZmikz,      0 },
    { X86::VPERMILPSZrikz,    X86::VPERMILPSZmikz,      0 },
    { X86::VPERMPDZrikz,      X86::VPERMPDZmikz,        0 },
    { X86::VPERMQZrikz,       X86::VPERMQZmikz,         0 },
    { X86::VPMOVSXBDZrrkz,    X86::VPMOVSXBDZrmkz,      0 },
    { X86::VPMOVSXBQZrrkz,    X86::VPMOVSXBQZrmkz,      TB_NO_REVERSE },
    { X86::VPMOVSXBWZrrkz,    X86::VPMOVSXBWZrmkz,      0 },
    { X86::VPMOVSXDQZrrkz,    X86::VPMOVSXDQZrmkz,      0 },
    { X86::VPMOVSXWDZrrkz,    X86::VPMOVSXWDZrmkz,      0 },
    { X86::VPMOVSXWQZrrkz,    X86::VPMOVSXWQZrmkz,      0 },
    { X86::VPMOVZXBDZrrkz,    X86::VPMOVZXBDZrmkz,      0 },
    { X86::VPMOVZXBQZrrkz,    X86::VPMOVZXBQZrmkz,      TB_NO_REVERSE },
    { X86::VPMOVZXBWZrrkz,    X86::VPMOVZXBWZrmkz,      0 },
    { X86::VPMOVZXDQZrrkz,    X86::VPMOVZXDQZrmkz,      0 },
    { X86::VPMOVZXWDZrrkz,    X86::VPMOVZXWDZrmkz,      0 },
    { X86::VPMOVZXWQZrrkz,    X86::VPMOVZXWQZrmkz,      0 },
    { X86::VPSHUFDZrikz,      X86::VPSHUFDZmikz,        0 },
    { X86::VPSHUFHWZrikz,     X86::VPSHUFHWZmikz,       0 },
    { X86::VPSHUFLWZrikz,     X86::VPSHUFLWZmikz,       0 },
    { X86::VPSLLDZrikz,       X86::VPSLLDZmikz,         0 },
    { X86::VPSLLQZrikz,       X86::VPSLLQZmikz,         0 },
    { X86::VPSLLWZrikz,       X86::VPSLLWZmikz,         0 },
    { X86::VPSRADZrikz,       X86::VPSRADZmikz,         0 },
    { X86::VPSRAQZrikz,       X86::VPSRAQZmikz,         0 },
    { X86::VPSRAWZrikz,       X86::VPSRAWZmikz,         0 },
    { X86::VPSRLDZrikz,       X86::VPSRLDZmikz,         0 },
    { X86::VPSRLQZrikz,       X86::VPSRLQZmikz,         0 },
    { X86::VPSRLWZrikz,       X86::VPSRLWZmikz,         0 },

    // AVX-512VL 256-bit masked foldable instructions
    { X86::VBROADCASTSDZ256rkz,  X86::VBROADCASTSDZ256mkz,      TB_NO_REVERSE },
    { X86::VBROADCASTSSZ256rkz,  X86::VBROADCASTSSZ256mkz,      TB_NO_REVERSE },
    { X86::VPABSBZ256rrkz,    X86::VPABSBZ256rmkz,      0 },
    { X86::VPABSDZ256rrkz,    X86::VPABSDZ256rmkz,      0 },
    { X86::VPABSQZ256rrkz,    X86::VPABSQZ256rmkz,      0 },
    { X86::VPABSWZ256rrkz,    X86::VPABSWZ256rmkz,      0 },
    { X86::VPERMILPDZ256rikz, X86::VPERMILPDZ256mikz,   0 },
    { X86::VPERMILPSZ256rikz, X86::VPERMILPSZ256mikz,   0 },
    { X86::VPERMPDZ256rikz,   X86::VPERMPDZ256mikz,     0 },
    { X86::VPERMQZ256rikz,    X86::VPERMQZ256mikz,      0 },
    { X86::VPMOVSXBDZ256rrkz, X86::VPMOVSXBDZ256rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXBQZ256rrkz, X86::VPMOVSXBQZ256rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXBWZ256rrkz, X86::VPMOVSXBWZ256rmkz,   0 },
    { X86::VPMOVSXDQZ256rrkz, X86::VPMOVSXDQZ256rmkz,   0 },
    { X86::VPMOVSXWDZ256rrkz, X86::VPMOVSXWDZ256rmkz,   0 },
    { X86::VPMOVSXWQZ256rrkz, X86::VPMOVSXWQZ256rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXBDZ256rrkz, X86::VPMOVZXBDZ256rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXBQZ256rrkz, X86::VPMOVZXBQZ256rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXBWZ256rrkz, X86::VPMOVZXBWZ256rmkz,   0 },
    { X86::VPMOVZXDQZ256rrkz, X86::VPMOVZXDQZ256rmkz,   0 },
    { X86::VPMOVZXWDZ256rrkz, X86::VPMOVZXWDZ256rmkz,   0 },
    { X86::VPMOVZXWQZ256rrkz, X86::VPMOVZXWQZ256rmkz,   TB_NO_REVERSE },
    { X86::VPSHUFDZ256rikz,   X86::VPSHUFDZ256mikz,     0 },
    { X86::VPSHUFHWZ256rikz,  X86::VPSHUFHWZ256mikz,    0 },
    { X86::VPSHUFLWZ256rikz,  X86::VPSHUFLWZ256mikz,    0 },
    { X86::VPSLLDZ256rikz,    X86::VPSLLDZ256mikz,      0 },
    { X86::VPSLLQZ256rikz,    X86::VPSLLQZ256mikz,      0 },
    { X86::VPSLLWZ256rikz,    X86::VPSLLWZ256mikz,      0 },
    { X86::VPSRADZ256rikz,    X86::VPSRADZ256mikz,      0 },
    { X86::VPSRAQZ256rikz,    X86::VPSRAQZ256mikz,      0 },
    { X86::VPSRAWZ256rikz,    X86::VPSRAWZ256mikz,      0 },
    { X86::VPSRLDZ256rikz,    X86::VPSRLDZ256mikz,      0 },
    { X86::VPSRLQZ256rikz,    X86::VPSRLQZ256mikz,      0 },
    { X86::VPSRLWZ256rikz,    X86::VPSRLWZ256mikz,      0 },

    // AVX-512VL 128-bit masked foldable instructions
    { X86::VBROADCASTSSZ128rkz,  X86::VBROADCASTSSZ128mkz,      TB_NO_REVERSE },
    { X86::VPABSBZ128rrkz,    X86::VPABSBZ128rmkz,      0 },
    { X86::VPABSDZ128rrkz,    X86::VPABSDZ128rmkz,      0 },
    { X86::VPABSQZ128rrkz,    X86::VPABSQZ128rmkz,      0 },
    { X86::VPABSWZ128rrkz,    X86::VPABSWZ128rmkz,      0 },
    { X86::VPERMILPDZ128rikz, X86::VPERMILPDZ128mikz,   0 },
    { X86::VPERMILPSZ128rikz, X86::VPERMILPSZ128mikz,   0 },
    { X86::VPMOVSXBDZ128rrkz, X86::VPMOVSXBDZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXBQZ128rrkz, X86::VPMOVSXBQZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXBWZ128rrkz, X86::VPMOVSXBWZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXDQZ128rrkz, X86::VPMOVSXDQZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXWDZ128rrkz, X86::VPMOVSXWDZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVSXWQZ128rrkz, X86::VPMOVSXWQZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXBDZ128rrkz, X86::VPMOVZXBDZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXBQZ128rrkz, X86::VPMOVZXBQZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXBWZ128rrkz, X86::VPMOVZXBWZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXDQZ128rrkz, X86::VPMOVZXDQZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXWDZ128rrkz, X86::VPMOVZXWDZ128rmkz,   TB_NO_REVERSE },
    { X86::VPMOVZXWQZ128rrkz, X86::VPMOVZXWQZ128rmkz,   TB_NO_REVERSE },
    { X86::VPSHUFDZ128rikz,   X86::VPSHUFDZ128mikz,     0 },
    { X86::VPSHUFHWZ128rikz,  X86::VPSHUFHWZ128mikz,    0 },
    { X86::VPSHUFLWZ128rikz,  X86::VPSHUFLWZ128mikz,    0 },
    { X86::VPSLLDZ128rikz,    X86::VPSLLDZ128mikz,      0 },
    { X86::VPSLLQZ128rikz,    X86::VPSLLQZ128mikz,      0 },
    { X86::VPSLLWZ128rikz,    X86::VPSLLWZ128mikz,      0 },
    { X86::VPSRADZ128rikz,    X86::VPSRADZ128mikz,      0 },
    { X86::VPSRAQZ128rikz,    X86::VPSRAQZ128mikz,      0 },
    { X86::VPSRAWZ128rikz,    X86::VPSRAWZ128mikz,      0 },
    { X86::VPSRLDZ128rikz,    X86::VPSRLDZ128mikz,      0 },
    { X86::VPSRLQZ128rikz,    X86::VPSRLQZ128mikz,      0 },
    { X86::VPSRLWZ128rikz,    X86::VPSRLWZ128mikz,      0 },

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
    // FMA4 foldable patterns
    { X86::VFMADDSS4rr,           X86::VFMADDSS4rm,           TB_ALIGN_NONE },
    { X86::VFMADDSS4rr_Int,       X86::VFMADDSS4rm_Int,       TB_NO_REVERSE },
    { X86::VFMADDSD4rr,           X86::VFMADDSD4rm,           TB_ALIGN_NONE },
    { X86::VFMADDSD4rr_Int,       X86::VFMADDSD4rm_Int,       TB_NO_REVERSE },
    { X86::VFMADDPS4rr,           X86::VFMADDPS4rm,           TB_ALIGN_NONE },
    { X86::VFMADDPD4rr,           X86::VFMADDPD4rm,           TB_ALIGN_NONE },
    { X86::VFMADDPS4Yrr,          X86::VFMADDPS4Yrm,          TB_ALIGN_NONE },
    { X86::VFMADDPD4Yrr,          X86::VFMADDPD4Yrm,          TB_ALIGN_NONE },
    { X86::VFNMADDSS4rr,          X86::VFNMADDSS4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDSS4rr_Int,      X86::VFNMADDSS4rm_Int,      TB_NO_REVERSE },
    { X86::VFNMADDSD4rr,          X86::VFNMADDSD4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDSD4rr_Int,      X86::VFNMADDSD4rm_Int,      TB_NO_REVERSE },
    { X86::VFNMADDPS4rr,          X86::VFNMADDPS4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDPD4rr,          X86::VFNMADDPD4rm,          TB_ALIGN_NONE },
    { X86::VFNMADDPS4Yrr,         X86::VFNMADDPS4Yrm,         TB_ALIGN_NONE },
    { X86::VFNMADDPD4Yrr,         X86::VFNMADDPD4Yrm,         TB_ALIGN_NONE },
    { X86::VFMSUBSS4rr,           X86::VFMSUBSS4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBSS4rr_Int,       X86::VFMSUBSS4rm_Int,       TB_NO_REVERSE },
    { X86::VFMSUBSD4rr,           X86::VFMSUBSD4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBSD4rr_Int,       X86::VFMSUBSD4rm_Int,       TB_NO_REVERSE },
    { X86::VFMSUBPS4rr,           X86::VFMSUBPS4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBPD4rr,           X86::VFMSUBPD4rm,           TB_ALIGN_NONE },
    { X86::VFMSUBPS4Yrr,          X86::VFMSUBPS4Yrm,          TB_ALIGN_NONE },
    { X86::VFMSUBPD4Yrr,          X86::VFMSUBPD4Yrm,          TB_ALIGN_NONE },
    { X86::VFNMSUBSS4rr,          X86::VFNMSUBSS4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBSS4rr_Int,      X86::VFNMSUBSS4rm_Int,      TB_NO_REVERSE },
    { X86::VFNMSUBSD4rr,          X86::VFNMSUBSD4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBSD4rr_Int,      X86::VFNMSUBSD4rm_Int,      TB_NO_REVERSE },
    { X86::VFNMSUBPS4rr,          X86::VFNMSUBPS4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBPD4rr,          X86::VFNMSUBPD4rm,          TB_ALIGN_NONE },
    { X86::VFNMSUBPS4Yrr,         X86::VFNMSUBPS4Yrm,         TB_ALIGN_NONE },
    { X86::VFNMSUBPD4Yrr,         X86::VFNMSUBPD4Yrm,         TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4rr,        X86::VFMADDSUBPS4rm,        TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4rr,        X86::VFMADDSUBPD4rm,        TB_ALIGN_NONE },
    { X86::VFMADDSUBPS4Yrr,       X86::VFMADDSUBPS4Yrm,       TB_ALIGN_NONE },
    { X86::VFMADDSUBPD4Yrr,       X86::VFMADDSUBPD4Yrm,       TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4rr,        X86::VFMSUBADDPS4rm,        TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4rr,        X86::VFMSUBADDPD4rm,        TB_ALIGN_NONE },
    { X86::VFMSUBADDPS4Yrr,       X86::VFMSUBADDPS4Yrm,       TB_ALIGN_NONE },
    { X86::VFMSUBADDPD4Yrr,       X86::VFMSUBADDPD4Yrm,       TB_ALIGN_NONE },

    // XOP foldable instructions
    { X86::VPCMOVrrr,             X86::VPCMOVrrm,             0 },
    { X86::VPCMOVYrrr,            X86::VPCMOVYrrm,            0 },
    { X86::VPERMIL2PDrr,          X86::VPERMIL2PDrm,          0 },
    { X86::VPERMIL2PDYrr,         X86::VPERMIL2PDYrm,         0 },
    { X86::VPERMIL2PSrr,          X86::VPERMIL2PSrm,          0 },
    { X86::VPERMIL2PSYrr,         X86::VPERMIL2PSYrm,         0 },
    { X86::VPPERMrrr,             X86::VPPERMrrm,             0 },

    // AVX-512 instructions with 3 source operands.
    { X86::VPERMI2Brr,            X86::VPERMI2Brm,            0 },
    { X86::VPERMI2Drr,            X86::VPERMI2Drm,            0 },
    { X86::VPERMI2PSrr,           X86::VPERMI2PSrm,           0 },
    { X86::VPERMI2PDrr,           X86::VPERMI2PDrm,           0 },
    { X86::VPERMI2Qrr,            X86::VPERMI2Qrm,            0 },
    { X86::VPERMI2Wrr,            X86::VPERMI2Wrm,            0 },
    { X86::VPERMT2Brr,            X86::VPERMT2Brm,            0 },
    { X86::VPERMT2Drr,            X86::VPERMT2Drm,            0 },
    { X86::VPERMT2PSrr,           X86::VPERMT2PSrm,           0 },
    { X86::VPERMT2PDrr,           X86::VPERMT2PDrm,           0 },
    { X86::VPERMT2Qrr,            X86::VPERMT2Qrm,            0 },
    { X86::VPERMT2Wrr,            X86::VPERMT2Wrm,            0 },
    { X86::VPTERNLOGDZrri,        X86::VPTERNLOGDZrmi,        0 },
    { X86::VPTERNLOGQZrri,        X86::VPTERNLOGQZrmi,        0 },

    // AVX-512VL 256-bit instructions with 3 source operands.
    { X86::VPERMI2B256rr,         X86::VPERMI2B256rm,         0 },
    { X86::VPERMI2D256rr,         X86::VPERMI2D256rm,         0 },
    { X86::VPERMI2PD256rr,        X86::VPERMI2PD256rm,        0 },
    { X86::VPERMI2PS256rr,        X86::VPERMI2PS256rm,        0 },
    { X86::VPERMI2Q256rr,         X86::VPERMI2Q256rm,         0 },
    { X86::VPERMI2W256rr,         X86::VPERMI2W256rm,         0 },
    { X86::VPERMT2B256rr,         X86::VPERMT2B256rm,         0 },
    { X86::VPERMT2D256rr,         X86::VPERMT2D256rm,         0 },
    { X86::VPERMT2PD256rr,        X86::VPERMT2PD256rm,        0 },
    { X86::VPERMT2PS256rr,        X86::VPERMT2PS256rm,        0 },
    { X86::VPERMT2Q256rr,         X86::VPERMT2Q256rm,         0 },
    { X86::VPERMT2W256rr,         X86::VPERMT2W256rm,         0 },
    { X86::VPTERNLOGDZ256rri,     X86::VPTERNLOGDZ256rmi,     0 },
    { X86::VPTERNLOGQZ256rri,     X86::VPTERNLOGQZ256rmi,     0 },

    // AVX-512VL 128-bit instructions with 3 source operands.
    { X86::VPERMI2B128rr,         X86::VPERMI2B128rm,         0 },
    { X86::VPERMI2D128rr,         X86::VPERMI2D128rm,         0 },
    { X86::VPERMI2PD128rr,        X86::VPERMI2PD128rm,        0 },
    { X86::VPERMI2PS128rr,        X86::VPERMI2PS128rm,        0 },
    { X86::VPERMI2Q128rr,         X86::VPERMI2Q128rm,         0 },
    { X86::VPERMI2W128rr,         X86::VPERMI2W128rm,         0 },
    { X86::VPERMT2B128rr,         X86::VPERMT2B128rm,         0 },
    { X86::VPERMT2D128rr,         X86::VPERMT2D128rm,         0 },
    { X86::VPERMT2PD128rr,        X86::VPERMT2PD128rm,        0 },
    { X86::VPERMT2PS128rr,        X86::VPERMT2PS128rm,        0 },
    { X86::VPERMT2Q128rr,         X86::VPERMT2Q128rm,         0 },
    { X86::VPERMT2W128rr,         X86::VPERMT2W128rm,         0 },
    { X86::VPTERNLOGDZ128rri,     X86::VPTERNLOGDZ128rmi,     0 },
    { X86::VPTERNLOGQZ128rri,     X86::VPTERNLOGQZ128rmi,     0 },

    // AVX-512 masked instructions
    { X86::VADDPDZrrkz,           X86::VADDPDZrmkz,           0 },
    { X86::VADDPSZrrkz,           X86::VADDPSZrmkz,           0 },
    { X86::VADDSDZrr_Intkz,       X86::VADDSDZrm_Intkz,       TB_NO_REVERSE },
    { X86::VADDSSZrr_Intkz,       X86::VADDSSZrm_Intkz,       TB_NO_REVERSE },
    { X86::VALIGNDZrrikz,         X86::VALIGNDZrmikz,         0 },
    { X86::VALIGNQZrrikz,         X86::VALIGNQZrmikz,         0 },
    { X86::VANDNPDZrrkz,          X86::VANDNPDZrmkz,          0 },
    { X86::VANDNPSZrrkz,          X86::VANDNPSZrmkz,          0 },
    { X86::VANDPDZrrkz,           X86::VANDPDZrmkz,           0 },
    { X86::VANDPSZrrkz,           X86::VANDPSZrmkz,           0 },
    { X86::VDIVPDZrrkz,           X86::VDIVPDZrmkz,           0 },
    { X86::VDIVPSZrrkz,           X86::VDIVPSZrmkz,           0 },
    { X86::VDIVSDZrr_Intkz,       X86::VDIVSDZrm_Intkz,       TB_NO_REVERSE },
    { X86::VDIVSSZrr_Intkz,       X86::VDIVSSZrm_Intkz,       TB_NO_REVERSE },
    { X86::VINSERTF32x4Zrrkz,     X86::VINSERTF32x4Zrmkz,     0 },
    { X86::VINSERTF32x8Zrrkz,     X86::VINSERTF32x8Zrmkz,     0 },
    { X86::VINSERTF64x2Zrrkz,     X86::VINSERTF64x2Zrmkz,     0 },
    { X86::VINSERTF64x4Zrrkz,     X86::VINSERTF64x4Zrmkz,     0 },
    { X86::VINSERTI32x4Zrrkz,     X86::VINSERTI32x4Zrmkz,     0 },
    { X86::VINSERTI32x8Zrrkz,     X86::VINSERTI32x8Zrmkz,     0 },
    { X86::VINSERTI64x2Zrrkz,     X86::VINSERTI64x2Zrmkz,     0 },
    { X86::VINSERTI64x4Zrrkz,     X86::VINSERTI64x4Zrmkz,     0 },
    { X86::VMAXCPDZrrkz,          X86::VMAXCPDZrmkz,          0 },
    { X86::VMAXCPSZrrkz,          X86::VMAXCPSZrmkz,          0 },
    { X86::VMAXPDZrrkz,           X86::VMAXPDZrmkz,           0 },
    { X86::VMAXPSZrrkz,           X86::VMAXPSZrmkz,           0 },
    { X86::VMAXSDZrr_Intkz,       X86::VMAXSDZrm_Intkz,       0 },
    { X86::VMAXSSZrr_Intkz,       X86::VMAXSSZrm_Intkz,       0 },
    { X86::VMINCPDZrrkz,          X86::VMINCPDZrmkz,          0 },
    { X86::VMINCPSZrrkz,          X86::VMINCPSZrmkz,          0 },
    { X86::VMINPDZrrkz,           X86::VMINPDZrmkz,           0 },
    { X86::VMINPSZrrkz,           X86::VMINPSZrmkz,           0 },
    { X86::VMINSDZrr_Intkz,       X86::VMINSDZrm_Intkz,       0 },
    { X86::VMINSSZrr_Intkz,       X86::VMINSSZrm_Intkz,       0 },
    { X86::VMULPDZrrkz,           X86::VMULPDZrmkz,           0 },
    { X86::VMULPSZrrkz,           X86::VMULPSZrmkz,           0 },
    { X86::VMULSDZrr_Intkz,       X86::VMULSDZrm_Intkz,       TB_NO_REVERSE },
    { X86::VMULSSZrr_Intkz,       X86::VMULSSZrm_Intkz,       TB_NO_REVERSE },
    { X86::VORPDZrrkz,            X86::VORPDZrmkz,            0 },
    { X86::VORPSZrrkz,            X86::VORPSZrmkz,            0 },
    { X86::VPACKSSDWZrrkz,        X86::VPACKSSDWZrmkz,        0 },
    { X86::VPACKSSWBZrrkz,        X86::VPACKSSWBZrmkz,        0 },
    { X86::VPACKUSDWZrrkz,        X86::VPACKUSDWZrmkz,        0 },
    { X86::VPACKUSWBZrrkz,        X86::VPACKUSWBZrmkz,        0 },
    { X86::VPADDBZrrkz,           X86::VPADDBZrmkz,           0 },
    { X86::VPADDDZrrkz,           X86::VPADDDZrmkz,           0 },
    { X86::VPADDQZrrkz,           X86::VPADDQZrmkz,           0 },
    { X86::VPADDSBZrrkz,          X86::VPADDSBZrmkz,          0 },
    { X86::VPADDSWZrrkz,          X86::VPADDSWZrmkz,          0 },
    { X86::VPADDUSBZrrkz,         X86::VPADDUSBZrmkz,         0 },
    { X86::VPADDUSWZrrkz,         X86::VPADDUSWZrmkz,         0 },
    { X86::VPADDWZrrkz,           X86::VPADDWZrmkz,           0 },
    { X86::VPALIGNRZrrikz,        X86::VPALIGNRZrmikz,        0 },
    { X86::VPANDDZrrkz,           X86::VPANDDZrmkz,           0 },
    { X86::VPANDNDZrrkz,          X86::VPANDNDZrmkz,          0 },
    { X86::VPANDNQZrrkz,          X86::VPANDNQZrmkz,          0 },
    { X86::VPANDQZrrkz,           X86::VPANDQZrmkz,           0 },
    { X86::VPAVGBZrrkz,           X86::VPAVGBZrmkz,           0 },
    { X86::VPAVGWZrrkz,           X86::VPAVGWZrmkz,           0 },
    { X86::VPERMBZrrkz,           X86::VPERMBZrmkz,           0 },
    { X86::VPERMDZrrkz,           X86::VPERMDZrmkz,           0 },
    { X86::VPERMILPDZrrkz,        X86::VPERMILPDZrmkz,        0 },
    { X86::VPERMILPSZrrkz,        X86::VPERMILPSZrmkz,        0 },
    { X86::VPERMPDZrrkz,          X86::VPERMPDZrmkz,          0 },
    { X86::VPERMPSZrrkz,          X86::VPERMPSZrmkz,          0 },
    { X86::VPERMQZrrkz,           X86::VPERMQZrmkz,           0 },
    { X86::VPERMWZrrkz,           X86::VPERMWZrmkz,           0 },
    { X86::VPMADDUBSWZrrkz,       X86::VPMADDUBSWZrmkz,       0 },
    { X86::VPMADDWDZrrkz,         X86::VPMADDWDZrmkz,         0 },
    { X86::VPMAXSBZrrkz,          X86::VPMAXSBZrmkz,          0 },
    { X86::VPMAXSDZrrkz,          X86::VPMAXSDZrmkz,          0 },
    { X86::VPMAXSQZrrkz,          X86::VPMAXSQZrmkz,          0 },
    { X86::VPMAXSWZrrkz,          X86::VPMAXSWZrmkz,          0 },
    { X86::VPMAXUBZrrkz,          X86::VPMAXUBZrmkz,          0 },
    { X86::VPMAXUDZrrkz,          X86::VPMAXUDZrmkz,          0 },
    { X86::VPMAXUQZrrkz,          X86::VPMAXUQZrmkz,          0 },
    { X86::VPMAXUWZrrkz,          X86::VPMAXUWZrmkz,          0 },
    { X86::VPMINSBZrrkz,          X86::VPMINSBZrmkz,          0 },
    { X86::VPMINSDZrrkz,          X86::VPMINSDZrmkz,          0 },
    { X86::VPMINSQZrrkz,          X86::VPMINSQZrmkz,          0 },
    { X86::VPMINSWZrrkz,          X86::VPMINSWZrmkz,          0 },
    { X86::VPMINUBZrrkz,          X86::VPMINUBZrmkz,          0 },
    { X86::VPMINUDZrrkz,          X86::VPMINUDZrmkz,          0 },
    { X86::VPMINUQZrrkz,          X86::VPMINUQZrmkz,          0 },
    { X86::VPMINUWZrrkz,          X86::VPMINUWZrmkz,          0 },
    { X86::VPMULLDZrrkz,          X86::VPMULLDZrmkz,          0 },
    { X86::VPMULLQZrrkz,          X86::VPMULLQZrmkz,          0 },
    { X86::VPMULLWZrrkz,          X86::VPMULLWZrmkz,          0 },
    { X86::VPMULDQZrrkz,          X86::VPMULDQZrmkz,          0 },
    { X86::VPMULUDQZrrkz,         X86::VPMULUDQZrmkz,         0 },
    { X86::VPORDZrrkz,            X86::VPORDZrmkz,            0 },
    { X86::VPORQZrrkz,            X86::VPORQZrmkz,            0 },
    { X86::VPSHUFBZrrkz,          X86::VPSHUFBZrmkz,          0 },
    { X86::VPSLLDZrrkz,           X86::VPSLLDZrmkz,           0 },
    { X86::VPSLLQZrrkz,           X86::VPSLLQZrmkz,           0 },
    { X86::VPSLLVDZrrkz,          X86::VPSLLVDZrmkz,          0 },
    { X86::VPSLLVQZrrkz,          X86::VPSLLVQZrmkz,          0 },
    { X86::VPSLLVWZrrkz,          X86::VPSLLVWZrmkz,          0 },
    { X86::VPSLLWZrrkz,           X86::VPSLLWZrmkz,           0 },
    { X86::VPSRADZrrkz,           X86::VPSRADZrmkz,           0 },
    { X86::VPSRAQZrrkz,           X86::VPSRAQZrmkz,           0 },
    { X86::VPSRAVDZrrkz,          X86::VPSRAVDZrmkz,          0 },
    { X86::VPSRAVQZrrkz,          X86::VPSRAVQZrmkz,          0 },
    { X86::VPSRAVWZrrkz,          X86::VPSRAVWZrmkz,          0 },
    { X86::VPSRAWZrrkz,           X86::VPSRAWZrmkz,           0 },
    { X86::VPSRLDZrrkz,           X86::VPSRLDZrmkz,           0 },
    { X86::VPSRLQZrrkz,           X86::VPSRLQZrmkz,           0 },
    { X86::VPSRLVDZrrkz,          X86::VPSRLVDZrmkz,          0 },
    { X86::VPSRLVQZrrkz,          X86::VPSRLVQZrmkz,          0 },
    { X86::VPSRLVWZrrkz,          X86::VPSRLVWZrmkz,          0 },
    { X86::VPSRLWZrrkz,           X86::VPSRLWZrmkz,           0 },
    { X86::VPSUBBZrrkz,           X86::VPSUBBZrmkz,           0 },
    { X86::VPSUBDZrrkz,           X86::VPSUBDZrmkz,           0 },
    { X86::VPSUBQZrrkz,           X86::VPSUBQZrmkz,           0 },
    { X86::VPSUBSBZrrkz,          X86::VPSUBSBZrmkz,          0 },
    { X86::VPSUBSWZrrkz,          X86::VPSUBSWZrmkz,          0 },
    { X86::VPSUBUSBZrrkz,         X86::VPSUBUSBZrmkz,         0 },
    { X86::VPSUBUSWZrrkz,         X86::VPSUBUSWZrmkz,         0 },
    { X86::VPSUBWZrrkz,           X86::VPSUBWZrmkz,           0 },
    { X86::VPUNPCKHBWZrrkz,       X86::VPUNPCKHBWZrmkz,       0 },
    { X86::VPUNPCKHDQZrrkz,       X86::VPUNPCKHDQZrmkz,       0 },
    { X86::VPUNPCKHQDQZrrkz,      X86::VPUNPCKHQDQZrmkz,      0 },
    { X86::VPUNPCKHWDZrrkz,       X86::VPUNPCKHWDZrmkz,       0 },
    { X86::VPUNPCKLBWZrrkz,       X86::VPUNPCKLBWZrmkz,       0 },
    { X86::VPUNPCKLDQZrrkz,       X86::VPUNPCKLDQZrmkz,       0 },
    { X86::VPUNPCKLQDQZrrkz,      X86::VPUNPCKLQDQZrmkz,      0 },
    { X86::VPUNPCKLWDZrrkz,       X86::VPUNPCKLWDZrmkz,       0 },
    { X86::VPXORDZrrkz,           X86::VPXORDZrmkz,           0 },
    { X86::VPXORQZrrkz,           X86::VPXORQZrmkz,           0 },
    { X86::VSHUFPDZrrikz,         X86::VSHUFPDZrmikz,         0 },
    { X86::VSHUFPSZrrikz,         X86::VSHUFPSZrmikz,         0 },
    { X86::VSUBPDZrrkz,           X86::VSUBPDZrmkz,           0 },
    { X86::VSUBPSZrrkz,           X86::VSUBPSZrmkz,           0 },
    { X86::VSUBSDZrr_Intkz,       X86::VSUBSDZrm_Intkz,       TB_NO_REVERSE },
    { X86::VSUBSSZrr_Intkz,       X86::VSUBSSZrm_Intkz,       TB_NO_REVERSE },
    { X86::VUNPCKHPDZrrkz,        X86::VUNPCKHPDZrmkz,        0 },
    { X86::VUNPCKHPSZrrkz,        X86::VUNPCKHPSZrmkz,        0 },
    { X86::VUNPCKLPDZrrkz,        X86::VUNPCKLPDZrmkz,        0 },
    { X86::VUNPCKLPSZrrkz,        X86::VUNPCKLPSZrmkz,        0 },
    { X86::VXORPDZrrkz,           X86::VXORPDZrmkz,           0 },
    { X86::VXORPSZrrkz,           X86::VXORPSZrmkz,           0 },

    // AVX-512{F,VL} masked arithmetic instructions 256-bit
    { X86::VADDPDZ256rrkz,        X86::VADDPDZ256rmkz,        0 },
    { X86::VADDPSZ256rrkz,        X86::VADDPSZ256rmkz,        0 },
    { X86::VALIGNDZ256rrikz,      X86::VALIGNDZ256rmikz,      0 },
    { X86::VALIGNQZ256rrikz,      X86::VALIGNQZ256rmikz,      0 },
    { X86::VANDNPDZ256rrkz,       X86::VANDNPDZ256rmkz,       0 },
    { X86::VANDNPSZ256rrkz,       X86::VANDNPSZ256rmkz,       0 },
    { X86::VANDPDZ256rrkz,        X86::VANDPDZ256rmkz,        0 },
    { X86::VANDPSZ256rrkz,        X86::VANDPSZ256rmkz,        0 },
    { X86::VDIVPDZ256rrkz,        X86::VDIVPDZ256rmkz,        0 },
    { X86::VDIVPSZ256rrkz,        X86::VDIVPSZ256rmkz,        0 },
    { X86::VINSERTF32x4Z256rrkz,  X86::VINSERTF32x4Z256rmkz,  0 },
    { X86::VINSERTF64x2Z256rrkz,  X86::VINSERTF64x2Z256rmkz,  0 },
    { X86::VINSERTI32x4Z256rrkz,  X86::VINSERTI32x4Z256rmkz,  0 },
    { X86::VINSERTI64x2Z256rrkz,  X86::VINSERTI64x2Z256rmkz,  0 },
    { X86::VMAXCPDZ256rrkz,       X86::VMAXCPDZ256rmkz,       0 },
    { X86::VMAXCPSZ256rrkz,       X86::VMAXCPSZ256rmkz,       0 },
    { X86::VMAXPDZ256rrkz,        X86::VMAXPDZ256rmkz,        0 },
    { X86::VMAXPSZ256rrkz,        X86::VMAXPSZ256rmkz,        0 },
    { X86::VMINCPDZ256rrkz,       X86::VMINCPDZ256rmkz,       0 },
    { X86::VMINCPSZ256rrkz,       X86::VMINCPSZ256rmkz,       0 },
    { X86::VMINPDZ256rrkz,        X86::VMINPDZ256rmkz,        0 },
    { X86::VMINPSZ256rrkz,        X86::VMINPSZ256rmkz,        0 },
    { X86::VMULPDZ256rrkz,        X86::VMULPDZ256rmkz,        0 },
    { X86::VMULPSZ256rrkz,        X86::VMULPSZ256rmkz,        0 },
    { X86::VORPDZ256rrkz,         X86::VORPDZ256rmkz,         0 },
    { X86::VORPSZ256rrkz,         X86::VORPSZ256rmkz,         0 },
    { X86::VPACKSSDWZ256rrkz,     X86::VPACKSSDWZ256rmkz,     0 },
    { X86::VPACKSSWBZ256rrkz,     X86::VPACKSSWBZ256rmkz,     0 },
    { X86::VPACKUSDWZ256rrkz,     X86::VPACKUSDWZ256rmkz,     0 },
    { X86::VPACKUSWBZ256rrkz,     X86::VPACKUSWBZ256rmkz,     0 },
    { X86::VPADDBZ256rrkz,        X86::VPADDBZ256rmkz,        0 },
    { X86::VPADDDZ256rrkz,        X86::VPADDDZ256rmkz,        0 },
    { X86::VPADDQZ256rrkz,        X86::VPADDQZ256rmkz,        0 },
    { X86::VPADDSBZ256rrkz,       X86::VPADDSBZ256rmkz,       0 },
    { X86::VPADDSWZ256rrkz,       X86::VPADDSWZ256rmkz,       0 },
    { X86::VPADDUSBZ256rrkz,      X86::VPADDUSBZ256rmkz,      0 },
    { X86::VPADDUSWZ256rrkz,      X86::VPADDUSWZ256rmkz,      0 },
    { X86::VPADDWZ256rrkz,        X86::VPADDWZ256rmkz,        0 },
    { X86::VPALIGNRZ256rrikz,     X86::VPALIGNRZ256rmikz,     0 },
    { X86::VPANDDZ256rrkz,        X86::VPANDDZ256rmkz,        0 },
    { X86::VPANDNDZ256rrkz,       X86::VPANDNDZ256rmkz,       0 },
    { X86::VPANDNQZ256rrkz,       X86::VPANDNQZ256rmkz,       0 },
    { X86::VPANDQZ256rrkz,        X86::VPANDQZ256rmkz,        0 },
    { X86::VPAVGBZ256rrkz,        X86::VPAVGBZ256rmkz,        0 },
    { X86::VPAVGWZ256rrkz,        X86::VPAVGWZ256rmkz,        0 },
    { X86::VPERMBZ256rrkz,        X86::VPERMBZ256rmkz,        0 },
    { X86::VPERMDZ256rrkz,        X86::VPERMDZ256rmkz,        0 },
    { X86::VPERMILPDZ256rrkz,     X86::VPERMILPDZ256rmkz,     0 },
    { X86::VPERMILPSZ256rrkz,     X86::VPERMILPSZ256rmkz,     0 },
    { X86::VPERMPDZ256rrkz,       X86::VPERMPDZ256rmkz,       0 },
    { X86::VPERMPSZ256rrkz,       X86::VPERMPSZ256rmkz,       0 },
    { X86::VPERMQZ256rrkz,        X86::VPERMQZ256rmkz,        0 },
    { X86::VPERMWZ256rrkz,        X86::VPERMWZ256rmkz,        0 },
    { X86::VPMADDUBSWZ256rrkz,    X86::VPMADDUBSWZ256rmkz,    0 },
    { X86::VPMADDWDZ256rrkz,      X86::VPMADDWDZ256rmkz,      0 },
    { X86::VPMAXSBZ256rrkz,       X86::VPMAXSBZ256rmkz,       0 },
    { X86::VPMAXSDZ256rrkz,       X86::VPMAXSDZ256rmkz,       0 },
    { X86::VPMAXSQZ256rrkz,       X86::VPMAXSQZ256rmkz,       0 },
    { X86::VPMAXSWZ256rrkz,       X86::VPMAXSWZ256rmkz,       0 },
    { X86::VPMAXUBZ256rrkz,       X86::VPMAXUBZ256rmkz,       0 },
    { X86::VPMAXUDZ256rrkz,       X86::VPMAXUDZ256rmkz,       0 },
    { X86::VPMAXUQZ256rrkz,       X86::VPMAXUQZ256rmkz,       0 },
    { X86::VPMAXUWZ256rrkz,       X86::VPMAXUWZ256rmkz,       0 },
    { X86::VPMINSBZ256rrkz,       X86::VPMINSBZ256rmkz,       0 },
    { X86::VPMINSDZ256rrkz,       X86::VPMINSDZ256rmkz,       0 },
    { X86::VPMINSQZ256rrkz,       X86::VPMINSQZ256rmkz,       0 },
    { X86::VPMINSWZ256rrkz,       X86::VPMINSWZ256rmkz,       0 },
    { X86::VPMINUBZ256rrkz,       X86::VPMINUBZ256rmkz,       0 },
    { X86::VPMINUDZ256rrkz,       X86::VPMINUDZ256rmkz,       0 },
    { X86::VPMINUQZ256rrkz,       X86::VPMINUQZ256rmkz,       0 },
    { X86::VPMINUWZ256rrkz,       X86::VPMINUWZ256rmkz,       0 },
    { X86::VPMULDQZ256rrkz,       X86::VPMULDQZ256rmkz,       0 },
    { X86::VPMULLDZ256rrkz,       X86::VPMULLDZ256rmkz,       0 },
    { X86::VPMULLQZ256rrkz,       X86::VPMULLQZ256rmkz,       0 },
    { X86::VPMULLWZ256rrkz,       X86::VPMULLWZ256rmkz,       0 },
    { X86::VPMULUDQZ256rrkz,      X86::VPMULUDQZ256rmkz,      0 },
    { X86::VPORDZ256rrkz,         X86::VPORDZ256rmkz,         0 },
    { X86::VPORQZ256rrkz,         X86::VPORQZ256rmkz,         0 },
    { X86::VPSHUFBZ256rrkz,       X86::VPSHUFBZ256rmkz,       0 },
    { X86::VPSLLDZ256rrkz,        X86::VPSLLDZ256rmkz,        0 },
    { X86::VPSLLQZ256rrkz,        X86::VPSLLQZ256rmkz,        0 },
    { X86::VPSLLVDZ256rrkz,       X86::VPSLLVDZ256rmkz,       0 },
    { X86::VPSLLVQZ256rrkz,       X86::VPSLLVQZ256rmkz,       0 },
    { X86::VPSLLVWZ256rrkz,       X86::VPSLLVWZ256rmkz,       0 },
    { X86::VPSLLWZ256rrkz,        X86::VPSLLWZ256rmkz,        0 },
    { X86::VPSRADZ256rrkz,        X86::VPSRADZ256rmkz,        0 },
    { X86::VPSRAQZ256rrkz,        X86::VPSRAQZ256rmkz,        0 },
    { X86::VPSRAVDZ256rrkz,       X86::VPSRAVDZ256rmkz,       0 },
    { X86::VPSRAVQZ256rrkz,       X86::VPSRAVQZ256rmkz,       0 },
    { X86::VPSRAVWZ256rrkz,       X86::VPSRAVWZ256rmkz,       0 },
    { X86::VPSRAWZ256rrkz,        X86::VPSRAWZ256rmkz,        0 },
    { X86::VPSRLDZ256rrkz,        X86::VPSRLDZ256rmkz,        0 },
    { X86::VPSRLQZ256rrkz,        X86::VPSRLQZ256rmkz,        0 },
    { X86::VPSRLVDZ256rrkz,       X86::VPSRLVDZ256rmkz,       0 },
    { X86::VPSRLVQZ256rrkz,       X86::VPSRLVQZ256rmkz,       0 },
    { X86::VPSRLVWZ256rrkz,       X86::VPSRLVWZ256rmkz,       0 },
    { X86::VPSRLWZ256rrkz,        X86::VPSRLWZ256rmkz,        0 },
    { X86::VPSUBBZ256rrkz,        X86::VPSUBBZ256rmkz,        0 },
    { X86::VPSUBDZ256rrkz,        X86::VPSUBDZ256rmkz,        0 },
    { X86::VPSUBQZ256rrkz,        X86::VPSUBQZ256rmkz,        0 },
    { X86::VPSUBSBZ256rrkz,       X86::VPSUBSBZ256rmkz,       0 },
    { X86::VPSUBSWZ256rrkz,       X86::VPSUBSWZ256rmkz,       0 },
    { X86::VPSUBUSBZ256rrkz,      X86::VPSUBUSBZ256rmkz,      0 },
    { X86::VPSUBUSWZ256rrkz,      X86::VPSUBUSWZ256rmkz,      0 },
    { X86::VPSUBWZ256rrkz,        X86::VPSUBWZ256rmkz,        0 },
    { X86::VPUNPCKHBWZ256rrkz,    X86::VPUNPCKHBWZ256rmkz,    0 },
    { X86::VPUNPCKHDQZ256rrkz,    X86::VPUNPCKHDQZ256rmkz,    0 },
    { X86::VPUNPCKHQDQZ256rrkz,   X86::VPUNPCKHQDQZ256rmkz,   0 },
    { X86::VPUNPCKHWDZ256rrkz,    X86::VPUNPCKHWDZ256rmkz,    0 },
    { X86::VPUNPCKLBWZ256rrkz,    X86::VPUNPCKLBWZ256rmkz,    0 },
    { X86::VPUNPCKLDQZ256rrkz,    X86::VPUNPCKLDQZ256rmkz,    0 },
    { X86::VPUNPCKLQDQZ256rrkz,   X86::VPUNPCKLQDQZ256rmkz,   0 },
    { X86::VPUNPCKLWDZ256rrkz,    X86::VPUNPCKLWDZ256rmkz,    0 },
    { X86::VPXORDZ256rrkz,        X86::VPXORDZ256rmkz,        0 },
    { X86::VPXORQZ256rrkz,        X86::VPXORQZ256rmkz,        0 },
    { X86::VSHUFPDZ256rrikz,      X86::VSHUFPDZ256rmikz,      0 },
    { X86::VSHUFPSZ256rrikz,      X86::VSHUFPSZ256rmikz,      0 },
    { X86::VSUBPDZ256rrkz,        X86::VSUBPDZ256rmkz,        0 },
    { X86::VSUBPSZ256rrkz,        X86::VSUBPSZ256rmkz,        0 },
    { X86::VUNPCKHPDZ256rrkz,     X86::VUNPCKHPDZ256rmkz,     0 },
    { X86::VUNPCKHPSZ256rrkz,     X86::VUNPCKHPSZ256rmkz,     0 },
    { X86::VUNPCKLPDZ256rrkz,     X86::VUNPCKLPDZ256rmkz,     0 },
    { X86::VUNPCKLPSZ256rrkz,     X86::VUNPCKLPSZ256rmkz,     0 },
    { X86::VXORPDZ256rrkz,        X86::VXORPDZ256rmkz,        0 },
    { X86::VXORPSZ256rrkz,        X86::VXORPSZ256rmkz,        0 },

    // AVX-512{F,VL} masked arithmetic instructions 128-bit
    { X86::VADDPDZ128rrkz,        X86::VADDPDZ128rmkz,        0 },
    { X86::VADDPSZ128rrkz,        X86::VADDPSZ128rmkz,        0 },
    { X86::VALIGNDZ128rrikz,      X86::VALIGNDZ128rmikz,      0 },
    { X86::VALIGNQZ128rrikz,      X86::VALIGNQZ128rmikz,      0 },
    { X86::VANDNPDZ128rrkz,       X86::VANDNPDZ128rmkz,       0 },
    { X86::VANDNPSZ128rrkz,       X86::VANDNPSZ128rmkz,       0 },
    { X86::VANDPDZ128rrkz,        X86::VANDPDZ128rmkz,        0 },
    { X86::VANDPSZ128rrkz,        X86::VANDPSZ128rmkz,        0 },
    { X86::VDIVPDZ128rrkz,        X86::VDIVPDZ128rmkz,        0 },
    { X86::VDIVPSZ128rrkz,        X86::VDIVPSZ128rmkz,        0 },
    { X86::VMAXCPDZ128rrkz,       X86::VMAXCPDZ128rmkz,       0 },
    { X86::VMAXCPSZ128rrkz,       X86::VMAXCPSZ128rmkz,       0 },
    { X86::VMAXPDZ128rrkz,        X86::VMAXPDZ128rmkz,        0 },
    { X86::VMAXPSZ128rrkz,        X86::VMAXPSZ128rmkz,        0 },
    { X86::VMINCPDZ128rrkz,       X86::VMINCPDZ128rmkz,       0 },
    { X86::VMINCPSZ128rrkz,       X86::VMINCPSZ128rmkz,       0 },
    { X86::VMINPDZ128rrkz,        X86::VMINPDZ128rmkz,        0 },
    { X86::VMINPSZ128rrkz,        X86::VMINPSZ128rmkz,        0 },
    { X86::VMULPDZ128rrkz,        X86::VMULPDZ128rmkz,        0 },
    { X86::VMULPSZ128rrkz,        X86::VMULPSZ128rmkz,        0 },
    { X86::VORPDZ128rrkz,         X86::VORPDZ128rmkz,         0 },
    { X86::VORPSZ128rrkz,         X86::VORPSZ128rmkz,         0 },
    { X86::VPACKSSDWZ128rrkz,     X86::VPACKSSDWZ128rmkz,     0 },
    { X86::VPACKSSWBZ128rrkz,     X86::VPACKSSWBZ128rmkz,     0 },
    { X86::VPACKUSDWZ128rrkz,     X86::VPACKUSDWZ128rmkz,     0 },
    { X86::VPACKUSWBZ128rrkz,     X86::VPACKUSWBZ128rmkz,     0 },
    { X86::VPADDBZ128rrkz,        X86::VPADDBZ128rmkz,        0 },
    { X86::VPADDDZ128rrkz,        X86::VPADDDZ128rmkz,        0 },
    { X86::VPADDQZ128rrkz,        X86::VPADDQZ128rmkz,        0 },
    { X86::VPADDSBZ128rrkz,       X86::VPADDSBZ128rmkz,       0 },
    { X86::VPADDSWZ128rrkz,       X86::VPADDSWZ128rmkz,       0 },
    { X86::VPADDUSBZ128rrkz,      X86::VPADDUSBZ128rmkz,      0 },
    { X86::VPADDUSWZ128rrkz,      X86::VPADDUSWZ128rmkz,      0 },
    { X86::VPADDWZ128rrkz,        X86::VPADDWZ128rmkz,        0 },
    { X86::VPALIGNRZ128rrikz,     X86::VPALIGNRZ128rmikz,     0 },
    { X86::VPANDDZ128rrkz,        X86::VPANDDZ128rmkz,        0 },
    { X86::VPANDNDZ128rrkz,       X86::VPANDNDZ128rmkz,       0 },
    { X86::VPANDNQZ128rrkz,       X86::VPANDNQZ128rmkz,       0 },
    { X86::VPANDQZ128rrkz,        X86::VPANDQZ128rmkz,        0 },
    { X86::VPAVGBZ128rrkz,        X86::VPAVGBZ128rmkz,        0 },
    { X86::VPAVGWZ128rrkz,        X86::VPAVGWZ128rmkz,        0 },
    { X86::VPERMBZ128rrkz,        X86::VPERMBZ128rmkz,        0 },
    { X86::VPERMILPDZ128rrkz,     X86::VPERMILPDZ128rmkz,     0 },
    { X86::VPERMILPSZ128rrkz,     X86::VPERMILPSZ128rmkz,     0 },
    { X86::VPERMWZ128rrkz,        X86::VPERMWZ128rmkz,        0 },
    { X86::VPMADDUBSWZ128rrkz,    X86::VPMADDUBSWZ128rmkz,    0 },
    { X86::VPMADDWDZ128rrkz,      X86::VPMADDWDZ128rmkz,      0 },
    { X86::VPMAXSBZ128rrkz,       X86::VPMAXSBZ128rmkz,       0 },
    { X86::VPMAXSDZ128rrkz,       X86::VPMAXSDZ128rmkz,       0 },
    { X86::VPMAXSQZ128rrkz,       X86::VPMAXSQZ128rmkz,       0 },
    { X86::VPMAXSWZ128rrkz,       X86::VPMAXSWZ128rmkz,       0 },
    { X86::VPMAXUBZ128rrkz,       X86::VPMAXUBZ128rmkz,       0 },
    { X86::VPMAXUDZ128rrkz,       X86::VPMAXUDZ128rmkz,       0 },
    { X86::VPMAXUQZ128rrkz,       X86::VPMAXUQZ128rmkz,       0 },
    { X86::VPMAXUWZ128rrkz,       X86::VPMAXUWZ128rmkz,       0 },
    { X86::VPMINSBZ128rrkz,       X86::VPMINSBZ128rmkz,       0 },
    { X86::VPMINSDZ128rrkz,       X86::VPMINSDZ128rmkz,       0 },
    { X86::VPMINSQZ128rrkz,       X86::VPMINSQZ128rmkz,       0 },
    { X86::VPMINSWZ128rrkz,       X86::VPMINSWZ128rmkz,       0 },
    { X86::VPMINUBZ128rrkz,       X86::VPMINUBZ128rmkz,       0 },
    { X86::VPMINUDZ128rrkz,       X86::VPMINUDZ128rmkz,       0 },
    { X86::VPMINUQZ128rrkz,       X86::VPMINUQZ128rmkz,       0 },
    { X86::VPMINUWZ128rrkz,       X86::VPMINUWZ128rmkz,       0 },
    { X86::VPMULDQZ128rrkz,       X86::VPMULDQZ128rmkz,       0 },
    { X86::VPMULLDZ128rrkz,       X86::VPMULLDZ128rmkz,       0 },
    { X86::VPMULLQZ128rrkz,       X86::VPMULLQZ128rmkz,       0 },
    { X86::VPMULLWZ128rrkz,       X86::VPMULLWZ128rmkz,       0 },
    { X86::VPMULUDQZ128rrkz,      X86::VPMULUDQZ128rmkz,      0 },
    { X86::VPORDZ128rrkz,         X86::VPORDZ128rmkz,         0 },
    { X86::VPORQZ128rrkz,         X86::VPORQZ128rmkz,         0 },
    { X86::VPSHUFBZ128rrkz,       X86::VPSHUFBZ128rmkz,       0 },
    { X86::VPSLLDZ128rrkz,        X86::VPSLLDZ128rmkz,        0 },
    { X86::VPSLLQZ128rrkz,        X86::VPSLLQZ128rmkz,        0 },
    { X86::VPSLLVDZ128rrkz,       X86::VPSLLVDZ128rmkz,       0 },
    { X86::VPSLLVQZ128rrkz,       X86::VPSLLVQZ128rmkz,       0 },
    { X86::VPSLLVWZ128rrkz,       X86::VPSLLVWZ128rmkz,       0 },
    { X86::VPSLLWZ128rrkz,        X86::VPSLLWZ128rmkz,        0 },
    { X86::VPSRADZ128rrkz,        X86::VPSRADZ128rmkz,        0 },
    { X86::VPSRAQZ128rrkz,        X86::VPSRAQZ128rmkz,        0 },
    { X86::VPSRAVDZ128rrkz,       X86::VPSRAVDZ128rmkz,       0 },
    { X86::VPSRAVQZ128rrkz,       X86::VPSRAVQZ128rmkz,       0 },
    { X86::VPSRAVWZ128rrkz,       X86::VPSRAVWZ128rmkz,       0 },
    { X86::VPSRAWZ128rrkz,        X86::VPSRAWZ128rmkz,        0 },
    { X86::VPSRLDZ128rrkz,        X86::VPSRLDZ128rmkz,        0 },
    { X86::VPSRLQZ128rrkz,        X86::VPSRLQZ128rmkz,        0 },
    { X86::VPSRLVDZ128rrkz,       X86::VPSRLVDZ128rmkz,       0 },
    { X86::VPSRLVQZ128rrkz,       X86::VPSRLVQZ128rmkz,       0 },
    { X86::VPSRLVWZ128rrkz,       X86::VPSRLVWZ128rmkz,       0 },
    { X86::VPSRLWZ128rrkz,        X86::VPSRLWZ128rmkz,        0 },
    { X86::VPSUBBZ128rrkz,        X86::VPSUBBZ128rmkz,        0 },
    { X86::VPSUBDZ128rrkz,        X86::VPSUBDZ128rmkz,        0 },
    { X86::VPSUBQZ128rrkz,        X86::VPSUBQZ128rmkz,        0 },
    { X86::VPSUBSBZ128rrkz,       X86::VPSUBSBZ128rmkz,       0 },
    { X86::VPSUBSWZ128rrkz,       X86::VPSUBSWZ128rmkz,       0 },
    { X86::VPSUBUSBZ128rrkz,      X86::VPSUBUSBZ128rmkz,      0 },
    { X86::VPSUBUSWZ128rrkz,      X86::VPSUBUSWZ128rmkz,      0 },
    { X86::VPSUBWZ128rrkz,        X86::VPSUBWZ128rmkz,        0 },
    { X86::VPUNPCKHBWZ128rrkz,    X86::VPUNPCKHBWZ128rmkz,    0 },
    { X86::VPUNPCKHDQZ128rrkz,    X86::VPUNPCKHDQZ128rmkz,    0 },
    { X86::VPUNPCKHQDQZ128rrkz,   X86::VPUNPCKHQDQZ128rmkz,   0 },
    { X86::VPUNPCKHWDZ128rrkz,    X86::VPUNPCKHWDZ128rmkz,    0 },
    { X86::VPUNPCKLBWZ128rrkz,    X86::VPUNPCKLBWZ128rmkz,    0 },
    { X86::VPUNPCKLDQZ128rrkz,    X86::VPUNPCKLDQZ128rmkz,    0 },
    { X86::VPUNPCKLQDQZ128rrkz,   X86::VPUNPCKLQDQZ128rmkz,   0 },
    { X86::VPUNPCKLWDZ128rrkz,    X86::VPUNPCKLWDZ128rmkz,    0 },
    { X86::VPXORDZ128rrkz,        X86::VPXORDZ128rmkz,        0 },
    { X86::VPXORQZ128rrkz,        X86::VPXORQZ128rmkz,        0 },
    { X86::VSHUFPDZ128rrikz,      X86::VSHUFPDZ128rmikz,      0 },
    { X86::VSHUFPSZ128rrikz,      X86::VSHUFPSZ128rmikz,      0 },
    { X86::VSUBPDZ128rrkz,        X86::VSUBPDZ128rmkz,        0 },
    { X86::VSUBPSZ128rrkz,        X86::VSUBPSZ128rmkz,        0 },
    { X86::VUNPCKHPDZ128rrkz,     X86::VUNPCKHPDZ128rmkz,     0 },
    { X86::VUNPCKHPSZ128rrkz,     X86::VUNPCKHPSZ128rmkz,     0 },
    { X86::VUNPCKLPDZ128rrkz,     X86::VUNPCKLPDZ128rmkz,     0 },
    { X86::VUNPCKLPSZ128rrkz,     X86::VUNPCKLPSZ128rmkz,     0 },
    { X86::VXORPDZ128rrkz,        X86::VXORPDZ128rmkz,        0 },
    { X86::VXORPSZ128rrkz,        X86::VXORPSZ128rmkz,        0 },

    // AVX-512 masked foldable instructions
    { X86::VBROADCASTSSZrk,       X86::VBROADCASTSSZmk,       TB_NO_REVERSE },
    { X86::VBROADCASTSDZrk,       X86::VBROADCASTSDZmk,       TB_NO_REVERSE },
    { X86::VPABSBZrrk,            X86::VPABSBZrmk,            0 },
    { X86::VPABSDZrrk,            X86::VPABSDZrmk,            0 },
    { X86::VPABSQZrrk,            X86::VPABSQZrmk,            0 },
    { X86::VPABSWZrrk,            X86::VPABSWZrmk,            0 },
    { X86::VPERMILPDZrik,         X86::VPERMILPDZmik,         0 },
    { X86::VPERMILPSZrik,         X86::VPERMILPSZmik,         0 },
    { X86::VPERMPDZrik,           X86::VPERMPDZmik,           0 },
    { X86::VPERMQZrik,            X86::VPERMQZmik,            0 },
    { X86::VPMOVSXBDZrrk,         X86::VPMOVSXBDZrmk,         0 },
    { X86::VPMOVSXBQZrrk,         X86::VPMOVSXBQZrmk,         TB_NO_REVERSE },
    { X86::VPMOVSXBWZrrk,         X86::VPMOVSXBWZrmk,         0 },
    { X86::VPMOVSXDQZrrk,         X86::VPMOVSXDQZrmk,         0 },
    { X86::VPMOVSXWDZrrk,         X86::VPMOVSXWDZrmk,         0 },
    { X86::VPMOVSXWQZrrk,         X86::VPMOVSXWQZrmk,         0 },
    { X86::VPMOVZXBDZrrk,         X86::VPMOVZXBDZrmk,         0 },
    { X86::VPMOVZXBQZrrk,         X86::VPMOVZXBQZrmk,         TB_NO_REVERSE },
    { X86::VPMOVZXBWZrrk,         X86::VPMOVZXBWZrmk,         0 },
    { X86::VPMOVZXDQZrrk,         X86::VPMOVZXDQZrmk,         0 },
    { X86::VPMOVZXWDZrrk,         X86::VPMOVZXWDZrmk,         0 },
    { X86::VPMOVZXWQZrrk,         X86::VPMOVZXWQZrmk,         0 },
    { X86::VPSHUFDZrik,           X86::VPSHUFDZmik,           0 },
    { X86::VPSHUFHWZrik,          X86::VPSHUFHWZmik,          0 },
    { X86::VPSHUFLWZrik,          X86::VPSHUFLWZmik,          0 },
    { X86::VPSLLDZrik,            X86::VPSLLDZmik,            0 },
    { X86::VPSLLQZrik,            X86::VPSLLQZmik,            0 },
    { X86::VPSLLWZrik,            X86::VPSLLWZmik,            0 },
    { X86::VPSRADZrik,            X86::VPSRADZmik,            0 },
    { X86::VPSRAQZrik,            X86::VPSRAQZmik,            0 },
    { X86::VPSRAWZrik,            X86::VPSRAWZmik,            0 },
    { X86::VPSRLDZrik,            X86::VPSRLDZmik,            0 },
    { X86::VPSRLQZrik,            X86::VPSRLQZmik,            0 },
    { X86::VPSRLWZrik,            X86::VPSRLWZmik,            0 },

    // AVX-512VL 256-bit masked foldable instructions
    { X86::VBROADCASTSSZ256rk,    X86::VBROADCASTSSZ256mk,    TB_NO_REVERSE },
    { X86::VBROADCASTSDZ256rk,    X86::VBROADCASTSDZ256mk,    TB_NO_REVERSE },
    { X86::VPABSBZ256rrk,         X86::VPABSBZ256rmk,         0 },
    { X86::VPABSDZ256rrk,         X86::VPABSDZ256rmk,         0 },
    { X86::VPABSQZ256rrk,         X86::VPABSQZ256rmk,         0 },
    { X86::VPABSWZ256rrk,         X86::VPABSWZ256rmk,         0 },
    { X86::VPERMILPDZ256rik,      X86::VPERMILPDZ256mik,      0 },
    { X86::VPERMILPSZ256rik,      X86::VPERMILPSZ256mik,      0 },
    { X86::VPERMPDZ256rik,        X86::VPERMPDZ256mik,        0 },
    { X86::VPERMQZ256rik,         X86::VPERMQZ256mik,         0 },
    { X86::VPMOVSXBDZ256rrk,      X86::VPMOVSXBDZ256rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXBQZ256rrk,      X86::VPMOVSXBQZ256rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXBWZ256rrk,      X86::VPMOVSXBWZ256rmk,      0 },
    { X86::VPMOVSXDQZ256rrk,      X86::VPMOVSXDQZ256rmk,      0 },
    { X86::VPMOVSXWDZ256rrk,      X86::VPMOVSXWDZ256rmk,      0 },
    { X86::VPMOVSXWQZ256rrk,      X86::VPMOVSXWQZ256rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXBDZ256rrk,      X86::VPMOVZXBDZ256rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXBQZ256rrk,      X86::VPMOVZXBQZ256rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXBWZ256rrk,      X86::VPMOVZXBWZ256rmk,      0 },
    { X86::VPMOVZXDQZ256rrk,      X86::VPMOVZXDQZ256rmk,      0 },
    { X86::VPMOVZXWDZ256rrk,      X86::VPMOVZXWDZ256rmk,      0 },
    { X86::VPMOVZXWQZ256rrk,      X86::VPMOVZXWQZ256rmk,      TB_NO_REVERSE },
    { X86::VPSHUFDZ256rik,        X86::VPSHUFDZ256mik,        0 },
    { X86::VPSHUFHWZ256rik,       X86::VPSHUFHWZ256mik,       0 },
    { X86::VPSHUFLWZ256rik,       X86::VPSHUFLWZ256mik,       0 },
    { X86::VPSLLDZ256rik,         X86::VPSLLDZ256mik,         0 },
    { X86::VPSLLQZ256rik,         X86::VPSLLQZ256mik,         0 },
    { X86::VPSLLWZ256rik,         X86::VPSLLWZ256mik,         0 },
    { X86::VPSRADZ256rik,         X86::VPSRADZ256mik,         0 },
    { X86::VPSRAQZ256rik,         X86::VPSRAQZ256mik,         0 },
    { X86::VPSRAWZ256rik,         X86::VPSRAWZ256mik,         0 },
    { X86::VPSRLDZ256rik,         X86::VPSRLDZ256mik,         0 },
    { X86::VPSRLQZ256rik,         X86::VPSRLQZ256mik,         0 },
    { X86::VPSRLWZ256rik,         X86::VPSRLWZ256mik,         0 },

    // AVX-512VL 128-bit masked foldable instructions
    { X86::VBROADCASTSSZ128rk,    X86::VBROADCASTSSZ128mk,    TB_NO_REVERSE },
    { X86::VPABSBZ128rrk,         X86::VPABSBZ128rmk,         0 },
    { X86::VPABSDZ128rrk,         X86::VPABSDZ128rmk,         0 },
    { X86::VPABSQZ128rrk,         X86::VPABSQZ128rmk,         0 },
    { X86::VPABSWZ128rrk,         X86::VPABSWZ128rmk,         0 },
    { X86::VPERMILPDZ128rik,      X86::VPERMILPDZ128mik,      0 },
    { X86::VPERMILPSZ128rik,      X86::VPERMILPSZ128mik,      0 },
    { X86::VPMOVSXBDZ128rrk,      X86::VPMOVSXBDZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXBQZ128rrk,      X86::VPMOVSXBQZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXBWZ128rrk,      X86::VPMOVSXBWZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXDQZ128rrk,      X86::VPMOVSXDQZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXWDZ128rrk,      X86::VPMOVSXWDZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVSXWQZ128rrk,      X86::VPMOVSXWQZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXBDZ128rrk,      X86::VPMOVZXBDZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXBQZ128rrk,      X86::VPMOVZXBQZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXBWZ128rrk,      X86::VPMOVZXBWZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXDQZ128rrk,      X86::VPMOVZXDQZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXWDZ128rrk,      X86::VPMOVZXWDZ128rmk,      TB_NO_REVERSE },
    { X86::VPMOVZXWQZ128rrk,      X86::VPMOVZXWQZ128rmk,      TB_NO_REVERSE },
    { X86::VPSHUFDZ128rik,        X86::VPSHUFDZ128mik,        0 },
    { X86::VPSHUFHWZ128rik,       X86::VPSHUFHWZ128mik,       0 },
    { X86::VPSHUFLWZ128rik,       X86::VPSHUFLWZ128mik,       0 },
    { X86::VPSLLDZ128rik,         X86::VPSLLDZ128mik,         0 },
    { X86::VPSLLQZ128rik,         X86::VPSLLQZ128mik,         0 },
    { X86::VPSLLWZ128rik,         X86::VPSLLWZ128mik,         0 },
    { X86::VPSRADZ128rik,         X86::VPSRADZ128mik,         0 },
    { X86::VPSRAQZ128rik,         X86::VPSRAQZ128mik,         0 },
    { X86::VPSRAWZ128rik,         X86::VPSRAWZ128mik,         0 },
    { X86::VPSRLDZ128rik,         X86::VPSRLDZ128mik,         0 },
    { X86::VPSRLQZ128rik,         X86::VPSRLQZ128mik,         0 },
    { X86::VPSRLWZ128rik,         X86::VPSRLWZ128mik,         0 },
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable3) {
    AddTableEntry(RegOp2MemOpTable3, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 3, folded load
                  Entry.Flags | TB_INDEX_3 | TB_FOLDED_LOAD);
  }
  auto I = X86InstrFMA3Info::rm_begin();
  auto E = X86InstrFMA3Info::rm_end();
  for (; I != E; ++I) {
    if (!I.getGroup()->isKMasked()) {
      // Intrinsic forms need to pass TB_NO_REVERSE.
      if (I.getGroup()->isIntrinsic()) {
        AddTableEntry(RegOp2MemOpTable3, MemOp2RegOpTable,
                      I.getRegOpcode(), I.getMemOpcode(),
                      TB_ALIGN_NONE | TB_INDEX_3 | TB_FOLDED_LOAD | TB_NO_REVERSE);
      } else {
        AddTableEntry(RegOp2MemOpTable3, MemOp2RegOpTable,
                      I.getRegOpcode(), I.getMemOpcode(),
                      TB_ALIGN_NONE | TB_INDEX_3 | TB_FOLDED_LOAD);
      }
    }
  }

  static const X86MemoryFoldTableEntry MemoryFoldTable4[] = {
    // AVX-512 foldable masked instructions
    { X86::VADDPDZrrk,         X86::VADDPDZrmk,           0 },
    { X86::VADDPSZrrk,         X86::VADDPSZrmk,           0 },
    { X86::VADDSDZrr_Intk,     X86::VADDSDZrm_Intk,       TB_NO_REVERSE },
    { X86::VADDSSZrr_Intk,     X86::VADDSSZrm_Intk,       TB_NO_REVERSE },
    { X86::VALIGNDZrrik,       X86::VALIGNDZrmik,         0 },
    { X86::VALIGNQZrrik,       X86::VALIGNQZrmik,         0 },
    { X86::VANDNPDZrrk,        X86::VANDNPDZrmk,          0 },
    { X86::VANDNPSZrrk,        X86::VANDNPSZrmk,          0 },
    { X86::VANDPDZrrk,         X86::VANDPDZrmk,           0 },
    { X86::VANDPSZrrk,         X86::VANDPSZrmk,           0 },
    { X86::VDIVPDZrrk,         X86::VDIVPDZrmk,           0 },
    { X86::VDIVPSZrrk,         X86::VDIVPSZrmk,           0 },
    { X86::VDIVSDZrr_Intk,     X86::VDIVSDZrm_Intk,       TB_NO_REVERSE },
    { X86::VDIVSSZrr_Intk,     X86::VDIVSSZrm_Intk,       TB_NO_REVERSE },
    { X86::VINSERTF32x4Zrrk,   X86::VINSERTF32x4Zrmk,     0 },
    { X86::VINSERTF32x8Zrrk,   X86::VINSERTF32x8Zrmk,     0 },
    { X86::VINSERTF64x2Zrrk,   X86::VINSERTF64x2Zrmk,     0 },
    { X86::VINSERTF64x4Zrrk,   X86::VINSERTF64x4Zrmk,     0 },
    { X86::VINSERTI32x4Zrrk,   X86::VINSERTI32x4Zrmk,     0 },
    { X86::VINSERTI32x8Zrrk,   X86::VINSERTI32x8Zrmk,     0 },
    { X86::VINSERTI64x2Zrrk,   X86::VINSERTI64x2Zrmk,     0 },
    { X86::VINSERTI64x4Zrrk,   X86::VINSERTI64x4Zrmk,     0 },
    { X86::VMAXCPDZrrk,        X86::VMAXCPDZrmk,          0 },
    { X86::VMAXCPSZrrk,        X86::VMAXCPSZrmk,          0 },
    { X86::VMAXPDZrrk,         X86::VMAXPDZrmk,           0 },
    { X86::VMAXPSZrrk,         X86::VMAXPSZrmk,           0 },
    { X86::VMAXSDZrr_Intk,     X86::VMAXSDZrm_Intk,       0 },
    { X86::VMAXSSZrr_Intk,     X86::VMAXSSZrm_Intk,       0 },
    { X86::VMINCPDZrrk,        X86::VMINCPDZrmk,          0 },
    { X86::VMINCPSZrrk,        X86::VMINCPSZrmk,          0 },
    { X86::VMINPDZrrk,         X86::VMINPDZrmk,           0 },
    { X86::VMINPSZrrk,         X86::VMINPSZrmk,           0 },
    { X86::VMINSDZrr_Intk,     X86::VMINSDZrm_Intk,       0 },
    { X86::VMINSSZrr_Intk,     X86::VMINSSZrm_Intk,       0 },
    { X86::VMULPDZrrk,         X86::VMULPDZrmk,           0 },
    { X86::VMULPSZrrk,         X86::VMULPSZrmk,           0 },
    { X86::VMULSDZrr_Intk,     X86::VMULSDZrm_Intk,       TB_NO_REVERSE },
    { X86::VMULSSZrr_Intk,     X86::VMULSSZrm_Intk,       TB_NO_REVERSE },
    { X86::VORPDZrrk,          X86::VORPDZrmk,            0 },
    { X86::VORPSZrrk,          X86::VORPSZrmk,            0 },
    { X86::VPACKSSDWZrrk,      X86::VPACKSSDWZrmk,        0 },
    { X86::VPACKSSWBZrrk,      X86::VPACKSSWBZrmk,        0 },
    { X86::VPACKUSDWZrrk,      X86::VPACKUSDWZrmk,        0 },
    { X86::VPACKUSWBZrrk,      X86::VPACKUSWBZrmk,        0 },
    { X86::VPADDBZrrk,         X86::VPADDBZrmk,           0 },
    { X86::VPADDDZrrk,         X86::VPADDDZrmk,           0 },
    { X86::VPADDQZrrk,         X86::VPADDQZrmk,           0 },
    { X86::VPADDSBZrrk,        X86::VPADDSBZrmk,          0 },
    { X86::VPADDSWZrrk,        X86::VPADDSWZrmk,          0 },
    { X86::VPADDUSBZrrk,       X86::VPADDUSBZrmk,         0 },
    { X86::VPADDUSWZrrk,       X86::VPADDUSWZrmk,         0 },
    { X86::VPADDWZrrk,         X86::VPADDWZrmk,           0 },
    { X86::VPALIGNRZrrik,      X86::VPALIGNRZrmik,        0 },
    { X86::VPANDDZrrk,         X86::VPANDDZrmk,           0 },
    { X86::VPANDNDZrrk,        X86::VPANDNDZrmk,          0 },
    { X86::VPANDNQZrrk,        X86::VPANDNQZrmk,          0 },
    { X86::VPANDQZrrk,         X86::VPANDQZrmk,           0 },
    { X86::VPAVGBZrrk,         X86::VPAVGBZrmk,           0 },
    { X86::VPAVGWZrrk,         X86::VPAVGWZrmk,           0 },
    { X86::VPERMBZrrk,         X86::VPERMBZrmk,           0 },
    { X86::VPERMDZrrk,         X86::VPERMDZrmk,           0 },
    { X86::VPERMI2Brrk,        X86::VPERMI2Brmk,          0 },
    { X86::VPERMI2Drrk,        X86::VPERMI2Drmk,          0 },
    { X86::VPERMI2PSrrk,       X86::VPERMI2PSrmk,         0 },
    { X86::VPERMI2PDrrk,       X86::VPERMI2PDrmk,         0 },
    { X86::VPERMI2Qrrk,        X86::VPERMI2Qrmk,          0 },
    { X86::VPERMI2Wrrk,        X86::VPERMI2Wrmk,          0 },
    { X86::VPERMILPDZrrk,      X86::VPERMILPDZrmk,        0 },
    { X86::VPERMILPSZrrk,      X86::VPERMILPSZrmk,        0 },
    { X86::VPERMPDZrrk,        X86::VPERMPDZrmk,          0 },
    { X86::VPERMPSZrrk,        X86::VPERMPSZrmk,          0 },
    { X86::VPERMQZrrk,         X86::VPERMQZrmk,           0 },
    { X86::VPERMT2Brrk,        X86::VPERMT2Brmk,          0 },
    { X86::VPERMT2Drrk,        X86::VPERMT2Drmk,          0 },
    { X86::VPERMT2PSrrk,       X86::VPERMT2PSrmk,         0 },
    { X86::VPERMT2PDrrk,       X86::VPERMT2PDrmk,         0 },
    { X86::VPERMT2Qrrk,        X86::VPERMT2Qrmk,          0 },
    { X86::VPERMT2Wrrk,        X86::VPERMT2Wrmk,          0 },
    { X86::VPERMWZrrk,         X86::VPERMWZrmk,           0 },
    { X86::VPMADDUBSWZrrk,     X86::VPMADDUBSWZrmk,       0 },
    { X86::VPMADDWDZrrk,       X86::VPMADDWDZrmk,         0 },
    { X86::VPMAXSBZrrk,        X86::VPMAXSBZrmk,          0 },
    { X86::VPMAXSDZrrk,        X86::VPMAXSDZrmk,          0 },
    { X86::VPMAXSQZrrk,        X86::VPMAXSQZrmk,          0 },
    { X86::VPMAXSWZrrk,        X86::VPMAXSWZrmk,          0 },
    { X86::VPMAXUBZrrk,        X86::VPMAXUBZrmk,          0 },
    { X86::VPMAXUDZrrk,        X86::VPMAXUDZrmk,          0 },
    { X86::VPMAXUQZrrk,        X86::VPMAXUQZrmk,          0 },
    { X86::VPMAXUWZrrk,        X86::VPMAXUWZrmk,          0 },
    { X86::VPMINSBZrrk,        X86::VPMINSBZrmk,          0 },
    { X86::VPMINSDZrrk,        X86::VPMINSDZrmk,          0 },
    { X86::VPMINSQZrrk,        X86::VPMINSQZrmk,          0 },
    { X86::VPMINSWZrrk,        X86::VPMINSWZrmk,          0 },
    { X86::VPMINUBZrrk,        X86::VPMINUBZrmk,          0 },
    { X86::VPMINUDZrrk,        X86::VPMINUDZrmk,          0 },
    { X86::VPMINUQZrrk,        X86::VPMINUQZrmk,          0 },
    { X86::VPMINUWZrrk,        X86::VPMINUWZrmk,          0 },
    { X86::VPMULDQZrrk,        X86::VPMULDQZrmk,          0 },
    { X86::VPMULLDZrrk,        X86::VPMULLDZrmk,          0 },
    { X86::VPMULLQZrrk,        X86::VPMULLQZrmk,          0 },
    { X86::VPMULLWZrrk,        X86::VPMULLWZrmk,          0 },
    { X86::VPMULUDQZrrk,       X86::VPMULUDQZrmk,         0 },
    { X86::VPORDZrrk,          X86::VPORDZrmk,            0 },
    { X86::VPORQZrrk,          X86::VPORQZrmk,            0 },
    { X86::VPSHUFBZrrk,        X86::VPSHUFBZrmk,          0 },
    { X86::VPSLLDZrrk,         X86::VPSLLDZrmk,           0 },
    { X86::VPSLLQZrrk,         X86::VPSLLQZrmk,           0 },
    { X86::VPSLLVDZrrk,        X86::VPSLLVDZrmk,          0 },
    { X86::VPSLLVQZrrk,        X86::VPSLLVQZrmk,          0 },
    { X86::VPSLLVWZrrk,        X86::VPSLLVWZrmk,          0 },
    { X86::VPSLLWZrrk,         X86::VPSLLWZrmk,           0 },
    { X86::VPSRADZrrk,         X86::VPSRADZrmk,           0 },
    { X86::VPSRAQZrrk,         X86::VPSRAQZrmk,           0 },
    { X86::VPSRAVDZrrk,        X86::VPSRAVDZrmk,          0 },
    { X86::VPSRAVQZrrk,        X86::VPSRAVQZrmk,          0 },
    { X86::VPSRAVWZrrk,        X86::VPSRAVWZrmk,          0 },
    { X86::VPSRAWZrrk,         X86::VPSRAWZrmk,           0 },
    { X86::VPSRLDZrrk,         X86::VPSRLDZrmk,           0 },
    { X86::VPSRLQZrrk,         X86::VPSRLQZrmk,           0 },
    { X86::VPSRLVDZrrk,        X86::VPSRLVDZrmk,          0 },
    { X86::VPSRLVQZrrk,        X86::VPSRLVQZrmk,          0 },
    { X86::VPSRLVWZrrk,        X86::VPSRLVWZrmk,          0 },
    { X86::VPSRLWZrrk,         X86::VPSRLWZrmk,           0 },
    { X86::VPSUBBZrrk,         X86::VPSUBBZrmk,           0 },
    { X86::VPSUBDZrrk,         X86::VPSUBDZrmk,           0 },
    { X86::VPSUBQZrrk,         X86::VPSUBQZrmk,           0 },
    { X86::VPSUBSBZrrk,        X86::VPSUBSBZrmk,          0 },
    { X86::VPSUBSWZrrk,        X86::VPSUBSWZrmk,          0 },
    { X86::VPSUBUSBZrrk,       X86::VPSUBUSBZrmk,         0 },
    { X86::VPSUBUSWZrrk,       X86::VPSUBUSWZrmk,         0 },
    { X86::VPTERNLOGDZrrik,    X86::VPTERNLOGDZrmik,      0 },
    { X86::VPTERNLOGQZrrik,    X86::VPTERNLOGQZrmik,      0 },
    { X86::VPUNPCKHBWZrrk,     X86::VPUNPCKHBWZrmk,       0 },
    { X86::VPUNPCKHDQZrrk,     X86::VPUNPCKHDQZrmk,       0 },
    { X86::VPUNPCKHQDQZrrk,    X86::VPUNPCKHQDQZrmk,      0 },
    { X86::VPUNPCKHWDZrrk,     X86::VPUNPCKHWDZrmk,       0 },
    { X86::VPUNPCKLBWZrrk,     X86::VPUNPCKLBWZrmk,       0 },
    { X86::VPUNPCKLDQZrrk,     X86::VPUNPCKLDQZrmk,       0 },
    { X86::VPUNPCKLQDQZrrk,    X86::VPUNPCKLQDQZrmk,      0 },
    { X86::VPUNPCKLWDZrrk,     X86::VPUNPCKLWDZrmk,       0 },
    { X86::VPXORDZrrk,         X86::VPXORDZrmk,           0 },
    { X86::VPXORQZrrk,         X86::VPXORQZrmk,           0 },
    { X86::VSHUFPDZrrik,       X86::VSHUFPDZrmik,         0 },
    { X86::VSHUFPSZrrik,       X86::VSHUFPSZrmik,         0 },
    { X86::VSUBPDZrrk,         X86::VSUBPDZrmk,           0 },
    { X86::VSUBPSZrrk,         X86::VSUBPSZrmk,           0 },
    { X86::VSUBSDZrr_Intk,     X86::VSUBSDZrm_Intk,       TB_NO_REVERSE },
    { X86::VSUBSSZrr_Intk,     X86::VSUBSSZrm_Intk,       TB_NO_REVERSE },
    { X86::VUNPCKHPDZrrk,      X86::VUNPCKHPDZrmk,        0 },
    { X86::VUNPCKHPSZrrk,      X86::VUNPCKHPSZrmk,        0 },
    { X86::VUNPCKLPDZrrk,      X86::VUNPCKLPDZrmk,        0 },
    { X86::VUNPCKLPSZrrk,      X86::VUNPCKLPSZrmk,        0 },
    { X86::VXORPDZrrk,         X86::VXORPDZrmk,           0 },
    { X86::VXORPSZrrk,         X86::VXORPSZrmk,           0 },

    // AVX-512{F,VL} foldable masked instructions 256-bit
    { X86::VADDPDZ256rrk,      X86::VADDPDZ256rmk,        0 },
    { X86::VADDPSZ256rrk,      X86::VADDPSZ256rmk,        0 },
    { X86::VALIGNDZ256rrik,    X86::VALIGNDZ256rmik,      0 },
    { X86::VALIGNQZ256rrik,    X86::VALIGNQZ256rmik,      0 },
    { X86::VANDNPDZ256rrk,     X86::VANDNPDZ256rmk,       0 },
    { X86::VANDNPSZ256rrk,     X86::VANDNPSZ256rmk,       0 },
    { X86::VANDPDZ256rrk,      X86::VANDPDZ256rmk,        0 },
    { X86::VANDPSZ256rrk,      X86::VANDPSZ256rmk,        0 },
    { X86::VDIVPDZ256rrk,      X86::VDIVPDZ256rmk,        0 },
    { X86::VDIVPSZ256rrk,      X86::VDIVPSZ256rmk,        0 },
    { X86::VINSERTF32x4Z256rrk,X86::VINSERTF32x4Z256rmk,  0 },
    { X86::VINSERTF64x2Z256rrk,X86::VINSERTF64x2Z256rmk,  0 },
    { X86::VINSERTI32x4Z256rrk,X86::VINSERTI32x4Z256rmk,  0 },
    { X86::VINSERTI64x2Z256rrk,X86::VINSERTI64x2Z256rmk,  0 },
    { X86::VMAXCPDZ256rrk,     X86::VMAXCPDZ256rmk,       0 },
    { X86::VMAXCPSZ256rrk,     X86::VMAXCPSZ256rmk,       0 },
    { X86::VMAXPDZ256rrk,      X86::VMAXPDZ256rmk,        0 },
    { X86::VMAXPSZ256rrk,      X86::VMAXPSZ256rmk,        0 },
    { X86::VMINCPDZ256rrk,     X86::VMINCPDZ256rmk,       0 },
    { X86::VMINCPSZ256rrk,     X86::VMINCPSZ256rmk,       0 },
    { X86::VMINPDZ256rrk,      X86::VMINPDZ256rmk,        0 },
    { X86::VMINPSZ256rrk,      X86::VMINPSZ256rmk,        0 },
    { X86::VMULPDZ256rrk,      X86::VMULPDZ256rmk,        0 },
    { X86::VMULPSZ256rrk,      X86::VMULPSZ256rmk,        0 },
    { X86::VORPDZ256rrk,       X86::VORPDZ256rmk,         0 },
    { X86::VORPSZ256rrk,       X86::VORPSZ256rmk,         0 },
    { X86::VPACKSSDWZ256rrk,   X86::VPACKSSDWZ256rmk,     0 },
    { X86::VPACKSSWBZ256rrk,   X86::VPACKSSWBZ256rmk,     0 },
    { X86::VPACKUSDWZ256rrk,   X86::VPACKUSDWZ256rmk,     0 },
    { X86::VPACKUSWBZ256rrk,   X86::VPACKUSWBZ256rmk,     0 },
    { X86::VPADDBZ256rrk,      X86::VPADDBZ256rmk,        0 },
    { X86::VPADDDZ256rrk,      X86::VPADDDZ256rmk,        0 },
    { X86::VPADDQZ256rrk,      X86::VPADDQZ256rmk,        0 },
    { X86::VPADDSBZ256rrk,     X86::VPADDSBZ256rmk,       0 },
    { X86::VPADDSWZ256rrk,     X86::VPADDSWZ256rmk,       0 },
    { X86::VPADDUSBZ256rrk,    X86::VPADDUSBZ256rmk,      0 },
    { X86::VPADDUSWZ256rrk,    X86::VPADDUSWZ256rmk,      0 },
    { X86::VPADDWZ256rrk,      X86::VPADDWZ256rmk,        0 },
    { X86::VPALIGNRZ256rrik,   X86::VPALIGNRZ256rmik,     0 },
    { X86::VPANDDZ256rrk,      X86::VPANDDZ256rmk,        0 },
    { X86::VPANDNDZ256rrk,     X86::VPANDNDZ256rmk,       0 },
    { X86::VPANDNQZ256rrk,     X86::VPANDNQZ256rmk,       0 },
    { X86::VPANDQZ256rrk,      X86::VPANDQZ256rmk,        0 },
    { X86::VPAVGBZ256rrk,      X86::VPAVGBZ256rmk,        0 },
    { X86::VPAVGWZ256rrk,      X86::VPAVGWZ256rmk,        0 },
    { X86::VPERMBZ256rrk,      X86::VPERMBZ256rmk,        0 },
    { X86::VPERMDZ256rrk,      X86::VPERMDZ256rmk,        0 },
    { X86::VPERMI2B256rrk,     X86::VPERMI2B256rmk,       0 },
    { X86::VPERMI2D256rrk,     X86::VPERMI2D256rmk,       0 },
    { X86::VPERMI2PD256rrk,    X86::VPERMI2PD256rmk,      0 },
    { X86::VPERMI2PS256rrk,    X86::VPERMI2PS256rmk,      0 },
    { X86::VPERMI2Q256rrk,     X86::VPERMI2Q256rmk,       0 },
    { X86::VPERMI2W256rrk,     X86::VPERMI2W256rmk,       0 },
    { X86::VPERMILPDZ256rrk,   X86::VPERMILPDZ256rmk,     0 },
    { X86::VPERMILPSZ256rrk,   X86::VPERMILPSZ256rmk,     0 },
    { X86::VPERMPDZ256rrk,     X86::VPERMPDZ256rmk,       0 },
    { X86::VPERMPSZ256rrk,     X86::VPERMPSZ256rmk,       0 },
    { X86::VPERMQZ256rrk,      X86::VPERMQZ256rmk,        0 },
    { X86::VPERMT2B256rrk,     X86::VPERMT2B256rmk,       0 },
    { X86::VPERMT2D256rrk,     X86::VPERMT2D256rmk,       0 },
    { X86::VPERMT2PD256rrk,    X86::VPERMT2PD256rmk,      0 },
    { X86::VPERMT2PS256rrk,    X86::VPERMT2PS256rmk,      0 },
    { X86::VPERMT2Q256rrk,     X86::VPERMT2Q256rmk,       0 },
    { X86::VPERMT2W256rrk,     X86::VPERMT2W256rmk,       0 },
    { X86::VPERMWZ256rrk,      X86::VPERMWZ256rmk,        0 },
    { X86::VPMADDUBSWZ256rrk,  X86::VPMADDUBSWZ256rmk,    0 },
    { X86::VPMADDWDZ256rrk,    X86::VPMADDWDZ256rmk,      0 },
    { X86::VPMAXSBZ256rrk,     X86::VPMAXSBZ256rmk,       0 },
    { X86::VPMAXSDZ256rrk,     X86::VPMAXSDZ256rmk,       0 },
    { X86::VPMAXSQZ256rrk,     X86::VPMAXSQZ256rmk,       0 },
    { X86::VPMAXSWZ256rrk,     X86::VPMAXSWZ256rmk,       0 },
    { X86::VPMAXUBZ256rrk,     X86::VPMAXUBZ256rmk,       0 },
    { X86::VPMAXUDZ256rrk,     X86::VPMAXUDZ256rmk,       0 },
    { X86::VPMAXUQZ256rrk,     X86::VPMAXUQZ256rmk,       0 },
    { X86::VPMAXUWZ256rrk,     X86::VPMAXUWZ256rmk,       0 },
    { X86::VPMINSBZ256rrk,     X86::VPMINSBZ256rmk,       0 },
    { X86::VPMINSDZ256rrk,     X86::VPMINSDZ256rmk,       0 },
    { X86::VPMINSQZ256rrk,     X86::VPMINSQZ256rmk,       0 },
    { X86::VPMINSWZ256rrk,     X86::VPMINSWZ256rmk,       0 },
    { X86::VPMINUBZ256rrk,     X86::VPMINUBZ256rmk,       0 },
    { X86::VPMINUDZ256rrk,     X86::VPMINUDZ256rmk,       0 },
    { X86::VPMINUQZ256rrk,     X86::VPMINUQZ256rmk,       0 },
    { X86::VPMINUWZ256rrk,     X86::VPMINUWZ256rmk,       0 },
    { X86::VPMULDQZ256rrk,     X86::VPMULDQZ256rmk,       0 },
    { X86::VPMULLDZ256rrk,     X86::VPMULLDZ256rmk,       0 },
    { X86::VPMULLQZ256rrk,     X86::VPMULLQZ256rmk,       0 },
    { X86::VPMULLWZ256rrk,     X86::VPMULLWZ256rmk,       0 },
    { X86::VPMULUDQZ256rrk,    X86::VPMULUDQZ256rmk,      0 },
    { X86::VPORDZ256rrk,       X86::VPORDZ256rmk,         0 },
    { X86::VPORQZ256rrk,       X86::VPORQZ256rmk,         0 },
    { X86::VPSHUFBZ256rrk,     X86::VPSHUFBZ256rmk,       0 },
    { X86::VPSLLDZ256rrk,      X86::VPSLLDZ256rmk,        0 },
    { X86::VPSLLQZ256rrk,      X86::VPSLLQZ256rmk,        0 },
    { X86::VPSLLVDZ256rrk,     X86::VPSLLVDZ256rmk,       0 },
    { X86::VPSLLVQZ256rrk,     X86::VPSLLVQZ256rmk,       0 },
    { X86::VPSLLVWZ256rrk,     X86::VPSLLVWZ256rmk,       0 },
    { X86::VPSLLWZ256rrk,      X86::VPSLLWZ256rmk,        0 },
    { X86::VPSRADZ256rrk,      X86::VPSRADZ256rmk,        0 },
    { X86::VPSRAQZ256rrk,      X86::VPSRAQZ256rmk,        0 },
    { X86::VPSRAVDZ256rrk,     X86::VPSRAVDZ256rmk,       0 },
    { X86::VPSRAVQZ256rrk,     X86::VPSRAVQZ256rmk,       0 },
    { X86::VPSRAVWZ256rrk,     X86::VPSRAVWZ256rmk,       0 },
    { X86::VPSRAWZ256rrk,      X86::VPSRAWZ256rmk,        0 },
    { X86::VPSRLDZ256rrk,      X86::VPSRLDZ256rmk,        0 },
    { X86::VPSRLQZ256rrk,      X86::VPSRLQZ256rmk,        0 },
    { X86::VPSRLVDZ256rrk,     X86::VPSRLVDZ256rmk,       0 },
    { X86::VPSRLVQZ256rrk,     X86::VPSRLVQZ256rmk,       0 },
    { X86::VPSRLVWZ256rrk,     X86::VPSRLVWZ256rmk,       0 },
    { X86::VPSRLWZ256rrk,      X86::VPSRLWZ256rmk,        0 },
    { X86::VPSUBBZ256rrk,      X86::VPSUBBZ256rmk,        0 },
    { X86::VPSUBDZ256rrk,      X86::VPSUBDZ256rmk,        0 },
    { X86::VPSUBQZ256rrk,      X86::VPSUBQZ256rmk,        0 },
    { X86::VPSUBSBZ256rrk,     X86::VPSUBSBZ256rmk,       0 },
    { X86::VPSUBSWZ256rrk,     X86::VPSUBSWZ256rmk,       0 },
    { X86::VPSUBUSBZ256rrk,    X86::VPSUBUSBZ256rmk,      0 },
    { X86::VPSUBUSWZ256rrk,    X86::VPSUBUSWZ256rmk,      0 },
    { X86::VPSUBWZ256rrk,      X86::VPSUBWZ256rmk,        0 },
    { X86::VPTERNLOGDZ256rrik, X86::VPTERNLOGDZ256rmik,   0 },
    { X86::VPTERNLOGQZ256rrik, X86::VPTERNLOGQZ256rmik,   0 },
    { X86::VPUNPCKHBWZ256rrk,  X86::VPUNPCKHBWZ256rmk,    0 },
    { X86::VPUNPCKHDQZ256rrk,  X86::VPUNPCKHDQZ256rmk,    0 },
    { X86::VPUNPCKHQDQZ256rrk, X86::VPUNPCKHQDQZ256rmk,   0 },
    { X86::VPUNPCKHWDZ256rrk,  X86::VPUNPCKHWDZ256rmk,    0 },
    { X86::VPUNPCKLBWZ256rrk,  X86::VPUNPCKLBWZ256rmk,    0 },
    { X86::VPUNPCKLDQZ256rrk,  X86::VPUNPCKLDQZ256rmk,    0 },
    { X86::VPUNPCKLQDQZ256rrk, X86::VPUNPCKLQDQZ256rmk,   0 },
    { X86::VPUNPCKLWDZ256rrk,  X86::VPUNPCKLWDZ256rmk,    0 },
    { X86::VPXORDZ256rrk,      X86::VPXORDZ256rmk,        0 },
    { X86::VPXORQZ256rrk,      X86::VPXORQZ256rmk,        0 },
    { X86::VSHUFPDZ256rrik,    X86::VSHUFPDZ256rmik,      0 },
    { X86::VSHUFPSZ256rrik,    X86::VSHUFPSZ256rmik,      0 },
    { X86::VSUBPDZ256rrk,      X86::VSUBPDZ256rmk,        0 },
    { X86::VSUBPSZ256rrk,      X86::VSUBPSZ256rmk,        0 },
    { X86::VUNPCKHPDZ256rrk,   X86::VUNPCKHPDZ256rmk,     0 },
    { X86::VUNPCKHPSZ256rrk,   X86::VUNPCKHPSZ256rmk,     0 },
    { X86::VUNPCKLPDZ256rrk,   X86::VUNPCKLPDZ256rmk,     0 },
    { X86::VUNPCKLPSZ256rrk,   X86::VUNPCKLPSZ256rmk,     0 },
    { X86::VXORPDZ256rrk,      X86::VXORPDZ256rmk,        0 },
    { X86::VXORPSZ256rrk,      X86::VXORPSZ256rmk,        0 },

    // AVX-512{F,VL} foldable instructions 128-bit
    { X86::VADDPDZ128rrk,      X86::VADDPDZ128rmk,        0 },
    { X86::VADDPSZ128rrk,      X86::VADDPSZ128rmk,        0 },
    { X86::VALIGNDZ128rrik,    X86::VALIGNDZ128rmik,      0 },
    { X86::VALIGNQZ128rrik,    X86::VALIGNQZ128rmik,      0 },
    { X86::VANDNPDZ128rrk,     X86::VANDNPDZ128rmk,       0 },
    { X86::VANDNPSZ128rrk,     X86::VANDNPSZ128rmk,       0 },
    { X86::VANDPDZ128rrk,      X86::VANDPDZ128rmk,        0 },
    { X86::VANDPSZ128rrk,      X86::VANDPSZ128rmk,        0 },
    { X86::VDIVPDZ128rrk,      X86::VDIVPDZ128rmk,        0 },
    { X86::VDIVPSZ128rrk,      X86::VDIVPSZ128rmk,        0 },
    { X86::VMAXCPDZ128rrk,     X86::VMAXCPDZ128rmk,       0 },
    { X86::VMAXCPSZ128rrk,     X86::VMAXCPSZ128rmk,       0 },
    { X86::VMAXPDZ128rrk,      X86::VMAXPDZ128rmk,        0 },
    { X86::VMAXPSZ128rrk,      X86::VMAXPSZ128rmk,        0 },
    { X86::VMINCPDZ128rrk,     X86::VMINCPDZ128rmk,       0 },
    { X86::VMINCPSZ128rrk,     X86::VMINCPSZ128rmk,       0 },
    { X86::VMINPDZ128rrk,      X86::VMINPDZ128rmk,        0 },
    { X86::VMINPSZ128rrk,      X86::VMINPSZ128rmk,        0 },
    { X86::VMULPDZ128rrk,      X86::VMULPDZ128rmk,        0 },
    { X86::VMULPSZ128rrk,      X86::VMULPSZ128rmk,        0 },
    { X86::VORPDZ128rrk,       X86::VORPDZ128rmk,         0 },
    { X86::VORPSZ128rrk,       X86::VORPSZ128rmk,         0 },
    { X86::VPACKSSDWZ128rrk,   X86::VPACKSSDWZ128rmk,     0 },
    { X86::VPACKSSWBZ128rrk,   X86::VPACKSSWBZ128rmk,     0 },
    { X86::VPACKUSDWZ128rrk,   X86::VPACKUSDWZ128rmk,     0 },
    { X86::VPACKUSWBZ128rrk,   X86::VPACKUSWBZ128rmk,     0 },
    { X86::VPADDBZ128rrk,      X86::VPADDBZ128rmk,        0 },
    { X86::VPADDDZ128rrk,      X86::VPADDDZ128rmk,        0 },
    { X86::VPADDQZ128rrk,      X86::VPADDQZ128rmk,        0 },
    { X86::VPADDSBZ128rrk,     X86::VPADDSBZ128rmk,       0 },
    { X86::VPADDSWZ128rrk,     X86::VPADDSWZ128rmk,       0 },
    { X86::VPADDUSBZ128rrk,    X86::VPADDUSBZ128rmk,      0 },
    { X86::VPADDUSWZ128rrk,    X86::VPADDUSWZ128rmk,      0 },
    { X86::VPADDWZ128rrk,      X86::VPADDWZ128rmk,        0 },
    { X86::VPALIGNRZ128rrik,   X86::VPALIGNRZ128rmik,     0 },
    { X86::VPANDDZ128rrk,      X86::VPANDDZ128rmk,        0 },
    { X86::VPANDNDZ128rrk,     X86::VPANDNDZ128rmk,       0 },
    { X86::VPANDNQZ128rrk,     X86::VPANDNQZ128rmk,       0 },
    { X86::VPANDQZ128rrk,      X86::VPANDQZ128rmk,        0 },
    { X86::VPAVGBZ128rrk,      X86::VPAVGBZ128rmk,        0 },
    { X86::VPAVGWZ128rrk,      X86::VPAVGWZ128rmk,        0 },
    { X86::VPERMBZ128rrk,      X86::VPERMBZ128rmk,        0 },
    { X86::VPERMI2B128rrk,     X86::VPERMI2B128rmk,       0 },
    { X86::VPERMI2D128rrk,     X86::VPERMI2D128rmk,       0 },
    { X86::VPERMI2PD128rrk,    X86::VPERMI2PD128rmk,      0 },
    { X86::VPERMI2PS128rrk,    X86::VPERMI2PS128rmk,      0 },
    { X86::VPERMI2Q128rrk,     X86::VPERMI2Q128rmk,       0 },
    { X86::VPERMI2W128rrk,     X86::VPERMI2W128rmk,       0 },
    { X86::VPERMILPDZ128rrk,   X86::VPERMILPDZ128rmk,     0 },
    { X86::VPERMILPSZ128rrk,   X86::VPERMILPSZ128rmk,     0 },
    { X86::VPERMT2B128rrk,     X86::VPERMT2B128rmk,       0 },
    { X86::VPERMT2D128rrk,     X86::VPERMT2D128rmk,       0 },
    { X86::VPERMT2PD128rrk,    X86::VPERMT2PD128rmk,      0 },
    { X86::VPERMT2PS128rrk,    X86::VPERMT2PS128rmk,      0 },
    { X86::VPERMT2Q128rrk,     X86::VPERMT2Q128rmk,       0 },
    { X86::VPERMT2W128rrk,     X86::VPERMT2W128rmk,       0 },
    { X86::VPERMWZ128rrk,      X86::VPERMWZ128rmk,        0 },
    { X86::VPMADDUBSWZ128rrk,  X86::VPMADDUBSWZ128rmk,    0 },
    { X86::VPMADDWDZ128rrk,    X86::VPMADDWDZ128rmk,      0 },
    { X86::VPMAXSBZ128rrk,     X86::VPMAXSBZ128rmk,       0 },
    { X86::VPMAXSDZ128rrk,     X86::VPMAXSDZ128rmk,       0 },
    { X86::VPMAXSQZ128rrk,     X86::VPMAXSQZ128rmk,       0 },
    { X86::VPMAXSWZ128rrk,     X86::VPMAXSWZ128rmk,       0 },
    { X86::VPMAXUBZ128rrk,     X86::VPMAXUBZ128rmk,       0 },
    { X86::VPMAXUDZ128rrk,     X86::VPMAXUDZ128rmk,       0 },
    { X86::VPMAXUQZ128rrk,     X86::VPMAXUQZ128rmk,       0 },
    { X86::VPMAXUWZ128rrk,     X86::VPMAXUWZ128rmk,       0 },
    { X86::VPMINSBZ128rrk,     X86::VPMINSBZ128rmk,       0 },
    { X86::VPMINSDZ128rrk,     X86::VPMINSDZ128rmk,       0 },
    { X86::VPMINSQZ128rrk,     X86::VPMINSQZ128rmk,       0 },
    { X86::VPMINSWZ128rrk,     X86::VPMINSWZ128rmk,       0 },
    { X86::VPMINUBZ128rrk,     X86::VPMINUBZ128rmk,       0 },
    { X86::VPMINUDZ128rrk,     X86::VPMINUDZ128rmk,       0 },
    { X86::VPMINUQZ128rrk,     X86::VPMINUQZ128rmk,       0 },
    { X86::VPMINUWZ128rrk,     X86::VPMINUWZ128rmk,       0 },
    { X86::VPMULDQZ128rrk,     X86::VPMULDQZ128rmk,       0 },
    { X86::VPMULLDZ128rrk,     X86::VPMULLDZ128rmk,       0 },
    { X86::VPMULLQZ128rrk,     X86::VPMULLQZ128rmk,       0 },
    { X86::VPMULLWZ128rrk,     X86::VPMULLWZ128rmk,       0 },
    { X86::VPMULUDQZ128rrk,    X86::VPMULUDQZ128rmk,      0 },
    { X86::VPORDZ128rrk,       X86::VPORDZ128rmk,         0 },
    { X86::VPORQZ128rrk,       X86::VPORQZ128rmk,         0 },
    { X86::VPSHUFBZ128rrk,     X86::VPSHUFBZ128rmk,       0 },
    { X86::VPSLLDZ128rrk,      X86::VPSLLDZ128rmk,        0 },
    { X86::VPSLLQZ128rrk,      X86::VPSLLQZ128rmk,        0 },
    { X86::VPSLLVDZ128rrk,     X86::VPSLLVDZ128rmk,       0 },
    { X86::VPSLLVQZ128rrk,     X86::VPSLLVQZ128rmk,       0 },
    { X86::VPSLLVWZ128rrk,     X86::VPSLLVWZ128rmk,       0 },
    { X86::VPSLLWZ128rrk,      X86::VPSLLWZ128rmk,        0 },
    { X86::VPSRADZ128rrk,      X86::VPSRADZ128rmk,        0 },
    { X86::VPSRAQZ128rrk,      X86::VPSRAQZ128rmk,        0 },
    { X86::VPSRAVDZ128rrk,     X86::VPSRAVDZ128rmk,       0 },
    { X86::VPSRAVQZ128rrk,     X86::VPSRAVQZ128rmk,       0 },
    { X86::VPSRAVWZ128rrk,     X86::VPSRAVWZ128rmk,       0 },
    { X86::VPSRAWZ128rrk,      X86::VPSRAWZ128rmk,        0 },
    { X86::VPSRLDZ128rrk,      X86::VPSRLDZ128rmk,        0 },
    { X86::VPSRLQZ128rrk,      X86::VPSRLQZ128rmk,        0 },
    { X86::VPSRLVDZ128rrk,     X86::VPSRLVDZ128rmk,       0 },
    { X86::VPSRLVQZ128rrk,     X86::VPSRLVQZ128rmk,       0 },
    { X86::VPSRLVWZ128rrk,     X86::VPSRLVWZ128rmk,       0 },
    { X86::VPSRLWZ128rrk,      X86::VPSRLWZ128rmk,        0 },
    { X86::VPSUBBZ128rrk,      X86::VPSUBBZ128rmk,        0 },
    { X86::VPSUBDZ128rrk,      X86::VPSUBDZ128rmk,        0 },
    { X86::VPSUBQZ128rrk,      X86::VPSUBQZ128rmk,        0 },
    { X86::VPSUBSBZ128rrk,     X86::VPSUBSBZ128rmk,       0 },
    { X86::VPSUBSWZ128rrk,     X86::VPSUBSWZ128rmk,       0 },
    { X86::VPSUBUSBZ128rrk,    X86::VPSUBUSBZ128rmk,      0 },
    { X86::VPSUBUSWZ128rrk,    X86::VPSUBUSWZ128rmk,      0 },
    { X86::VPSUBWZ128rrk,      X86::VPSUBWZ128rmk,        0 },
    { X86::VPTERNLOGDZ128rrik, X86::VPTERNLOGDZ128rmik,   0 },
    { X86::VPTERNLOGQZ128rrik, X86::VPTERNLOGQZ128rmik,   0 },
    { X86::VPUNPCKHBWZ128rrk,  X86::VPUNPCKHBWZ128rmk,    0 },
    { X86::VPUNPCKHDQZ128rrk,  X86::VPUNPCKHDQZ128rmk,    0 },
    { X86::VPUNPCKHQDQZ128rrk, X86::VPUNPCKHQDQZ128rmk,   0 },
    { X86::VPUNPCKHWDZ128rrk,  X86::VPUNPCKHWDZ128rmk,    0 },
    { X86::VPUNPCKLBWZ128rrk,  X86::VPUNPCKLBWZ128rmk,    0 },
    { X86::VPUNPCKLDQZ128rrk,  X86::VPUNPCKLDQZ128rmk,    0 },
    { X86::VPUNPCKLQDQZ128rrk, X86::VPUNPCKLQDQZ128rmk,   0 },
    { X86::VPUNPCKLWDZ128rrk,  X86::VPUNPCKLWDZ128rmk,    0 },
    { X86::VPXORDZ128rrk,      X86::VPXORDZ128rmk,        0 },
    { X86::VPXORQZ128rrk,      X86::VPXORQZ128rmk,        0 },
    { X86::VSHUFPDZ128rrik,    X86::VSHUFPDZ128rmik,      0 },
    { X86::VSHUFPSZ128rrik,    X86::VSHUFPSZ128rmik,      0 },
    { X86::VSUBPDZ128rrk,      X86::VSUBPDZ128rmk,        0 },
    { X86::VSUBPSZ128rrk,      X86::VSUBPSZ128rmk,        0 },
    { X86::VUNPCKHPDZ128rrk,   X86::VUNPCKHPDZ128rmk,     0 },
    { X86::VUNPCKHPSZ128rrk,   X86::VUNPCKHPSZ128rmk,     0 },
    { X86::VUNPCKLPDZ128rrk,   X86::VUNPCKLPDZ128rmk,     0 },
    { X86::VUNPCKLPSZ128rrk,   X86::VUNPCKLPSZ128rmk,     0 },
    { X86::VXORPDZ128rrk,      X86::VXORPDZ128rmk,        0 },
    { X86::VXORPSZ128rrk,      X86::VXORPSZ128rmk,        0 },

    // 512-bit three source instructions with zero masking.
    { X86::VPERMI2Brrkz,       X86::VPERMI2Brmkz,         0 },
    { X86::VPERMI2Drrkz,       X86::VPERMI2Drmkz,         0 },
    { X86::VPERMI2PSrrkz,      X86::VPERMI2PSrmkz,        0 },
    { X86::VPERMI2PDrrkz,      X86::VPERMI2PDrmkz,        0 },
    { X86::VPERMI2Qrrkz,       X86::VPERMI2Qrmkz,         0 },
    { X86::VPERMI2Wrrkz,       X86::VPERMI2Wrmkz,         0 },
    { X86::VPERMT2Brrkz,       X86::VPERMT2Brmkz,         0 },
    { X86::VPERMT2Drrkz,       X86::VPERMT2Drmkz,         0 },
    { X86::VPERMT2PSrrkz,      X86::VPERMT2PSrmkz,        0 },
    { X86::VPERMT2PDrrkz,      X86::VPERMT2PDrmkz,        0 },
    { X86::VPERMT2Qrrkz,       X86::VPERMT2Qrmkz,         0 },
    { X86::VPERMT2Wrrkz,       X86::VPERMT2Wrmkz,         0 },
    { X86::VPTERNLOGDZrrikz,   X86::VPTERNLOGDZrmikz,     0 },
    { X86::VPTERNLOGQZrrikz,   X86::VPTERNLOGQZrmikz,     0 },

    // 256-bit three source instructions with zero masking.
    { X86::VPERMI2B256rrkz,    X86::VPERMI2B256rmkz,      0 },
    { X86::VPERMI2D256rrkz,    X86::VPERMI2D256rmkz,      0 },
    { X86::VPERMI2PD256rrkz,   X86::VPERMI2PD256rmkz,     0 },
    { X86::VPERMI2PS256rrkz,   X86::VPERMI2PS256rmkz,     0 },
    { X86::VPERMI2Q256rrkz,    X86::VPERMI2Q256rmkz,      0 },
    { X86::VPERMI2W256rrkz,    X86::VPERMI2W256rmkz,      0 },
    { X86::VPERMT2B256rrkz,    X86::VPERMT2B256rmkz,      0 },
    { X86::VPERMT2D256rrkz,    X86::VPERMT2D256rmkz,      0 },
    { X86::VPERMT2PD256rrkz,   X86::VPERMT2PD256rmkz,     0 },
    { X86::VPERMT2PS256rrkz,   X86::VPERMT2PS256rmkz,     0 },
    { X86::VPERMT2Q256rrkz,    X86::VPERMT2Q256rmkz,      0 },
    { X86::VPERMT2W256rrkz,    X86::VPERMT2W256rmkz,      0 },
    { X86::VPTERNLOGDZ256rrikz,X86::VPTERNLOGDZ256rmikz,  0 },
    { X86::VPTERNLOGQZ256rrikz,X86::VPTERNLOGQZ256rmikz,  0 },

    // 128-bit three source instructions with zero masking.
    { X86::VPERMI2B128rrkz,    X86::VPERMI2B128rmkz,      0 },
    { X86::VPERMI2D128rrkz,    X86::VPERMI2D128rmkz,      0 },
    { X86::VPERMI2PD128rrkz,   X86::VPERMI2PD128rmkz,     0 },
    { X86::VPERMI2PS128rrkz,   X86::VPERMI2PS128rmkz,     0 },
    { X86::VPERMI2Q128rrkz,    X86::VPERMI2Q128rmkz,      0 },
    { X86::VPERMI2W128rrkz,    X86::VPERMI2W128rmkz,      0 },
    { X86::VPERMT2B128rrkz,    X86::VPERMT2B128rmkz,      0 },
    { X86::VPERMT2D128rrkz,    X86::VPERMT2D128rmkz,      0 },
    { X86::VPERMT2PD128rrkz,   X86::VPERMT2PD128rmkz,     0 },
    { X86::VPERMT2PS128rrkz,   X86::VPERMT2PS128rmkz,     0 },
    { X86::VPERMT2Q128rrkz,    X86::VPERMT2Q128rmkz,      0 },
    { X86::VPERMT2W128rrkz,    X86::VPERMT2W128rmkz,      0 },
    { X86::VPTERNLOGDZ128rrikz,X86::VPTERNLOGDZ128rmikz,  0 },
    { X86::VPTERNLOGQZ128rrikz,X86::VPTERNLOGQZ128rmikz,  0 },
  };

  for (X86MemoryFoldTableEntry Entry : MemoryFoldTable4) {
    AddTableEntry(RegOp2MemOpTable4, MemOp2RegOpTable,
                  Entry.RegOp, Entry.MemOp,
                  // Index 4, folded load
                  Entry.Flags | TB_INDEX_4 | TB_FOLDED_LOAD);
  }
  for (I = X86InstrFMA3Info::rm_begin(); I != E; ++I) {
    if (I.getGroup()->isKMasked()) {
      // Intrinsics need to pass TB_NO_REVERSE.
      if (I.getGroup()->isIntrinsic()) {
        AddTableEntry(RegOp2MemOpTable4, MemOp2RegOpTable,
                      I.getRegOpcode(), I.getMemOpcode(),
                      TB_ALIGN_NONE | TB_INDEX_4 | TB_FOLDED_LOAD | TB_NO_REVERSE);
      } else {
        AddTableEntry(RegOp2MemOpTable4, MemOp2RegOpTable,
                      I.getRegOpcode(), I.getMemOpcode(),
                      TB_ALIGN_NONE | TB_INDEX_4 | TB_FOLDED_LOAD);
      }
    }
  }
}

void
X86InstrInfo::AddTableEntry(RegOp2MemOpTableType &R2MTable,
                            MemOp2RegOpTableType &M2RTable,
                            uint16_t RegOp, uint16_t MemOp, uint16_t Flags) {
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

int X86InstrInfo::getSPAdjust(const MachineInstr &MI) const {
  const MachineFunction *MF = MI.getParent()->getParent();
  const TargetFrameLowering *TFI = MF->getSubtarget().getFrameLowering();

  if (isFrameInstr(MI)) {
    unsigned StackAlign = TFI->getStackAlignment();
    int SPAdj = alignTo(getFrameSize(MI), StackAlign);
    SPAdj -= getFrameAdjustment(MI);
    if (!isFrameSetup(MI))
      SPAdj = -SPAdj;
    return SPAdj;
  }

  // To know whether a call adjusts the stack, we need information
  // that is bound to the following ADJCALLSTACKUP pseudo.
  // Look for the next ADJCALLSTACKUP that follows the call.
  if (MI.isCall()) {
    const MachineBasicBlock *MBB = MI.getParent();
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
  switch (MI.getOpcode()) {
  default:
    return 0;
  case X86::PUSH32i8:
  case X86::PUSH32r:
  case X86::PUSH32rmm:
  case X86::PUSH32rmr:
  case X86::PUSHi32:
    return 4;
  case X86::PUSH64i8:
  case X86::PUSH64r:
  case X86::PUSH64rmm:
  case X86::PUSH64rmr:
  case X86::PUSH64i32:
    return 8;
  }
}

/// Return true and the FrameIndex if the specified
/// operand and follow operands form a reference to the stack frame.
bool X86InstrInfo::isFrameOperand(const MachineInstr &MI, unsigned int Op,
                                  int &FrameIndex) const {
  if (MI.getOperand(Op + X86::AddrBaseReg).isFI() &&
      MI.getOperand(Op + X86::AddrScaleAmt).isImm() &&
      MI.getOperand(Op + X86::AddrIndexReg).isReg() &&
      MI.getOperand(Op + X86::AddrDisp).isImm() &&
      MI.getOperand(Op + X86::AddrScaleAmt).getImm() == 1 &&
      MI.getOperand(Op + X86::AddrIndexReg).getReg() == 0 &&
      MI.getOperand(Op + X86::AddrDisp).getImm() == 0) {
    FrameIndex = MI.getOperand(Op + X86::AddrBaseReg).getIndex();
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
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVUPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVUPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPDYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVDQUYrm:
  case X86::VMOVDQAYrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::VMOVSSZrm:
  case X86::VMOVSDZrm:
  case X86::VMOVAPSZrm:
  case X86::VMOVAPSZ128rm:
  case X86::VMOVAPSZ256rm:
  case X86::VMOVAPSZ128rm_NOVLX:
  case X86::VMOVAPSZ256rm_NOVLX:
  case X86::VMOVUPSZrm:
  case X86::VMOVUPSZ128rm:
  case X86::VMOVUPSZ256rm:
  case X86::VMOVUPSZ128rm_NOVLX:
  case X86::VMOVUPSZ256rm_NOVLX:
  case X86::VMOVAPDZrm:
  case X86::VMOVAPDZ128rm:
  case X86::VMOVAPDZ256rm:
  case X86::VMOVUPDZrm:
  case X86::VMOVUPDZ128rm:
  case X86::VMOVUPDZ256rm:
  case X86::VMOVDQA32Zrm:
  case X86::VMOVDQA32Z128rm:
  case X86::VMOVDQA32Z256rm:
  case X86::VMOVDQU32Zrm:
  case X86::VMOVDQU32Z128rm:
  case X86::VMOVDQU32Z256rm:
  case X86::VMOVDQA64Zrm:
  case X86::VMOVDQA64Z128rm:
  case X86::VMOVDQA64Z256rm:
  case X86::VMOVDQU64Zrm:
  case X86::VMOVDQU64Z128rm:
  case X86::VMOVDQU64Z256rm:
  case X86::VMOVDQU8Zrm:
  case X86::VMOVDQU8Z128rm:
  case X86::VMOVDQU8Z256rm:
  case X86::VMOVDQU16Zrm:
  case X86::VMOVDQU16Z128rm:
  case X86::VMOVDQU16Z256rm:
  case X86::KMOVBkm:
  case X86::KMOVWkm:
  case X86::KMOVDkm:
  case X86::KMOVQkm:
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
  case X86::MOVUPSmr:
  case X86::MOVAPDmr:
  case X86::MOVUPDmr:
  case X86::MOVDQAmr:
  case X86::MOVDQUmr:
  case X86::VMOVSSmr:
  case X86::VMOVSDmr:
  case X86::VMOVAPSmr:
  case X86::VMOVUPSmr:
  case X86::VMOVAPDmr:
  case X86::VMOVUPDmr:
  case X86::VMOVDQAmr:
  case X86::VMOVDQUmr:
  case X86::VMOVUPSYmr:
  case X86::VMOVAPSYmr:
  case X86::VMOVUPDYmr:
  case X86::VMOVAPDYmr:
  case X86::VMOVDQUYmr:
  case X86::VMOVDQAYmr:
  case X86::VMOVSSZmr:
  case X86::VMOVSDZmr:
  case X86::VMOVUPSZmr:
  case X86::VMOVUPSZ128mr:
  case X86::VMOVUPSZ256mr:
  case X86::VMOVUPSZ128mr_NOVLX:
  case X86::VMOVUPSZ256mr_NOVLX:
  case X86::VMOVAPSZmr:
  case X86::VMOVAPSZ128mr:
  case X86::VMOVAPSZ256mr:
  case X86::VMOVAPSZ128mr_NOVLX:
  case X86::VMOVAPSZ256mr_NOVLX:
  case X86::VMOVUPDZmr:
  case X86::VMOVUPDZ128mr:
  case X86::VMOVUPDZ256mr:
  case X86::VMOVAPDZmr:
  case X86::VMOVAPDZ128mr:
  case X86::VMOVAPDZ256mr:
  case X86::VMOVDQA32Zmr:
  case X86::VMOVDQA32Z128mr:
  case X86::VMOVDQA32Z256mr:
  case X86::VMOVDQU32Zmr:
  case X86::VMOVDQU32Z128mr:
  case X86::VMOVDQU32Z256mr:
  case X86::VMOVDQA64Zmr:
  case X86::VMOVDQA64Z128mr:
  case X86::VMOVDQA64Z256mr:
  case X86::VMOVDQU64Zmr:
  case X86::VMOVDQU64Z128mr:
  case X86::VMOVDQU64Z256mr:
  case X86::VMOVDQU8Zmr:
  case X86::VMOVDQU8Z128mr:
  case X86::VMOVDQU8Z256mr:
  case X86::VMOVDQU16Zmr:
  case X86::VMOVDQU16Z128mr:
  case X86::VMOVDQU16Z256mr:
  case X86::MMX_MOVD64mr:
  case X86::MMX_MOVQ64mr:
  case X86::MMX_MOVNTQmr:
  case X86::KMOVBmk:
  case X86::KMOVWmk:
  case X86::KMOVDmk:
  case X86::KMOVQmk:
    return true;
  }
  return false;
}

unsigned X86InstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  if (isFrameLoadOpcode(MI.getOpcode()))
    if (MI.getOperand(0).getSubReg() == 0 && isFrameOperand(MI, 1, FrameIndex))
      return MI.getOperand(0).getReg();
  return 0;
}

unsigned X86InstrInfo::isLoadFromStackSlotPostFE(const MachineInstr &MI,
                                                 int &FrameIndex) const {
  if (isFrameLoadOpcode(MI.getOpcode())) {
    unsigned Reg;
    if ((Reg = isLoadFromStackSlot(MI, FrameIndex)))
      return Reg;
    // Check for post-frame index elimination operations
    const MachineMemOperand *Dummy;
    return hasLoadFromStackSlot(MI, Dummy, FrameIndex);
  }
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                          int &FrameIndex) const {
  if (isFrameStoreOpcode(MI.getOpcode()))
    if (MI.getOperand(X86::AddrNumOperands).getSubReg() == 0 &&
        isFrameOperand(MI, 0, FrameIndex))
      return MI.getOperand(X86::AddrNumOperands).getReg();
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlotPostFE(const MachineInstr &MI,
                                                int &FrameIndex) const {
  if (isFrameStoreOpcode(MI.getOpcode())) {
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

bool X86InstrInfo::isReallyTriviallyReMaterializable(const MachineInstr &MI,
                                                     AliasAnalysis *AA) const {
  switch (MI.getOpcode()) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV8rm_NOREX:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVUPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVUPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVUPDYrm:
  case X86::VMOVDQAYrm:
  case X86::VMOVDQUYrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  // AVX-512
  case X86::VMOVSSZrm:
  case X86::VMOVSDZrm:
  case X86::VMOVAPDZ128rm:
  case X86::VMOVAPDZ256rm:
  case X86::VMOVAPDZrm:
  case X86::VMOVAPSZ128rm:
  case X86::VMOVAPSZ256rm:
  case X86::VMOVAPSZ128rm_NOVLX:
  case X86::VMOVAPSZ256rm_NOVLX:
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
  case X86::VMOVUPDZ128rm:
  case X86::VMOVUPDZ256rm:
  case X86::VMOVUPDZrm:
  case X86::VMOVUPSZ128rm:
  case X86::VMOVUPSZ256rm:
  case X86::VMOVUPSZ128rm_NOVLX:
  case X86::VMOVUPSZ256rm_NOVLX:
  case X86::VMOVUPSZrm: {
    // Loads from constant pools are trivially rematerializable.
    if (MI.getOperand(1 + X86::AddrBaseReg).isReg() &&
        MI.getOperand(1 + X86::AddrScaleAmt).isImm() &&
        MI.getOperand(1 + X86::AddrIndexReg).isReg() &&
        MI.getOperand(1 + X86::AddrIndexReg).getReg() == 0 &&
        MI.isDereferenceableInvariantLoad(AA)) {
      unsigned BaseReg = MI.getOperand(1 + X86::AddrBaseReg).getReg();
      if (BaseReg == 0 || BaseReg == X86::RIP)
        return true;
      // Allow re-materialization of PIC load.
      if (!ReMatPICStubLoad && MI.getOperand(1 + X86::AddrDisp).isGlobal())
        return false;
      const MachineFunction &MF = *MI.getParent()->getParent();
      const MachineRegisterInfo &MRI = MF.getRegInfo();
      return regIsPICBase(BaseReg, MRI);
    }
    return false;
  }

  case X86::LEA32r:
  case X86::LEA64r: {
    if (MI.getOperand(1 + X86::AddrScaleAmt).isImm() &&
        MI.getOperand(1 + X86::AddrIndexReg).isReg() &&
        MI.getOperand(1 + X86::AddrIndexReg).getReg() == 0 &&
        !MI.getOperand(1 + X86::AddrDisp).isReg()) {
      // lea fi#, lea GV, etc. are all rematerializable.
      if (!MI.getOperand(1 + X86::AddrBaseReg).isReg())
        return true;
      unsigned BaseReg = MI.getOperand(1 + X86::AddrBaseReg).getReg();
      if (BaseReg == 0)
        return true;
      // Allow re-materialization of lea PICBase + x.
      const MachineFunction &MF = *MI.getParent()->getParent();
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
    for (MachineBasicBlock *S : MBB.successors())
      if (S->isLiveIn(X86::EFLAGS))
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
                                 const MachineInstr &Orig,
                                 const TargetRegisterInfo &TRI) const {
  bool ClobbersEFLAGS = false;
  for (const MachineOperand &MO : Orig.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == X86::EFLAGS) {
      ClobbersEFLAGS = true;
      break;
    }
  }

  if (ClobbersEFLAGS && !isSafeToClobberEFLAGS(MBB, I)) {
    // The instruction clobbers EFLAGS. Re-materialize as MOV32ri to avoid side
    // effects.
    int Value;
    switch (Orig.getOpcode()) {
    case X86::MOV32r0:  Value = 0; break;
    case X86::MOV32r1:  Value = 1; break;
    case X86::MOV32r_1: Value = -1; break;
    default:
      llvm_unreachable("Unexpected instruction!");
    }

    const DebugLoc &DL = Orig.getDebugLoc();
    BuildMI(MBB, I, DL, get(X86::MOV32ri))
        .add(Orig.getOperand(0))
        .addImm(Value);
  } else {
    MachineInstr *MI = MBB.getParent()->CloneMachineInstr(&Orig);
    MBB.insert(I, MI);
  }

  MachineInstr &NewMI = *std::prev(I);
  NewMI.substituteRegister(Orig.getOperand(0).getReg(), DestReg, SubIdx, TRI);
}

/// True if MI has a condition code def, e.g. EFLAGS, that is not marked dead.
bool X86InstrInfo::hasLiveCondCodeDef(MachineInstr &MI) const {
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        MO.getReg() == X86::EFLAGS && !MO.isDead()) {
      return true;
    }
  }
  return false;
}

/// Check whether the shift count for a machine operand is non-zero.
inline static unsigned getTruncatedShiftCount(MachineInstr &MI,
                                              unsigned ShiftAmtOperandIdx) {
  // The shift count is six bits with the REX.W prefix and five bits without.
  unsigned ShiftCountMask = (MI.getDesc().TSFlags & X86II::REX_W) ? 63 : 31;
  unsigned Imm = MI.getOperand(ShiftAmtOperandIdx).getImm();
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

bool X86InstrInfo::classifyLEAReg(MachineInstr &MI, const MachineOperand &Src,
                                  unsigned Opc, bool AllowSP, unsigned &NewSrc,
                                  bool &isKill, bool &isUndef,
                                  MachineOperand &ImplicitOp,
                                  LiveVariables *LV) const {
  MachineFunction &MF = *MI.getParent()->getParent();
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

    NewSrc = getX86SubSuperRegister(Src.getReg(), 64);
    isKill = Src.isKill();
    isUndef = Src.isUndef();
  } else {
    // Virtual register of the wrong class, we have to create a temporary 64-bit
    // vreg to feed into the LEA.
    NewSrc = MF.getRegInfo().createVirtualRegister(RC);
    MachineInstr *Copy =
        BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), get(TargetOpcode::COPY))
            .addReg(NewSrc, RegState::Define | RegState::Undef, X86::sub_32bit)
            .add(Src);

    // Which is obviously going to be dead after we're done with it.
    isKill = true;
    isUndef = false;

    if (LV)
      LV->replaceKillInstruction(SrcReg, MI, *Copy);
  }

  // We've set all the parameters without issue.
  return true;
}

/// Helper for convertToThreeAddress when 16-bit LEA is disabled, use 32-bit
/// LEA to form 3-address code by promoting to a 32-bit superregister and then
/// truncating back down to a 16-bit subregister.
MachineInstr *X86InstrInfo::convertToThreeAddressWithLEA(
    unsigned MIOpc, MachineFunction::iterator &MFI, MachineInstr &MI,
    LiveVariables *LV) const {
  MachineBasicBlock::iterator MBBI = MI.getIterator();
  unsigned Dest = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();
  bool isDead = MI.getOperand(0).isDead();
  bool isKill = MI.getOperand(1).isKill();

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
  BuildMI(*MFI, MBBI, MI.getDebugLoc(), get(X86::IMPLICIT_DEF), leaInReg);
  MachineInstr *InsMI =
      BuildMI(*MFI, MBBI, MI.getDebugLoc(), get(TargetOpcode::COPY))
          .addReg(leaInReg, RegState::Define, X86::sub_16bit)
          .addReg(Src, getKillRegState(isKill));

  MachineInstrBuilder MIB =
      BuildMI(*MFI, MBBI, MI.getDebugLoc(), get(Opc), leaOutReg);
  switch (MIOpc) {
  default: llvm_unreachable("Unreachable!");
  case X86::SHL16ri: {
    unsigned ShAmt = MI.getOperand(2).getImm();
    MIB.addReg(0).addImm(1ULL << ShAmt)
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
    addRegOffset(MIB, leaInReg, true, MI.getOperand(2).getImm());
    break;
  case X86::ADD16rr:
  case X86::ADD16rr_DB: {
    unsigned Src2 = MI.getOperand(2).getReg();
    bool isKill2 = MI.getOperand(2).isKill();
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
      BuildMI(*MFI, &*MIB, MI.getDebugLoc(), get(X86::IMPLICIT_DEF), leaInReg2);
      InsMI2 = BuildMI(*MFI, &*MIB, MI.getDebugLoc(), get(TargetOpcode::COPY))
                   .addReg(leaInReg2, RegState::Define, X86::sub_16bit)
                   .addReg(Src2, getKillRegState(isKill2));
      addRegReg(MIB, leaInReg, true, leaInReg2, true);
    }
    if (LV && isKill2 && InsMI2)
      LV->replaceKillInstruction(Src2, MI, *InsMI2);
    break;
  }
  }

  MachineInstr *NewMI = MIB;
  MachineInstr *ExtMI =
      BuildMI(*MFI, MBBI, MI.getDebugLoc(), get(TargetOpcode::COPY))
          .addReg(Dest, RegState::Define | getDeadRegState(isDead))
          .addReg(leaOutReg, RegState::Kill, X86::sub_16bit);

  if (LV) {
    // Update live variables
    LV->getVarInfo(leaInReg).Kills.push_back(NewMI);
    LV->getVarInfo(leaOutReg).Kills.push_back(ExtMI);
    if (isKill)
      LV->replaceKillInstruction(Src, MI, *InsMI);
    if (isDead)
      LV->replaceKillInstruction(Dest, MI, *ExtMI);
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
                                    MachineInstr &MI, LiveVariables *LV) const {
  // The following opcodes also sets the condition code register(s). Only
  // convert them to equivalent lea if the condition code register def's
  // are dead!
  if (hasLiveCondCodeDef(MI))
    return nullptr;

  MachineFunction &MF = *MI.getParent()->getParent();
  // All instructions input are two-addr instructions.  Get the known operands.
  const MachineOperand &Dest = MI.getOperand(0);
  const MachineOperand &Src = MI.getOperand(1);

  MachineInstr *NewMI = nullptr;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  // 16-bit LEA is also slow on Core2.
  bool DisableLEA16 = true;
  bool is64Bit = Subtarget.is64Bit();

  unsigned MIOpc = MI.getOpcode();
  switch (MIOpc) {
  default: return nullptr;
  case X86::SHL64ri: {
    assert(MI.getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return nullptr;

    // LEA can't handle RSP.
    if (TargetRegisterInfo::isVirtualRegister(Src.getReg()) &&
        !MF.getRegInfo().constrainRegClass(Src.getReg(),
                                           &X86::GR64_NOSPRegClass))
      return nullptr;

    NewMI = BuildMI(MF, MI.getDebugLoc(), get(X86::LEA64r))
                .add(Dest)
                .addReg(0)
                .addImm(1ULL << ShAmt)
                .add(Src)
                .addImm(0)
                .addReg(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI.getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return nullptr;

    unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    // LEA can't handle ESP.
    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp, LV))
      return nullptr;

    MachineInstrBuilder MIB =
        BuildMI(MF, MI.getDebugLoc(), get(Opc))
            .add(Dest)
            .addReg(0)
            .addImm(1ULL << ShAmt)
            .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef))
            .addImm(0)
            .addReg(0);
    if (ImplicitOp.getReg() != 0)
      MIB.add(ImplicitOp);
    NewMI = MIB;

    break;
  }
  case X86::SHL16ri: {
    assert(MI.getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return nullptr;

    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MI, LV)
                     : nullptr;
    NewMI = BuildMI(MF, MI.getDebugLoc(), get(X86::LEA16r))
                .add(Dest)
                .addReg(0)
                .addImm(1ULL << ShAmt)
                .add(Src)
                .addImm(0)
                .addReg(0);
    break;
  }
  case X86::INC64r:
  case X86::INC32r: {
    assert(MI.getNumOperands() >= 2 && "Unknown inc instruction!");
    unsigned Opc = MIOpc == X86::INC64r ? X86::LEA64r
      : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp, LV))
      return nullptr;

    MachineInstrBuilder MIB =
        BuildMI(MF, MI.getDebugLoc(), get(Opc))
            .add(Dest)
            .addReg(SrcReg,
                    getKillRegState(isKill) | getUndefRegState(isUndef));
    if (ImplicitOp.getReg() != 0)
      MIB.add(ImplicitOp);

    NewMI = addOffset(MIB, 1);
    break;
  }
  case X86::INC16r:
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MI, LV)
                     : nullptr;
    assert(MI.getNumOperands() >= 2 && "Unknown inc instruction!");
    NewMI = addOffset(
        BuildMI(MF, MI.getDebugLoc(), get(X86::LEA16r)).add(Dest).add(Src), 1);
    break;
  case X86::DEC64r:
  case X86::DEC32r: {
    assert(MI.getNumOperands() >= 2 && "Unknown dec instruction!");
    unsigned Opc = MIOpc == X86::DEC64r ? X86::LEA64r
      : (is64Bit ? X86::LEA64_32r : X86::LEA32r);

    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp, LV))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI.getDebugLoc(), get(Opc))
                                  .add(Dest)
                                  .addReg(SrcReg, getUndefRegState(isUndef) |
                                                      getKillRegState(isKill));
    if (ImplicitOp.getReg() != 0)
      MIB.add(ImplicitOp);

    NewMI = addOffset(MIB, -1);

    break;
  }
  case X86::DEC16r:
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MI, LV)
                     : nullptr;
    assert(MI.getNumOperands() >= 2 && "Unknown dec instruction!");
    NewMI = addOffset(
        BuildMI(MF, MI.getDebugLoc(), get(X86::LEA16r)).add(Dest).add(Src), -1);
    break;
  case X86::ADD64rr:
  case X86::ADD64rr_DB:
  case X86::ADD32rr:
  case X86::ADD32rr_DB: {
    assert(MI.getNumOperands() >= 3 && "Unknown add instruction!");
    unsigned Opc;
    if (MIOpc == X86::ADD64rr || MIOpc == X86::ADD64rr_DB)
      Opc = X86::LEA64r;
    else
      Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ true,
                        SrcReg, isKill, isUndef, ImplicitOp, LV))
      return nullptr;

    const MachineOperand &Src2 = MI.getOperand(2);
    bool isKill2, isUndef2;
    unsigned SrcReg2;
    MachineOperand ImplicitOp2 = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src2, Opc, /*AllowSP=*/ false,
                        SrcReg2, isKill2, isUndef2, ImplicitOp2, LV))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI.getDebugLoc(), get(Opc)).add(Dest);
    if (ImplicitOp.getReg() != 0)
      MIB.add(ImplicitOp);
    if (ImplicitOp2.getReg() != 0)
      MIB.add(ImplicitOp2);

    NewMI = addRegReg(MIB, SrcReg, isKill, SrcReg2, isKill2);

    // Preserve undefness of the operands.
    NewMI->getOperand(1).setIsUndef(isUndef);
    NewMI->getOperand(3).setIsUndef(isUndef2);

    if (LV && Src2.isKill())
      LV->replaceKillInstruction(SrcReg2, MI, *NewMI);
    break;
  }
  case X86::ADD16rr:
  case X86::ADD16rr_DB: {
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MI, LV)
                     : nullptr;
    assert(MI.getNumOperands() >= 3 && "Unknown add instruction!");
    unsigned Src2 = MI.getOperand(2).getReg();
    bool isKill2 = MI.getOperand(2).isKill();
    NewMI = addRegReg(BuildMI(MF, MI.getDebugLoc(), get(X86::LEA16r)).add(Dest),
                      Src.getReg(), Src.isKill(), Src2, isKill2);

    // Preserve undefness of the operands.
    bool isUndef = MI.getOperand(1).isUndef();
    bool isUndef2 = MI.getOperand(2).isUndef();
    NewMI->getOperand(1).setIsUndef(isUndef);
    NewMI->getOperand(3).setIsUndef(isUndef2);

    if (LV && isKill2)
      LV->replaceKillInstruction(Src2, MI, *NewMI);
    break;
  }
  case X86::ADD64ri32:
  case X86::ADD64ri8:
  case X86::ADD64ri32_DB:
  case X86::ADD64ri8_DB:
    assert(MI.getNumOperands() >= 3 && "Unknown add instruction!");
    NewMI = addOffset(
        BuildMI(MF, MI.getDebugLoc(), get(X86::LEA64r)).add(Dest).add(Src),
        MI.getOperand(2));
    break;
  case X86::ADD32ri:
  case X86::ADD32ri8:
  case X86::ADD32ri_DB:
  case X86::ADD32ri8_DB: {
    assert(MI.getNumOperands() >= 3 && "Unknown add instruction!");
    unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ true,
                        SrcReg, isKill, isUndef, ImplicitOp, LV))
      return nullptr;

    MachineInstrBuilder MIB = BuildMI(MF, MI.getDebugLoc(), get(Opc))
                                  .add(Dest)
                                  .addReg(SrcReg, getUndefRegState(isUndef) |
                                                      getKillRegState(isKill));
    if (ImplicitOp.getReg() != 0)
      MIB.add(ImplicitOp);

    NewMI = addOffset(MIB, MI.getOperand(2));
    break;
  }
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri_DB:
  case X86::ADD16ri8_DB:
    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MI, LV)
                     : nullptr;
    assert(MI.getNumOperands() >= 3 && "Unknown add instruction!");
    NewMI = addOffset(
        BuildMI(MF, MI.getDebugLoc(), get(X86::LEA16r)).add(Dest).add(Src),
        MI.getOperand(2));
    break;

  case X86::VMOVDQU8Z128rmk:
  case X86::VMOVDQU8Z256rmk:
  case X86::VMOVDQU8Zrmk:
  case X86::VMOVDQU16Z128rmk:
  case X86::VMOVDQU16Z256rmk:
  case X86::VMOVDQU16Zrmk:
  case X86::VMOVDQU32Z128rmk: case X86::VMOVDQA32Z128rmk:
  case X86::VMOVDQU32Z256rmk: case X86::VMOVDQA32Z256rmk:
  case X86::VMOVDQU32Zrmk:    case X86::VMOVDQA32Zrmk:
  case X86::VMOVDQU64Z128rmk: case X86::VMOVDQA64Z128rmk:
  case X86::VMOVDQU64Z256rmk: case X86::VMOVDQA64Z256rmk:
  case X86::VMOVDQU64Zrmk:    case X86::VMOVDQA64Zrmk:
  case X86::VMOVUPDZ128rmk:   case X86::VMOVAPDZ128rmk:
  case X86::VMOVUPDZ256rmk:   case X86::VMOVAPDZ256rmk:
  case X86::VMOVUPDZrmk:      case X86::VMOVAPDZrmk:
  case X86::VMOVUPSZ128rmk:   case X86::VMOVAPSZ128rmk:
  case X86::VMOVUPSZ256rmk:   case X86::VMOVAPSZ256rmk:
  case X86::VMOVUPSZrmk:      case X86::VMOVAPSZrmk: {
    unsigned Opc;
    switch (MIOpc) {
    default: llvm_unreachable("Unreachable!");
    case X86::VMOVDQU8Z128rmk:  Opc = X86::VPBLENDMBZ128rmk; break;
    case X86::VMOVDQU8Z256rmk:  Opc = X86::VPBLENDMBZ256rmk; break;
    case X86::VMOVDQU8Zrmk:     Opc = X86::VPBLENDMBZrmk;    break;
    case X86::VMOVDQU16Z128rmk: Opc = X86::VPBLENDMWZ128rmk; break;
    case X86::VMOVDQU16Z256rmk: Opc = X86::VPBLENDMWZ256rmk; break;
    case X86::VMOVDQU16Zrmk:    Opc = X86::VPBLENDMWZrmk;    break;
    case X86::VMOVDQU32Z128rmk: Opc = X86::VPBLENDMDZ128rmk; break;
    case X86::VMOVDQU32Z256rmk: Opc = X86::VPBLENDMDZ256rmk; break;
    case X86::VMOVDQU32Zrmk:    Opc = X86::VPBLENDMDZrmk;    break;
    case X86::VMOVDQU64Z128rmk: Opc = X86::VPBLENDMQZ128rmk; break;
    case X86::VMOVDQU64Z256rmk: Opc = X86::VPBLENDMQZ256rmk; break;
    case X86::VMOVDQU64Zrmk:    Opc = X86::VPBLENDMQZrmk;    break;
    case X86::VMOVUPDZ128rmk:   Opc = X86::VBLENDMPDZ128rmk; break;
    case X86::VMOVUPDZ256rmk:   Opc = X86::VBLENDMPDZ256rmk; break;
    case X86::VMOVUPDZrmk:      Opc = X86::VBLENDMPDZrmk;    break;
    case X86::VMOVUPSZ128rmk:   Opc = X86::VBLENDMPSZ128rmk; break;
    case X86::VMOVUPSZ256rmk:   Opc = X86::VBLENDMPSZ256rmk; break;
    case X86::VMOVUPSZrmk:      Opc = X86::VBLENDMPSZrmk;    break;
    case X86::VMOVDQA32Z128rmk: Opc = X86::VPBLENDMDZ128rmk; break;
    case X86::VMOVDQA32Z256rmk: Opc = X86::VPBLENDMDZ256rmk; break;
    case X86::VMOVDQA32Zrmk:    Opc = X86::VPBLENDMDZrmk;    break;
    case X86::VMOVDQA64Z128rmk: Opc = X86::VPBLENDMQZ128rmk; break;
    case X86::VMOVDQA64Z256rmk: Opc = X86::VPBLENDMQZ256rmk; break;
    case X86::VMOVDQA64Zrmk:    Opc = X86::VPBLENDMQZrmk;    break;
    case X86::VMOVAPDZ128rmk:   Opc = X86::VBLENDMPDZ128rmk; break;
    case X86::VMOVAPDZ256rmk:   Opc = X86::VBLENDMPDZ256rmk; break;
    case X86::VMOVAPDZrmk:      Opc = X86::VBLENDMPDZrmk;    break;
    case X86::VMOVAPSZ128rmk:   Opc = X86::VBLENDMPSZ128rmk; break;
    case X86::VMOVAPSZ256rmk:   Opc = X86::VBLENDMPSZ256rmk; break;
    case X86::VMOVAPSZrmk:      Opc = X86::VBLENDMPSZrmk;    break;
    }

    NewMI = BuildMI(MF, MI.getDebugLoc(), get(Opc))
              .add(Dest)
              .add(MI.getOperand(2))
              .add(Src)
              .add(MI.getOperand(3))
              .add(MI.getOperand(4))
              .add(MI.getOperand(5))
              .add(MI.getOperand(6))
              .add(MI.getOperand(7));
    break;
  }
  case X86::VMOVDQU8Z128rrk:
  case X86::VMOVDQU8Z256rrk:
  case X86::VMOVDQU8Zrrk:
  case X86::VMOVDQU16Z128rrk:
  case X86::VMOVDQU16Z256rrk:
  case X86::VMOVDQU16Zrrk:
  case X86::VMOVDQU32Z128rrk: case X86::VMOVDQA32Z128rrk:
  case X86::VMOVDQU32Z256rrk: case X86::VMOVDQA32Z256rrk:
  case X86::VMOVDQU32Zrrk:    case X86::VMOVDQA32Zrrk:
  case X86::VMOVDQU64Z128rrk: case X86::VMOVDQA64Z128rrk:
  case X86::VMOVDQU64Z256rrk: case X86::VMOVDQA64Z256rrk:
  case X86::VMOVDQU64Zrrk:    case X86::VMOVDQA64Zrrk:
  case X86::VMOVUPDZ128rrk:   case X86::VMOVAPDZ128rrk:
  case X86::VMOVUPDZ256rrk:   case X86::VMOVAPDZ256rrk:
  case X86::VMOVUPDZrrk:      case X86::VMOVAPDZrrk:
  case X86::VMOVUPSZ128rrk:   case X86::VMOVAPSZ128rrk:
  case X86::VMOVUPSZ256rrk:   case X86::VMOVAPSZ256rrk:
  case X86::VMOVUPSZrrk:      case X86::VMOVAPSZrrk: {
    unsigned Opc;
    switch (MIOpc) {
    default: llvm_unreachable("Unreachable!");
    case X86::VMOVDQU8Z128rrk:  Opc = X86::VPBLENDMBZ128rrk; break;
    case X86::VMOVDQU8Z256rrk:  Opc = X86::VPBLENDMBZ256rrk; break;
    case X86::VMOVDQU8Zrrk:     Opc = X86::VPBLENDMBZrrk;    break;
    case X86::VMOVDQU16Z128rrk: Opc = X86::VPBLENDMWZ128rrk; break;
    case X86::VMOVDQU16Z256rrk: Opc = X86::VPBLENDMWZ256rrk; break;
    case X86::VMOVDQU16Zrrk:    Opc = X86::VPBLENDMWZrrk;    break;
    case X86::VMOVDQU32Z128rrk: Opc = X86::VPBLENDMDZ128rrk; break;
    case X86::VMOVDQU32Z256rrk: Opc = X86::VPBLENDMDZ256rrk; break;
    case X86::VMOVDQU32Zrrk:    Opc = X86::VPBLENDMDZrrk;    break;
    case X86::VMOVDQU64Z128rrk: Opc = X86::VPBLENDMQZ128rrk; break;
    case X86::VMOVDQU64Z256rrk: Opc = X86::VPBLENDMQZ256rrk; break;
    case X86::VMOVDQU64Zrrk:    Opc = X86::VPBLENDMQZrrk;    break;
    case X86::VMOVUPDZ128rrk:   Opc = X86::VBLENDMPDZ128rrk; break;
    case X86::VMOVUPDZ256rrk:   Opc = X86::VBLENDMPDZ256rrk; break;
    case X86::VMOVUPDZrrk:      Opc = X86::VBLENDMPDZrrk;    break;
    case X86::VMOVUPSZ128rrk:   Opc = X86::VBLENDMPSZ128rrk; break;
    case X86::VMOVUPSZ256rrk:   Opc = X86::VBLENDMPSZ256rrk; break;
    case X86::VMOVUPSZrrk:      Opc = X86::VBLENDMPSZrrk;    break;
    case X86::VMOVDQA32Z128rrk: Opc = X86::VPBLENDMDZ128rrk; break;
    case X86::VMOVDQA32Z256rrk: Opc = X86::VPBLENDMDZ256rrk; break;
    case X86::VMOVDQA32Zrrk:    Opc = X86::VPBLENDMDZrrk;    break;
    case X86::VMOVDQA64Z128rrk: Opc = X86::VPBLENDMQZ128rrk; break;
    case X86::VMOVDQA64Z256rrk: Opc = X86::VPBLENDMQZ256rrk; break;
    case X86::VMOVDQA64Zrrk:    Opc = X86::VPBLENDMQZrrk;    break;
    case X86::VMOVAPDZ128rrk:   Opc = X86::VBLENDMPDZ128rrk; break;
    case X86::VMOVAPDZ256rrk:   Opc = X86::VBLENDMPDZ256rrk; break;
    case X86::VMOVAPDZrrk:      Opc = X86::VBLENDMPDZrrk;    break;
    case X86::VMOVAPSZ128rrk:   Opc = X86::VBLENDMPSZ128rrk; break;
    case X86::VMOVAPSZ256rrk:   Opc = X86::VBLENDMPSZ256rrk; break;
    case X86::VMOVAPSZrrk:      Opc = X86::VBLENDMPSZrrk;    break;
    }

    NewMI = BuildMI(MF, MI.getDebugLoc(), get(Opc))
              .add(Dest)
              .add(MI.getOperand(2))
              .add(Src)
              .add(MI.getOperand(3));
    break;
  }
  }

  if (!NewMI) return nullptr;

  if (LV) {  // Update live variables
    if (Src.isKill())
      LV->replaceKillInstruction(Src.getReg(), MI, *NewMI);
    if (Dest.isDead())
      LV->replaceKillInstruction(Dest.getReg(), MI, *NewMI);
  }

  MFI->insert(MI.getIterator(), NewMI); // Insert the new inst
  return NewMI;
}

/// This determines which of three possible cases of a three source commute
/// the source indexes correspond to taking into account any mask operands.
/// All prevents commuting a passthru operand. Returns -1 if the commute isn't
/// possible.
/// Case 0 - Possible to commute the first and second operands.
/// Case 1 - Possible to commute the first and third operands.
/// Case 2 - Possible to commute the second and third operands.
static int getThreeSrcCommuteCase(uint64_t TSFlags, unsigned SrcOpIdx1,
                                  unsigned SrcOpIdx2) {
  // Put the lowest index to SrcOpIdx1 to simplify the checks below.
  if (SrcOpIdx1 > SrcOpIdx2)
    std::swap(SrcOpIdx1, SrcOpIdx2);

  unsigned Op1 = 1, Op2 = 2, Op3 = 3;
  if (X86II::isKMasked(TSFlags)) {
    // The k-mask operand cannot be commuted.
    if (SrcOpIdx1 == 2)
      return -1;

    // For k-zero-masked operations it is Ok to commute the first vector
    // operand.
    // For regular k-masked operations a conservative choice is done as the
    // elements of the first vector operand, for which the corresponding bit
    // in the k-mask operand is set to 0, are copied to the result of the
    // instruction.
    // TODO/FIXME: The commute still may be legal if it is known that the
    // k-mask operand is set to either all ones or all zeroes.
    // It is also Ok to commute the 1st operand if all users of MI use only
    // the elements enabled by the k-mask operand. For example,
    //   v4 = VFMADD213PSZrk v1, k, v2, v3; // v1[i] = k[i] ? v2[i]*v1[i]+v3[i]
    //                                                     : v1[i];
    //   VMOVAPSZmrk <mem_addr>, k, v4; // this is the ONLY user of v4 ->
    //                                  // Ok, to commute v1 in FMADD213PSZrk.
    if (X86II::isKMergeMasked(TSFlags) && SrcOpIdx1 == Op1)
      return -1;
    Op2++;
    Op3++;
  }

  if (SrcOpIdx1 == Op1 && SrcOpIdx2 == Op2)
    return 0;
  if (SrcOpIdx1 == Op1 && SrcOpIdx2 == Op3)
    return 1;
  if (SrcOpIdx1 == Op2 && SrcOpIdx2 == Op3)
    return 2;
  return -1;
}

unsigned X86InstrInfo::getFMA3OpcodeToCommuteOperands(
    const MachineInstr &MI, unsigned SrcOpIdx1, unsigned SrcOpIdx2,
    const X86InstrFMA3Group &FMA3Group) const {

  unsigned Opc = MI.getOpcode();

  // Put the lowest index to SrcOpIdx1 to simplify the checks below.
  if (SrcOpIdx1 > SrcOpIdx2)
    std::swap(SrcOpIdx1, SrcOpIdx2);

  // TODO: Commuting the 1st operand of FMA*_Int requires some additional
  // analysis. The commute optimization is legal only if all users of FMA*_Int
  // use only the lowest element of the FMA*_Int instruction. Such analysis are
  // not implemented yet. So, just return 0 in that case.
  // When such analysis are available this place will be the right place for
  // calling it.
  if (FMA3Group.isIntrinsic() && SrcOpIdx1 == 1)
    return 0;

  // Determine which case this commute is or if it can't be done.
  int Case = getThreeSrcCommuteCase(MI.getDesc().TSFlags, SrcOpIdx1, SrcOpIdx2);
  if (Case < 0)
    return 0;

  // Define the FMA forms mapping array that helps to map input FMA form
  // to output FMA form to preserve the operation semantics after
  // commuting the operands.
  const unsigned Form132Index = 0;
  const unsigned Form213Index = 1;
  const unsigned Form231Index = 2;
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

  unsigned FMAForms[3];
  if (FMA3Group.isRegOpcodeFromGroup(Opc)) {
    FMAForms[0] = FMA3Group.getReg132Opcode();
    FMAForms[1] = FMA3Group.getReg213Opcode();
    FMAForms[2] = FMA3Group.getReg231Opcode();
  } else {
    FMAForms[0] = FMA3Group.getMem132Opcode();
    FMAForms[1] = FMA3Group.getMem213Opcode();
    FMAForms[2] = FMA3Group.getMem231Opcode();
  }
  unsigned FormIndex;
  for (FormIndex = 0; FormIndex < 3; FormIndex++)
    if (Opc == FMAForms[FormIndex])
      break;

  // Everything is ready, just adjust the FMA opcode and return it.
  FormIndex = FormMapping[Case][FormIndex];
  return FMAForms[FormIndex];
}

static bool commuteVPTERNLOG(MachineInstr &MI, unsigned SrcOpIdx1,
                             unsigned SrcOpIdx2) {
  uint64_t TSFlags = MI.getDesc().TSFlags;

  // Determine which case this commute is or if it can't be done.
  int Case = getThreeSrcCommuteCase(TSFlags, SrcOpIdx1, SrcOpIdx2);
  if (Case < 0)
    return false;

  // For each case we need to swap two pairs of bits in the final immediate.
  static const uint8_t SwapMasks[3][4] = {
    { 0x04, 0x10, 0x08, 0x20 }, // Swap bits 2/4 and 3/5.
    { 0x02, 0x10, 0x08, 0x40 }, // Swap bits 1/4 and 3/6.
    { 0x02, 0x04, 0x20, 0x40 }, // Swap bits 1/2 and 5/6.
  };

  uint8_t Imm = MI.getOperand(MI.getNumOperands()-1).getImm();
  // Clear out the bits we are swapping.
  uint8_t NewImm = Imm & ~(SwapMasks[Case][0] | SwapMasks[Case][1] |
                           SwapMasks[Case][2] | SwapMasks[Case][3]);
  // If the immediate had a bit of the pair set, then set the opposite bit.
  if (Imm & SwapMasks[Case][0]) NewImm |= SwapMasks[Case][1];
  if (Imm & SwapMasks[Case][1]) NewImm |= SwapMasks[Case][0];
  if (Imm & SwapMasks[Case][2]) NewImm |= SwapMasks[Case][3];
  if (Imm & SwapMasks[Case][3]) NewImm |= SwapMasks[Case][2];
  MI.getOperand(MI.getNumOperands()-1).setImm(NewImm);

  return true;
}

// Returns true if this is a VPERMI2 or VPERMT2 instrution that can be
// commuted.
static bool isCommutableVPERMV3Instruction(unsigned Opcode) {
#define VPERM_CASES(Suffix) \
  case X86::VPERMI2##Suffix##128rr:    case X86::VPERMT2##Suffix##128rr:    \
  case X86::VPERMI2##Suffix##256rr:    case X86::VPERMT2##Suffix##256rr:    \
  case X86::VPERMI2##Suffix##rr:       case X86::VPERMT2##Suffix##rr:       \
  case X86::VPERMI2##Suffix##128rm:    case X86::VPERMT2##Suffix##128rm:    \
  case X86::VPERMI2##Suffix##256rm:    case X86::VPERMT2##Suffix##256rm:    \
  case X86::VPERMI2##Suffix##rm:       case X86::VPERMT2##Suffix##rm:       \
  case X86::VPERMI2##Suffix##128rrkz:  case X86::VPERMT2##Suffix##128rrkz:  \
  case X86::VPERMI2##Suffix##256rrkz:  case X86::VPERMT2##Suffix##256rrkz:  \
  case X86::VPERMI2##Suffix##rrkz:     case X86::VPERMT2##Suffix##rrkz:     \
  case X86::VPERMI2##Suffix##128rmkz:  case X86::VPERMT2##Suffix##128rmkz:  \
  case X86::VPERMI2##Suffix##256rmkz:  case X86::VPERMT2##Suffix##256rmkz:  \
  case X86::VPERMI2##Suffix##rmkz:     case X86::VPERMT2##Suffix##rmkz:

#define VPERM_CASES_BROADCAST(Suffix) \
  VPERM_CASES(Suffix) \
  case X86::VPERMI2##Suffix##128rmb:   case X86::VPERMT2##Suffix##128rmb:   \
  case X86::VPERMI2##Suffix##256rmb:   case X86::VPERMT2##Suffix##256rmb:   \
  case X86::VPERMI2##Suffix##rmb:      case X86::VPERMT2##Suffix##rmb:      \
  case X86::VPERMI2##Suffix##128rmbkz: case X86::VPERMT2##Suffix##128rmbkz: \
  case X86::VPERMI2##Suffix##256rmbkz: case X86::VPERMT2##Suffix##256rmbkz: \
  case X86::VPERMI2##Suffix##rmbkz:    case X86::VPERMT2##Suffix##rmbkz:

  switch (Opcode) {
  default: return false;
  VPERM_CASES(B)
  VPERM_CASES_BROADCAST(D)
  VPERM_CASES_BROADCAST(PD)
  VPERM_CASES_BROADCAST(PS)
  VPERM_CASES_BROADCAST(Q)
  VPERM_CASES(W)
    return true;
  }
#undef VPERM_CASES_BROADCAST
#undef VPERM_CASES
}

// Returns commuted opcode for VPERMI2 and VPERMT2 instructions by switching
// from the I opcod to the T opcode and vice versa.
static unsigned getCommutedVPERMV3Opcode(unsigned Opcode) {
#define VPERM_CASES(Orig, New) \
  case X86::Orig##128rr:    return X86::New##128rr;   \
  case X86::Orig##128rrkz:  return X86::New##128rrkz; \
  case X86::Orig##128rm:    return X86::New##128rm;   \
  case X86::Orig##128rmkz:  return X86::New##128rmkz; \
  case X86::Orig##256rr:    return X86::New##256rr;   \
  case X86::Orig##256rrkz:  return X86::New##256rrkz; \
  case X86::Orig##256rm:    return X86::New##256rm;   \
  case X86::Orig##256rmkz:  return X86::New##256rmkz; \
  case X86::Orig##rr:       return X86::New##rr;      \
  case X86::Orig##rrkz:     return X86::New##rrkz;    \
  case X86::Orig##rm:       return X86::New##rm;      \
  case X86::Orig##rmkz:     return X86::New##rmkz;

#define VPERM_CASES_BROADCAST(Orig, New) \
  VPERM_CASES(Orig, New) \
  case X86::Orig##128rmb:   return X86::New##128rmb;   \
  case X86::Orig##128rmbkz: return X86::New##128rmbkz; \
  case X86::Orig##256rmb:   return X86::New##256rmb;   \
  case X86::Orig##256rmbkz: return X86::New##256rmbkz; \
  case X86::Orig##rmb:      return X86::New##rmb;      \
  case X86::Orig##rmbkz:    return X86::New##rmbkz;

  switch (Opcode) {
  VPERM_CASES(VPERMI2B, VPERMT2B)
  VPERM_CASES_BROADCAST(VPERMI2D,  VPERMT2D)
  VPERM_CASES_BROADCAST(VPERMI2PD, VPERMT2PD)
  VPERM_CASES_BROADCAST(VPERMI2PS, VPERMT2PS)
  VPERM_CASES_BROADCAST(VPERMI2Q,  VPERMT2Q)
  VPERM_CASES(VPERMI2W, VPERMT2W)
  VPERM_CASES(VPERMT2B, VPERMI2B)
  VPERM_CASES_BROADCAST(VPERMT2D,  VPERMI2D)
  VPERM_CASES_BROADCAST(VPERMT2PD, VPERMI2PD)
  VPERM_CASES_BROADCAST(VPERMT2PS, VPERMI2PS)
  VPERM_CASES_BROADCAST(VPERMT2Q,  VPERMI2Q)
  VPERM_CASES(VPERMT2W, VPERMI2W)
  }

  llvm_unreachable("Unreachable!");
#undef VPERM_CASES_BROADCAST
#undef VPERM_CASES
}

MachineInstr *X86InstrInfo::commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                                   unsigned OpIdx1,
                                                   unsigned OpIdx2) const {
  auto cloneIfNew = [NewMI](MachineInstr &MI) -> MachineInstr & {
    if (NewMI)
      return *MI.getParent()->getParent()->CloneMachineInstr(&MI);
    return MI;
  };

  switch (MI.getOpcode()) {
  case X86::SHRD16rri8: // A = SHRD16rri8 B, C, I -> A = SHLD16rri8 C, B, (16-I)
  case X86::SHLD16rri8: // A = SHLD16rri8 B, C, I -> A = SHRD16rri8 C, B, (16-I)
  case X86::SHRD32rri8: // A = SHRD32rri8 B, C, I -> A = SHLD32rri8 C, B, (32-I)
  case X86::SHLD32rri8: // A = SHLD32rri8 B, C, I -> A = SHRD32rri8 C, B, (32-I)
  case X86::SHRD64rri8: // A = SHRD64rri8 B, C, I -> A = SHLD64rri8 C, B, (64-I)
  case X86::SHLD64rri8:{// A = SHLD64rri8 B, C, I -> A = SHRD64rri8 C, B, (64-I)
    unsigned Opc;
    unsigned Size;
    switch (MI.getOpcode()) {
    default: llvm_unreachable("Unreachable!");
    case X86::SHRD16rri8: Size = 16; Opc = X86::SHLD16rri8; break;
    case X86::SHLD16rri8: Size = 16; Opc = X86::SHRD16rri8; break;
    case X86::SHRD32rri8: Size = 32; Opc = X86::SHLD32rri8; break;
    case X86::SHLD32rri8: Size = 32; Opc = X86::SHRD32rri8; break;
    case X86::SHRD64rri8: Size = 64; Opc = X86::SHLD64rri8; break;
    case X86::SHLD64rri8: Size = 64; Opc = X86::SHRD64rri8; break;
    }
    unsigned Amt = MI.getOperand(3).getImm();
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    WorkingMI.getOperand(3).setImm(Size - Amt);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::PFSUBrr:
  case X86::PFSUBRrr: {
    // PFSUB  x, y: x = x - y
    // PFSUBR x, y: x = y - x
    unsigned Opc =
        (X86::PFSUBRrr == MI.getOpcode() ? X86::PFSUBrr : X86::PFSUBRrr);
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
    break;
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
    switch (MI.getOpcode()) {
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
    unsigned Imm = MI.getOperand(3).getImm() & Mask;
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm(Mask ^ Imm);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::MOVSDrr:
  case X86::MOVSSrr:
  case X86::VMOVSDrr:
  case X86::VMOVSSrr:{
    // On SSE41 or later we can commute a MOVSS/MOVSD to a BLENDPS/BLENDPD.
    if (!Subtarget.hasSSE41())
      return nullptr;

    unsigned Mask, Opc;
    switch (MI.getOpcode()) {
    default: llvm_unreachable("Unreachable!");
    case X86::MOVSDrr:  Opc = X86::BLENDPDrri;  Mask = 0x02; break;
    case X86::MOVSSrr:  Opc = X86::BLENDPSrri;  Mask = 0x0E; break;
    case X86::VMOVSDrr: Opc = X86::VBLENDPDrri; Mask = 0x02; break;
    case X86::VMOVSSrr: Opc = X86::VBLENDPSrri; Mask = 0x0E; break;
    }

    // MOVSD/MOVSS's 2nd operand is a FR64/FR32 reg class - we need to copy
    // this over to a VR128 class like the 1st operand to use a BLENDPD/BLENDPS.
    auto &MRI = MI.getParent()->getParent()->getRegInfo();
    auto VR128RC = MRI.getRegClass(MI.getOperand(1).getReg());
    unsigned VR128 = MRI.createVirtualRegister(VR128RC);
    BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), get(TargetOpcode::COPY),
            VR128)
        .addReg(MI.getOperand(2).getReg());

    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    WorkingMI.getOperand(2).setReg(VR128);
    WorkingMI.addOperand(MachineOperand::CreateImm(Mask));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::PCLMULQDQrr:
  case X86::VPCLMULQDQrr:{
    // SRC1 64bits = Imm[0] ? SRC1[127:64] : SRC1[63:0]
    // SRC2 64bits = Imm[4] ? SRC2[127:64] : SRC2[63:0]
    unsigned Imm = MI.getOperand(3).getImm();
    unsigned Src1Hi = Imm & 0x01;
    unsigned Src2Hi = Imm & 0x10;
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm((Src1Hi << 4) | (Src2Hi >> 4));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::CMPSDrr:
  case X86::CMPSSrr:
  case X86::CMPPDrri:
  case X86::CMPPSrri:
  case X86::VCMPSDrr:
  case X86::VCMPSSrr:
  case X86::VCMPPDrri:
  case X86::VCMPPSrri:
  case X86::VCMPPDYrri:
  case X86::VCMPPSYrri:
  case X86::VCMPSDZrr:
  case X86::VCMPSSZrr:
  case X86::VCMPPDZrri:
  case X86::VCMPPSZrri:
  case X86::VCMPPDZ128rri:
  case X86::VCMPPSZ128rri:
  case X86::VCMPPDZ256rri:
  case X86::VCMPPSZ256rri: {
    // Float comparison can be safely commuted for
    // Ordered/Unordered/Equal/NotEqual tests
    unsigned Imm = MI.getOperand(3).getImm() & 0x7;
    switch (Imm) {
    case 0x00: // EQUAL
    case 0x03: // UNORDERED
    case 0x04: // NOT EQUAL
    case 0x07: // ORDERED
      return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
    default:
      return nullptr;
    }
  }
  case X86::VPCMPBZ128rri: case X86::VPCMPUBZ128rri:
  case X86::VPCMPBZ256rri: case X86::VPCMPUBZ256rri:
  case X86::VPCMPBZrri:    case X86::VPCMPUBZrri:
  case X86::VPCMPDZ128rri: case X86::VPCMPUDZ128rri:
  case X86::VPCMPDZ256rri: case X86::VPCMPUDZ256rri:
  case X86::VPCMPDZrri:    case X86::VPCMPUDZrri:
  case X86::VPCMPQZ128rri: case X86::VPCMPUQZ128rri:
  case X86::VPCMPQZ256rri: case X86::VPCMPUQZ256rri:
  case X86::VPCMPQZrri:    case X86::VPCMPUQZrri:
  case X86::VPCMPWZ128rri: case X86::VPCMPUWZ128rri:
  case X86::VPCMPWZ256rri: case X86::VPCMPUWZ256rri:
  case X86::VPCMPWZrri:    case X86::VPCMPUWZrri: {
    // Flip comparison mode immediate (if necessary).
    unsigned Imm = MI.getOperand(3).getImm() & 0x7;
    switch (Imm) {
    default: llvm_unreachable("Unreachable!");
    case 0x01: Imm = 0x06; break; // LT  -> NLE
    case 0x02: Imm = 0x05; break; // LE  -> NLT
    case 0x05: Imm = 0x02; break; // NLT -> LE
    case 0x06: Imm = 0x01; break; // NLE -> LT
    case 0x00: // EQ
    case 0x03: // FALSE
    case 0x04: // NE
    case 0x07: // TRUE
      break;
    }
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm(Imm);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::VPCOMBri: case X86::VPCOMUBri:
  case X86::VPCOMDri: case X86::VPCOMUDri:
  case X86::VPCOMQri: case X86::VPCOMUQri:
  case X86::VPCOMWri: case X86::VPCOMUWri: {
    // Flip comparison mode immediate (if necessary).
    unsigned Imm = MI.getOperand(3).getImm() & 0x7;
    switch (Imm) {
    default: llvm_unreachable("Unreachable!");
    case 0x00: Imm = 0x02; break; // LT -> GT
    case 0x01: Imm = 0x03; break; // LE -> GE
    case 0x02: Imm = 0x00; break; // GT -> LT
    case 0x03: Imm = 0x01; break; // GE -> LE
    case 0x04: // EQ
    case 0x05: // NE
    case 0x06: // FALSE
    case 0x07: // TRUE
      break;
    }
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm(Imm);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::VPERM2F128rr:
  case X86::VPERM2I128rr: {
    // Flip permute source immediate.
    // Imm & 0x02: lo = if set, select Op1.lo/hi else Op0.lo/hi.
    // Imm & 0x20: hi = if set, select Op1.lo/hi else Op0.lo/hi.
    unsigned Imm = MI.getOperand(3).getImm() & 0xFF;
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm(Imm ^ 0x22);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::MOVHLPSrr:
  case X86::UNPCKHPDrr: {
    if (!Subtarget.hasSSE2())
      return nullptr;

    unsigned Opc = MI.getOpcode();
    switch (Opc) {
      default: llvm_unreachable("Unreachable!");
      case X86::MOVHLPSrr: Opc = X86::UNPCKHPDrr; break;
      case X86::UNPCKHPDrr: Opc = X86::MOVHLPSrr; break;
    }
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
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
    switch (MI.getOpcode()) {
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
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case X86::VPTERNLOGDZrri:      case X86::VPTERNLOGDZrmi:
  case X86::VPTERNLOGDZ128rri:   case X86::VPTERNLOGDZ128rmi:
  case X86::VPTERNLOGDZ256rri:   case X86::VPTERNLOGDZ256rmi:
  case X86::VPTERNLOGQZrri:      case X86::VPTERNLOGQZrmi:
  case X86::VPTERNLOGQZ128rri:   case X86::VPTERNLOGQZ128rmi:
  case X86::VPTERNLOGQZ256rri:   case X86::VPTERNLOGQZ256rmi:
  case X86::VPTERNLOGDZrrik:
  case X86::VPTERNLOGDZ128rrik:
  case X86::VPTERNLOGDZ256rrik:
  case X86::VPTERNLOGQZrrik:
  case X86::VPTERNLOGQZ128rrik:
  case X86::VPTERNLOGQZ256rrik:
  case X86::VPTERNLOGDZrrikz:    case X86::VPTERNLOGDZrmikz:
  case X86::VPTERNLOGDZ128rrikz: case X86::VPTERNLOGDZ128rmikz:
  case X86::VPTERNLOGDZ256rrikz: case X86::VPTERNLOGDZ256rmikz:
  case X86::VPTERNLOGQZrrikz:    case X86::VPTERNLOGQZrmikz:
  case X86::VPTERNLOGQZ128rrikz: case X86::VPTERNLOGQZ128rmikz:
  case X86::VPTERNLOGQZ256rrikz: case X86::VPTERNLOGQZ256rmikz:
  case X86::VPTERNLOGDZ128rmbi:
  case X86::VPTERNLOGDZ256rmbi:
  case X86::VPTERNLOGDZrmbi:
  case X86::VPTERNLOGQZ128rmbi:
  case X86::VPTERNLOGQZ256rmbi:
  case X86::VPTERNLOGQZrmbi:
  case X86::VPTERNLOGDZ128rmbikz:
  case X86::VPTERNLOGDZ256rmbikz:
  case X86::VPTERNLOGDZrmbikz:
  case X86::VPTERNLOGQZ128rmbikz:
  case X86::VPTERNLOGQZ256rmbikz:
  case X86::VPTERNLOGQZrmbikz: {
    auto &WorkingMI = cloneIfNew(MI);
    if (!commuteVPTERNLOG(WorkingMI, OpIdx1, OpIdx2))
      return nullptr;
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  default: {
    if (isCommutableVPERMV3Instruction(MI.getOpcode())) {
      unsigned Opc = getCommutedVPERMV3Opcode(MI.getOpcode());
      auto &WorkingMI = cloneIfNew(MI);
      WorkingMI.setDesc(get(Opc));
      return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                     OpIdx1, OpIdx2);
    }

    const X86InstrFMA3Group *FMA3Group =
        X86InstrFMA3Info::getFMA3Group(MI.getOpcode());
    if (FMA3Group) {
      unsigned Opc =
        getFMA3OpcodeToCommuteOperands(MI, OpIdx1, OpIdx2, *FMA3Group);
      if (Opc == 0)
        return nullptr;
      auto &WorkingMI = cloneIfNew(MI);
      WorkingMI.setDesc(get(Opc));
      return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                     OpIdx1, OpIdx2);
    }

    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
  }
}

bool X86InstrInfo::findFMA3CommutedOpIndices(
    const MachineInstr &MI, unsigned &SrcOpIdx1, unsigned &SrcOpIdx2,
    const X86InstrFMA3Group &FMA3Group) const {

  if (!findThreeSrcCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2))
    return false;

  // Check if we can adjust the opcode to preserve the semantics when
  // commute the register operands.
  return getFMA3OpcodeToCommuteOperands(MI, SrcOpIdx1, SrcOpIdx2, FMA3Group) != 0;
}

bool X86InstrInfo::findThreeSrcCommutedOpIndices(const MachineInstr &MI,
                                                 unsigned &SrcOpIdx1,
                                                 unsigned &SrcOpIdx2) const {
  uint64_t TSFlags = MI.getDesc().TSFlags;

  unsigned FirstCommutableVecOp = 1;
  unsigned LastCommutableVecOp = 3;
  unsigned KMaskOp = 0;
  if (X86II::isKMasked(TSFlags)) {
    // The k-mask operand has index = 2 for masked and zero-masked operations.
    KMaskOp = 2;

    // The operand with index = 1 is used as a source for those elements for
    // which the corresponding bit in the k-mask is set to 0.
    if (X86II::isKMergeMasked(TSFlags))
      FirstCommutableVecOp = 3;

    LastCommutableVecOp++;
  }

  if (isMem(MI, LastCommutableVecOp))
    LastCommutableVecOp--;

  // Only the first RegOpsNum operands are commutable.
  // Also, the value 'CommuteAnyOperandIndex' is valid here as it means
  // that the operand is not specified/fixed.
  if (SrcOpIdx1 != CommuteAnyOperandIndex &&
      (SrcOpIdx1 < FirstCommutableVecOp || SrcOpIdx1 > LastCommutableVecOp ||
       SrcOpIdx1 == KMaskOp))
    return false;
  if (SrcOpIdx2 != CommuteAnyOperandIndex &&
      (SrcOpIdx2 < FirstCommutableVecOp || SrcOpIdx2 > LastCommutableVecOp ||
       SrcOpIdx2 == KMaskOp))
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
      CommutableOpIdx2 = LastCommutableVecOp;
    else if (SrcOpIdx2 == CommuteAnyOperandIndex)
      // Only one of operands is not fixed.
      CommutableOpIdx2 = SrcOpIdx1;

    // CommutableOpIdx2 is well defined now. Let's choose another commutable
    // operand and assign its index to CommutableOpIdx1.
    unsigned Op2Reg = MI.getOperand(CommutableOpIdx2).getReg();
    for (CommutableOpIdx1 = LastCommutableVecOp;
         CommutableOpIdx1 >= FirstCommutableVecOp; CommutableOpIdx1--) {
      // Just ignore and skip the k-mask operand.
      if (CommutableOpIdx1 == KMaskOp)
        continue;

      // The commuted operands must have different registers.
      // Otherwise, the commute transformation does not change anything and
      // is useless then.
      if (Op2Reg != MI.getOperand(CommutableOpIdx1).getReg())
        break;
    }

    // No appropriate commutable operands were found.
    if (CommutableOpIdx1 < FirstCommutableVecOp)
      return false;

    // Assign the found pair of commutable indices to SrcOpIdx1 and SrcOpidx2
    // to return those values.
    if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2,
                              CommutableOpIdx1, CommutableOpIdx2))
      return false;
  }

  return true;
}

bool X86InstrInfo::findCommutedOpIndices(MachineInstr &MI, unsigned &SrcOpIdx1,
                                         unsigned &SrcOpIdx2) const {
  const MCInstrDesc &Desc = MI.getDesc();
  if (!Desc.isCommutable())
    return false;

  switch (MI.getOpcode()) {
  case X86::CMPSDrr:
  case X86::CMPSSrr:
  case X86::CMPPDrri:
  case X86::CMPPSrri:
  case X86::VCMPSDrr:
  case X86::VCMPSSrr:
  case X86::VCMPPDrri:
  case X86::VCMPPSrri:
  case X86::VCMPPDYrri:
  case X86::VCMPPSYrri:
  case X86::VCMPSDZrr:
  case X86::VCMPSSZrr:
  case X86::VCMPPDZrri:
  case X86::VCMPPSZrri:
  case X86::VCMPPDZ128rri:
  case X86::VCMPPSZ128rri:
  case X86::VCMPPDZ256rri:
  case X86::VCMPPSZ256rri: {
    // Float comparison can be safely commuted for
    // Ordered/Unordered/Equal/NotEqual tests
    unsigned Imm = MI.getOperand(3).getImm() & 0x7;
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
  case X86::MOVSDrr:
  case X86::MOVSSrr:
  case X86::VMOVSDrr:
  case X86::VMOVSSrr: {
    if (Subtarget.hasSSE41())
      return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
    return false;
  }
  case X86::VPTERNLOGDZrri:      case X86::VPTERNLOGDZrmi:
  case X86::VPTERNLOGDZ128rri:   case X86::VPTERNLOGDZ128rmi:
  case X86::VPTERNLOGDZ256rri:   case X86::VPTERNLOGDZ256rmi:
  case X86::VPTERNLOGQZrri:      case X86::VPTERNLOGQZrmi:
  case X86::VPTERNLOGQZ128rri:   case X86::VPTERNLOGQZ128rmi:
  case X86::VPTERNLOGQZ256rri:   case X86::VPTERNLOGQZ256rmi:
  case X86::VPTERNLOGDZrrik:
  case X86::VPTERNLOGDZ128rrik:
  case X86::VPTERNLOGDZ256rrik:
  case X86::VPTERNLOGQZrrik:
  case X86::VPTERNLOGQZ128rrik:
  case X86::VPTERNLOGQZ256rrik:
  case X86::VPTERNLOGDZrrikz:    case X86::VPTERNLOGDZrmikz:
  case X86::VPTERNLOGDZ128rrikz: case X86::VPTERNLOGDZ128rmikz:
  case X86::VPTERNLOGDZ256rrikz: case X86::VPTERNLOGDZ256rmikz:
  case X86::VPTERNLOGQZrrikz:    case X86::VPTERNLOGQZrmikz:
  case X86::VPTERNLOGQZ128rrikz: case X86::VPTERNLOGQZ128rmikz:
  case X86::VPTERNLOGQZ256rrikz: case X86::VPTERNLOGQZ256rmikz:
  case X86::VPTERNLOGDZ128rmbi:
  case X86::VPTERNLOGDZ256rmbi:
  case X86::VPTERNLOGDZrmbi:
  case X86::VPTERNLOGQZ128rmbi:
  case X86::VPTERNLOGQZ256rmbi:
  case X86::VPTERNLOGQZrmbi:
  case X86::VPTERNLOGDZ128rmbikz:
  case X86::VPTERNLOGDZ256rmbikz:
  case X86::VPTERNLOGDZrmbikz:
  case X86::VPTERNLOGQZ128rmbikz:
  case X86::VPTERNLOGQZ256rmbikz:
  case X86::VPTERNLOGQZrmbikz:
    return findThreeSrcCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
  default:
    const X86InstrFMA3Group *FMA3Group =
        X86InstrFMA3Info::getFMA3Group(MI.getOpcode());
    if (FMA3Group)
      return findFMA3CommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2, *FMA3Group);

    // Handled masked instructions since we need to skip over the mask input
    // and the preserved input.
    if (Desc.TSFlags & X86II::EVEX_K) {
      // First assume that the first input is the mask operand and skip past it.
      unsigned CommutableOpIdx1 = Desc.getNumDefs() + 1;
      unsigned CommutableOpIdx2 = Desc.getNumDefs() + 2;
      // Check if the first input is tied. If there isn't one then we only
      // need to skip the mask operand which we did above.
      if ((MI.getDesc().getOperandConstraint(Desc.getNumDefs(),
                                             MCOI::TIED_TO) != -1)) {
        // If this is zero masking instruction with a tied operand, we need to
        // move the first index back to the first input since this must
        // be a 3 input instruction and we want the first two non-mask inputs.
        // Otherwise this is a 2 input instruction with a preserved input and
        // mask, so we need to move the indices to skip one more input.
        if (Desc.TSFlags & X86II::EVEX_Z)
          --CommutableOpIdx1;
        else {
          ++CommutableOpIdx1;
          ++CommutableOpIdx2;
        }
      }

      if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2,
                                CommutableOpIdx1, CommutableOpIdx2))
        return false;

      if (!MI.getOperand(SrcOpIdx1).isReg() ||
          !MI.getOperand(SrcOpIdx2).isReg())
        // No idea.
        return false;
      return true;
    }

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
  case X86::COND_NE_OR_P:  return X86::COND_E_AND_NP;
  case X86::COND_E_AND_NP: return X86::COND_NE_OR_P;
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

std::pair<X86::CondCode, bool>
X86::getX86ConditionCode(CmpInst::Predicate Predicate) {
  X86::CondCode CC = X86::COND_INVALID;
  bool NeedSwap = false;
  switch (Predicate) {
  default: break;
  // Floating-point Predicates
  case CmpInst::FCMP_UEQ: CC = X86::COND_E;       break;
  case CmpInst::FCMP_OLT: NeedSwap = true;        LLVM_FALLTHROUGH;
  case CmpInst::FCMP_OGT: CC = X86::COND_A;       break;
  case CmpInst::FCMP_OLE: NeedSwap = true;        LLVM_FALLTHROUGH;
  case CmpInst::FCMP_OGE: CC = X86::COND_AE;      break;
  case CmpInst::FCMP_UGT: NeedSwap = true;        LLVM_FALLTHROUGH;
  case CmpInst::FCMP_ULT: CC = X86::COND_B;       break;
  case CmpInst::FCMP_UGE: NeedSwap = true;        LLVM_FALLTHROUGH;
  case CmpInst::FCMP_ULE: CC = X86::COND_BE;      break;
  case CmpInst::FCMP_ONE: CC = X86::COND_NE;      break;
  case CmpInst::FCMP_UNO: CC = X86::COND_P;       break;
  case CmpInst::FCMP_ORD: CC = X86::COND_NP;      break;
  case CmpInst::FCMP_OEQ:                         LLVM_FALLTHROUGH;
  case CmpInst::FCMP_UNE: CC = X86::COND_INVALID; break;

  // Integer Predicates
  case CmpInst::ICMP_EQ:  CC = X86::COND_E;       break;
  case CmpInst::ICMP_NE:  CC = X86::COND_NE;      break;
  case CmpInst::ICMP_UGT: CC = X86::COND_A;       break;
  case CmpInst::ICMP_UGE: CC = X86::COND_AE;      break;
  case CmpInst::ICMP_ULT: CC = X86::COND_B;       break;
  case CmpInst::ICMP_ULE: CC = X86::COND_BE;      break;
  case CmpInst::ICMP_SGT: CC = X86::COND_G;       break;
  case CmpInst::ICMP_SGE: CC = X86::COND_GE;      break;
  case CmpInst::ICMP_SLT: CC = X86::COND_L;       break;
  case CmpInst::ICMP_SLE: CC = X86::COND_LE;      break;
  }

  return std::make_pair(CC, NeedSwap);
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

bool X86InstrInfo::isUnpredicatedTerminator(const MachineInstr &MI) const {
  if (!MI.isTerminator()) return false;

  // Conditional branch is a special case.
  if (MI.isBranch() && !MI.isBarrier())
    return true;
  if (!MI.isPredicable())
    return true;
  return !isPredicated(MI);
}

bool X86InstrInfo::isUnconditionalTailCall(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case X86::TCRETURNdi:
  case X86::TCRETURNri:
  case X86::TCRETURNmi:
  case X86::TCRETURNdi64:
  case X86::TCRETURNri64:
  case X86::TCRETURNmi64:
    return true;
  default:
    return false;
  }
}

bool X86InstrInfo::canMakeTailCallConditional(
    SmallVectorImpl<MachineOperand> &BranchCond,
    const MachineInstr &TailCall) const {
  if (TailCall.getOpcode() != X86::TCRETURNdi &&
      TailCall.getOpcode() != X86::TCRETURNdi64) {
    // Only direct calls can be done with a conditional branch.
    return false;
  }

  const MachineFunction *MF = TailCall.getParent()->getParent();
  if (Subtarget.isTargetWin64() && MF->hasWinCFI()) {
    // Conditional tail calls confuse the Win64 unwinder.
    return false;
  }

  assert(BranchCond.size() == 1);
  if (BranchCond[0].getImm() > X86::LAST_VALID_COND) {
    // Can't make a conditional tail call with this condition.
    return false;
  }

  const X86MachineFunctionInfo *X86FI = MF->getInfo<X86MachineFunctionInfo>();
  if (X86FI->getTCReturnAddrDelta() != 0 ||
      TailCall.getOperand(1).getImm() != 0) {
    // A conditional tail call cannot do any stack adjustment.
    return false;
  }

  return true;
}

void X86InstrInfo::replaceBranchWithTailCall(
    MachineBasicBlock &MBB, SmallVectorImpl<MachineOperand> &BranchCond,
    const MachineInstr &TailCall) const {
  assert(canMakeTailCallConditional(BranchCond, TailCall));

  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (!I->isBranch())
      assert(0 && "Can't find the branch to replace!");

    X86::CondCode CC = getCondFromBranchOpc(I->getOpcode());
    assert(BranchCond.size() == 1);
    if (CC != BranchCond[0].getImm())
      continue;

    break;
  }

  unsigned Opc = TailCall.getOpcode() == X86::TCRETURNdi ? X86::TCRETURNdicc
                                                         : X86::TCRETURNdi64cc;

  auto MIB = BuildMI(MBB, I, MBB.findDebugLoc(I), get(Opc));
  MIB->addOperand(TailCall.getOperand(0)); // Destination.
  MIB.addImm(0); // Stack offset (not used).
  MIB->addOperand(BranchCond[0]); // Condition.
  MIB.copyImplicitOps(TailCall); // Regmask and (imp-used) parameters.

  // Add implicit uses and defs of all live regs potentially clobbered by the
  // call. This way they still appear live across the call.
  LivePhysRegs LiveRegs(&getRegisterInfo());
  LiveRegs.addLiveOuts(MBB);
  SmallVector<std::pair<unsigned, const MachineOperand *>, 8> Clobbers;
  LiveRegs.stepForward(*MIB, Clobbers);
  for (const auto &C : Clobbers) {
    MIB.addReg(C.first, RegState::Implicit);
    MIB.addReg(C.first, RegState::Implicit | RegState::Define);
  }

  I->eraseFromParent();
}

// Given a MBB and its TBB, find the FBB which was a fallthrough MBB (it may
// not be a fallthrough MBB now due to layout changes). Return nullptr if the
// fallthrough MBB cannot be identified.
static MachineBasicBlock *getFallThroughMBB(MachineBasicBlock *MBB,
                                            MachineBasicBlock *TBB) {
  // Look for non-EHPad successors other than TBB. If we find exactly one, it
  // is the fallthrough MBB. If we find zero, then TBB is both the target MBB
  // and fallthrough MBB. If we find more than one, we cannot identify the
  // fallthrough MBB and should return nullptr.
  MachineBasicBlock *FallthroughBB = nullptr;
  for (auto SI = MBB->succ_begin(), SE = MBB->succ_end(); SI != SE; ++SI) {
    if ((*SI)->isEHPad() || (*SI == TBB && FallthroughBB))
      continue;
    // Return a nullptr if we found more than one fallthrough successor.
    if (FallthroughBB && FallthroughBB != TBB)
      return nullptr;
    FallthroughBB = *SI;
  }
  return FallthroughBB;
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
    if (!isUnpredicatedTerminator(*I))
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
      CondBranches.push_back(&*I);
      continue;
    }

    // Handle subsequent conditional branches. Only handle the case where all
    // conditional branches branch to the same destination and their condition
    // opcodes fit one of the special multi-branch idioms.
    assert(Cond.size() == 1);
    assert(TBB);

    // If the conditions are the same, we can leave them alone.
    X86::CondCode OldBranchCode = (X86::CondCode)Cond[0].getImm();
    auto NewTBB = I->getOperand(0).getMBB();
    if (OldBranchCode == BranchCode && TBB == NewTBB)
      continue;

    // If they differ, see if they fit one of the known patterns. Theoretically,
    // we could handle more patterns here, but we shouldn't expect to see them
    // if instruction selection has done a reasonable job.
    if (TBB == NewTBB &&
               ((OldBranchCode == X86::COND_P && BranchCode == X86::COND_NE) ||
                (OldBranchCode == X86::COND_NE && BranchCode == X86::COND_P))) {
      BranchCode = X86::COND_NE_OR_P;
    } else if ((OldBranchCode == X86::COND_NP && BranchCode == X86::COND_NE) ||
               (OldBranchCode == X86::COND_E && BranchCode == X86::COND_P)) {
      if (NewTBB != (FBB ? FBB : getFallThroughMBB(&MBB, TBB)))
        return true;

      // X86::COND_E_AND_NP usually has two different branch destinations.
      //
      // JP B1
      // JE B2
      // JMP B1
      // B1:
      // B2:
      //
      // Here this condition branches to B2 only if NP && E. It has another
      // equivalent form:
      //
      // JNE B1
      // JNP B2
      // JMP B1
      // B1:
      // B2:
      //
      // Similarly it branches to B2 only if E && NP. That is why this condition
      // is named with COND_E_AND_NP.
      BranchCode = X86::COND_E_AND_NP;
    } else
      return true;

    // Update the MachineOperand.
    Cond[0].setImm(BranchCode);
    CondBranches.push_back(&*I);
  }

  return false;
}

bool X86InstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  SmallVector<MachineInstr *, 4> CondBranches;
  return AnalyzeBranchImpl(MBB, TBB, FBB, Cond, CondBranches, AllowModify);
}

bool X86InstrInfo::analyzeBranchPredicate(MachineBasicBlock &MBB,
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

unsigned X86InstrInfo::removeBranch(MachineBasicBlock &MBB,
                                    int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

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

unsigned X86InstrInfo::insertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    ArrayRef<MachineOperand> Cond,
                                    const DebugLoc &DL,
                                    int *BytesAdded) const {
  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");
  assert(!BytesAdded && "code size not handled");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(X86::JMP_1)).addMBB(TBB);
    return 1;
  }

  // If FBB is null, it is implied to be a fall-through block.
  bool FallThru = FBB == nullptr;

  // Conditional branch.
  unsigned Count = 0;
  X86::CondCode CC = (X86::CondCode)Cond[0].getImm();
  switch (CC) {
  case X86::COND_NE_OR_P:
    // Synthesize NE_OR_P with two branches.
    BuildMI(&MBB, DL, get(X86::JNE_1)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JP_1)).addMBB(TBB);
    ++Count;
    break;
  case X86::COND_E_AND_NP:
    // Use the next block of MBB as FBB if it is null.
    if (FBB == nullptr) {
      FBB = getFallThroughMBB(&MBB, TBB);
      assert(FBB && "MBB cannot be the last block in function when the false "
                    "body is a fall-through.");
    }
    // Synthesize COND_E_AND_NP with two branches.
    BuildMI(&MBB, DL, get(X86::JNE_1)).addMBB(FBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JNP_1)).addMBB(TBB);
    ++Count;
    break;
  default: {
    unsigned Opc = GetCondBranchFromCond(CC);
    BuildMI(&MBB, DL, get(Opc)).addMBB(TBB);
    ++Count;
  }
  }
  if (!FallThru) {
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
                                MachineBasicBlock::iterator I,
                                const DebugLoc &DL, unsigned DstReg,
                                ArrayRef<MachineOperand> Cond, unsigned TrueReg,
                                unsigned FalseReg) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
  const TargetRegisterClass &RC = *MRI.getRegClass(DstReg);
  assert(Cond.size() == 1 && "Invalid Cond array");
  unsigned Opc = getCMovFromCond((X86::CondCode)Cond[0].getImm(),
                                 TRI.getRegSizeInBits(RC) / 8,
                                 false /*HasMemoryOperand*/);
  BuildMI(MBB, I, DL, get(Opc), DstReg).addReg(FalseReg).addReg(TrueReg);
}

/// Test if the given register is a physical h register.
static bool isHReg(unsigned Reg) {
  return X86::GR8_ABCD_HRegClass.contains(Reg);
}

// Try and copy between VR128/VR64 and GR64 registers.
static unsigned CopyToFromAsymmetricReg(unsigned &DestReg, unsigned &SrcReg,
                                        const X86Subtarget &Subtarget) {
  bool HasAVX = Subtarget.hasAVX();
  bool HasAVX512 = Subtarget.hasAVX512();

  // SrcReg(MaskReg) -> DestReg(GR64)
  // SrcReg(MaskReg) -> DestReg(GR32)

  // All KMASK RegClasses hold the same k registers, can be tested against anyone.
  if (X86::VK16RegClass.contains(SrcReg)) {
    if (X86::GR64RegClass.contains(DestReg)) {
      assert(Subtarget.hasBWI());
      return X86::KMOVQrk;
    }
    if (X86::GR32RegClass.contains(DestReg))
      return Subtarget.hasBWI() ? X86::KMOVDrk : X86::KMOVWrk;
  }

  // SrcReg(GR64) -> DestReg(MaskReg)
  // SrcReg(GR32) -> DestReg(MaskReg)

  // All KMASK RegClasses hold the same k registers, can be tested against anyone.
  if (X86::VK16RegClass.contains(DestReg)) {
    if (X86::GR64RegClass.contains(SrcReg)) {
      assert(Subtarget.hasBWI());
      return X86::KMOVQkr;
    }
    if (X86::GR32RegClass.contains(SrcReg))
      return Subtarget.hasBWI() ? X86::KMOVDkr : X86::KMOVWkr;
  }


  // SrcReg(VR128) -> DestReg(GR64)
  // SrcReg(VR64)  -> DestReg(GR64)
  // SrcReg(GR64)  -> DestReg(VR128)
  // SrcReg(GR64)  -> DestReg(VR64)

  if (X86::GR64RegClass.contains(DestReg)) {
    if (X86::VR128XRegClass.contains(SrcReg))
      // Copy from a VR128 register to a GR64 register.
      return HasAVX512 ? X86::VMOVPQIto64Zrr :
             HasAVX    ? X86::VMOVPQIto64rr  :
                         X86::MOVPQIto64rr;
    if (X86::VR64RegClass.contains(SrcReg))
      // Copy from a VR64 register to a GR64 register.
      return X86::MMX_MOVD64from64rr;
  } else if (X86::GR64RegClass.contains(SrcReg)) {
    // Copy from a GR64 register to a VR128 register.
    if (X86::VR128XRegClass.contains(DestReg))
      return HasAVX512 ? X86::VMOV64toPQIZrr :
             HasAVX    ? X86::VMOV64toPQIrr  :
                         X86::MOV64toPQIrr;
    // Copy from a GR64 register to a VR64 register.
    if (X86::VR64RegClass.contains(DestReg))
      return X86::MMX_MOVD64to64rr;
  }

  // SrcReg(FR32) -> DestReg(GR32)
  // SrcReg(GR32) -> DestReg(FR32)

  if (X86::GR32RegClass.contains(DestReg) &&
      X86::FR32XRegClass.contains(SrcReg))
    // Copy from a FR32 register to a GR32 register.
    return HasAVX512 ? X86::VMOVSS2DIZrr :
           HasAVX    ? X86::VMOVSS2DIrr  :
                       X86::MOVSS2DIrr;

  if (X86::FR32XRegClass.contains(DestReg) &&
      X86::GR32RegClass.contains(SrcReg))
    // Copy from a GR32 register to a FR32 register.
    return HasAVX512 ? X86::VMOVDI2SSZrr :
           HasAVX    ? X86::VMOVDI2SSrr  :
                       X86::MOVDI2SSrr;
  return 0;
}

void X86InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI,
                               const DebugLoc &DL, unsigned DestReg,
                               unsigned SrcReg, bool KillSrc) const {
  // First deal with the normal symmetric copies.
  bool HasAVX = Subtarget.hasAVX();
  bool HasVLX = Subtarget.hasVLX();
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
  else if (X86::VR128XRegClass.contains(DestReg, SrcReg)) {
    if (HasVLX)
      Opc = X86::VMOVAPSZ128rr;
    else if (X86::VR128RegClass.contains(DestReg, SrcReg))
      Opc = HasAVX ? X86::VMOVAPSrr : X86::MOVAPSrr;
    else {
      // If this an extended register and we don't have VLX we need to use a
      // 512-bit move.
      Opc = X86::VMOVAPSZrr;
      const TargetRegisterInfo *TRI = &getRegisterInfo();
      DestReg = TRI->getMatchingSuperReg(DestReg, X86::sub_xmm,
                                         &X86::VR512RegClass);
      SrcReg = TRI->getMatchingSuperReg(SrcReg, X86::sub_xmm,
                                        &X86::VR512RegClass);
    }
  } else if (X86::VR256XRegClass.contains(DestReg, SrcReg)) {
    if (HasVLX)
      Opc = X86::VMOVAPSZ256rr;
    else if (X86::VR256RegClass.contains(DestReg, SrcReg))
      Opc = X86::VMOVAPSYrr;
    else {
      // If this an extended register and we don't have VLX we need to use a
      // 512-bit move.
      Opc = X86::VMOVAPSZrr;
      const TargetRegisterInfo *TRI = &getRegisterInfo();
      DestReg = TRI->getMatchingSuperReg(DestReg, X86::sub_ymm,
                                         &X86::VR512RegClass);
      SrcReg = TRI->getMatchingSuperReg(SrcReg, X86::sub_ymm,
                                        &X86::VR512RegClass);
    }
  } else if (X86::VR512RegClass.contains(DestReg, SrcReg))
    Opc = X86::VMOVAPSZrr;
  // All KMASK RegClasses hold the same k registers, can be tested against anyone.
  else if (X86::VK16RegClass.contains(DestReg, SrcReg))
    Opc = Subtarget.hasBWI() ? X86::KMOVQkk : X86::KMOVWkk;
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
    int Mov = is64 ? X86::MOV64rr : X86::MOV32rr;
    int Push = is64 ? X86::PUSH64r : X86::PUSH32r;
    int PushF = is64 ? X86::PUSHF64 : X86::PUSHF32;
    int Pop = is64 ? X86::POP64r : X86::POP32r;
    int PopF = is64 ? X86::POPF64 : X86::POPF32;
    int AX = is64 ? X86::RAX : X86::EAX;

    if (!Subtarget.hasLAHFSAHF()) {
      assert(Subtarget.is64Bit() &&
             "Not having LAHF/SAHF only happens on 64-bit.");
      // Moving EFLAGS to / from another register requires a push and a pop.
      // Notice that we have to adjust the stack if we don't want to clobber the
      // first frame index. See X86FrameLowering.cpp - usesTheStack.
      if (FromEFLAGS) {
        BuildMI(MBB, MI, DL, get(PushF));
        BuildMI(MBB, MI, DL, get(Pop), DestReg);
      }
      if (ToEFLAGS) {
        BuildMI(MBB, MI, DL, get(Push))
            .addReg(SrcReg, getKillRegState(KillSrc));
        BuildMI(MBB, MI, DL, get(PopF));
      }
      return;
    }

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
    // first frame index.
    // See X86ISelLowering.cpp - X86::hasCopyImplyingStackAdjustment.

    const TargetRegisterInfo *TRI = &getRegisterInfo();
    MachineBasicBlock::LivenessQueryResult LQR =
        MBB.computeRegisterLiveness(TRI, AX, MI);
    // We do not want to save and restore AX if we do not have to.
    // Moreover, if we do so whereas AX is dead, we would need to set
    // an undef flag on the use of AX, otherwise the verifier will
    // complain that we read an undef value.
    // We do not want to change the behavior of the machine verifier
    // as this is usually wrong to read an undef value.
    if (MachineBasicBlock::LQR_Unknown == LQR) {
      LivePhysRegs LPR(TRI);
      LPR.addLiveOuts(MBB);
      MachineBasicBlock::iterator I = MBB.end();
      while (I != MI) {
        --I;
        LPR.stepBackward(*I);
      }
      // AX contains the top most register in the aliasing hierarchy.
      // It may not be live, but one of its aliases may be.
      for (MCRegAliasIterator AI(AX, TRI, true);
           AI.isValid() && LQR != MachineBasicBlock::LQR_Live; ++AI)
        LQR = LPR.contains(*AI) ? MachineBasicBlock::LQR_Live
                                : MachineBasicBlock::LQR_Dead;
    }
    bool AXDead = (Reg == AX) || (MachineBasicBlock::LQR_Dead == LQR);
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
  bool HasAVX = STI.hasAVX();
  bool HasAVX512 = STI.hasAVX512();
  bool HasVLX = STI.hasVLX();

  switch (STI.getRegisterInfo()->getSpillSize(*RC)) {
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
    if (X86::VK16RegClass.hasSubClassEq(RC))
      return load ? X86::KMOVWkm : X86::KMOVWmk;
    assert(X86::GR16RegClass.hasSubClassEq(RC) && "Unknown 2-byte regclass");
    return load ? X86::MOV16rm : X86::MOV16mr;
  case 4:
    if (X86::GR32RegClass.hasSubClassEq(RC))
      return load ? X86::MOV32rm : X86::MOV32mr;
    if (X86::FR32XRegClass.hasSubClassEq(RC))
      return load ?
        (HasAVX512 ? X86::VMOVSSZrm : HasAVX ? X86::VMOVSSrm : X86::MOVSSrm) :
        (HasAVX512 ? X86::VMOVSSZmr : HasAVX ? X86::VMOVSSmr : X86::MOVSSmr);
    if (X86::RFP32RegClass.hasSubClassEq(RC))
      return load ? X86::LD_Fp32m : X86::ST_Fp32m;
    if (X86::VK32RegClass.hasSubClassEq(RC))
      return load ? X86::KMOVDkm : X86::KMOVDmk;
    llvm_unreachable("Unknown 4-byte regclass");
  case 8:
    if (X86::GR64RegClass.hasSubClassEq(RC))
      return load ? X86::MOV64rm : X86::MOV64mr;
    if (X86::FR64XRegClass.hasSubClassEq(RC))
      return load ?
        (HasAVX512 ? X86::VMOVSDZrm : HasAVX ? X86::VMOVSDrm : X86::MOVSDrm) :
        (HasAVX512 ? X86::VMOVSDZmr : HasAVX ? X86::VMOVSDmr : X86::MOVSDmr);
    if (X86::VR64RegClass.hasSubClassEq(RC))
      return load ? X86::MMX_MOVQ64rm : X86::MMX_MOVQ64mr;
    if (X86::RFP64RegClass.hasSubClassEq(RC))
      return load ? X86::LD_Fp64m : X86::ST_Fp64m;
    if (X86::VK64RegClass.hasSubClassEq(RC))
      return load ? X86::KMOVQkm : X86::KMOVQmk;
    llvm_unreachable("Unknown 8-byte regclass");
  case 10:
    assert(X86::RFP80RegClass.hasSubClassEq(RC) && "Unknown 10-byte regclass");
    return load ? X86::LD_Fp80m : X86::ST_FpP80m;
  case 16: {
    if (X86::VR128XRegClass.hasSubClassEq(RC)) {
      // If stack is realigned we can use aligned stores.
      if (isStackAligned)
        return load ?
          (HasVLX    ? X86::VMOVAPSZ128rm :
           HasAVX512 ? X86::VMOVAPSZ128rm_NOVLX :
           HasAVX    ? X86::VMOVAPSrm :
                       X86::MOVAPSrm):
          (HasVLX    ? X86::VMOVAPSZ128mr :
           HasAVX512 ? X86::VMOVAPSZ128mr_NOVLX :
           HasAVX    ? X86::VMOVAPSmr :
                       X86::MOVAPSmr);
      else
        return load ?
          (HasVLX    ? X86::VMOVUPSZ128rm :
           HasAVX512 ? X86::VMOVUPSZ128rm_NOVLX :
           HasAVX    ? X86::VMOVUPSrm :
                       X86::MOVUPSrm):
          (HasVLX    ? X86::VMOVUPSZ128mr :
           HasAVX512 ? X86::VMOVUPSZ128mr_NOVLX :
           HasAVX    ? X86::VMOVUPSmr :
                       X86::MOVUPSmr);
    }
    if (X86::BNDRRegClass.hasSubClassEq(RC)) {
      if (STI.is64Bit())
        return load ? X86::BNDMOVRM64rm : X86::BNDMOVMR64mr;
      else
        return load ? X86::BNDMOVRM32rm : X86::BNDMOVMR32mr;
    }
    llvm_unreachable("Unknown 16-byte regclass");
  }
  case 32:
    assert(X86::VR256XRegClass.hasSubClassEq(RC) && "Unknown 32-byte regclass");
    // If stack is realigned we can use aligned stores.
    if (isStackAligned)
      return load ?
        (HasVLX    ? X86::VMOVAPSZ256rm :
         HasAVX512 ? X86::VMOVAPSZ256rm_NOVLX :
                     X86::VMOVAPSYrm) :
        (HasVLX    ? X86::VMOVAPSZ256mr :
         HasAVX512 ? X86::VMOVAPSZ256mr_NOVLX :
                     X86::VMOVAPSYmr);
    else
      return load ?
        (HasVLX    ? X86::VMOVUPSZ256rm :
         HasAVX512 ? X86::VMOVUPSZ256rm_NOVLX :
                     X86::VMOVUPSYrm) :
        (HasVLX    ? X86::VMOVUPSZ256mr :
         HasAVX512 ? X86::VMOVUPSZ256mr_NOVLX :
                     X86::VMOVUPSYmr);
  case 64:
    assert(X86::VR512RegClass.hasSubClassEq(RC) && "Unknown 64-byte regclass");
    assert(STI.hasAVX512() && "Using 512-bit register requires AVX512");
    if (isStackAligned)
      return load ? X86::VMOVAPSZrm : X86::VMOVAPSZmr;
    else
      return load ? X86::VMOVUPSZrm : X86::VMOVUPSZmr;
  }
}

bool X86InstrInfo::getMemOpBaseRegImmOfs(MachineInstr &MemOp, unsigned &BaseReg,
                                         int64_t &Offset,
                                         const TargetRegisterInfo *TRI) const {
  const MCInstrDesc &Desc = MemOp.getDesc();
  int MemRefBegin = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemRefBegin < 0)
    return false;

  MemRefBegin += X86II::getOperandBias(Desc);

  MachineOperand &BaseMO = MemOp.getOperand(MemRefBegin + X86::AddrBaseReg);
  if (!BaseMO.isReg()) // Can be an MO_FrameIndex
    return false;

  BaseReg = BaseMO.getReg();
  if (MemOp.getOperand(MemRefBegin + X86::AddrScaleAmt).getImm() != 1)
    return false;

  if (MemOp.getOperand(MemRefBegin + X86::AddrIndexReg).getReg() !=
      X86::NoRegister)
    return false;

  const MachineOperand &DispMO = MemOp.getOperand(MemRefBegin + X86::AddrDisp);

  // Displacement can be symbolic
  if (!DispMO.isImm())
    return false;

  Offset = DispMO.getImm();

  return true;
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
  assert(MF.getFrameInfo().getObjectSize(FrameIdx) >= TRI->getSpillSize(*RC) &&
         "Stack slot too small for store");
  unsigned Alignment = std::max<uint32_t>(TRI->getSpillSize(*RC), 16);
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
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  unsigned Alignment = std::max<uint32_t>(TRI.getSpillSize(*RC), 16);
  bool isAligned = MMOBegin != MMOEnd &&
                   (*MMOBegin)->getAlignment() >= Alignment;
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, Subtarget);
  DebugLoc DL;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.add(Addr[i]);
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
  unsigned Alignment = std::max<uint32_t>(TRI->getSpillSize(*RC), 16);
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
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  unsigned Alignment = std::max<uint32_t>(TRI.getSpillSize(*RC), 16);
  bool isAligned = MMOBegin != MMOEnd &&
                   (*MMOBegin)->getAlignment() >= Alignment;
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, Subtarget);
  DebugLoc DL;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.add(Addr[i]);
  (*MIB).setMemRefs(MMOBegin, MMOEnd);
  NewMIs.push_back(MIB);
}

bool X86InstrInfo::analyzeCompare(const MachineInstr &MI, unsigned &SrcReg,
                                  unsigned &SrcReg2, int &CmpMask,
                                  int &CmpValue) const {
  switch (MI.getOpcode()) {
  default: break;
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP8ri:
    SrcReg = MI.getOperand(0).getReg();
    SrcReg2 = 0;
    if (MI.getOperand(1).isImm()) {
      CmpMask = ~0;
      CmpValue = MI.getOperand(1).getImm();
    } else {
      CmpMask = CmpValue = 0;
    }
    return true;
  // A SUB can be used to perform comparison.
  case X86::SUB64rm:
  case X86::SUB32rm:
  case X86::SUB16rm:
  case X86::SUB8rm:
    SrcReg = MI.getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = 0;
    CmpValue = 0;
    return true;
  case X86::SUB64rr:
  case X86::SUB32rr:
  case X86::SUB16rr:
  case X86::SUB8rr:
    SrcReg = MI.getOperand(1).getReg();
    SrcReg2 = MI.getOperand(2).getReg();
    CmpMask = 0;
    CmpValue = 0;
    return true;
  case X86::SUB64ri32:
  case X86::SUB64ri8:
  case X86::SUB32ri:
  case X86::SUB32ri8:
  case X86::SUB16ri:
  case X86::SUB16ri8:
  case X86::SUB8ri:
    SrcReg = MI.getOperand(1).getReg();
    SrcReg2 = 0;
    if (MI.getOperand(2).isImm()) {
      CmpMask = ~0;
      CmpValue = MI.getOperand(2).getImm();
    } else {
      CmpMask = CmpValue = 0;
    }
    return true;
  case X86::CMP64rr:
  case X86::CMP32rr:
  case X86::CMP16rr:
  case X86::CMP8rr:
    SrcReg = MI.getOperand(0).getReg();
    SrcReg2 = MI.getOperand(1).getReg();
    CmpMask = 0;
    CmpValue = 0;
    return true;
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
    SrcReg = MI.getOperand(0).getReg();
    if (MI.getOperand(1).getReg() != SrcReg)
      return false;
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
inline static bool isRedundantFlagInstr(MachineInstr &FlagI, unsigned SrcReg,
                                        unsigned SrcReg2, int ImmMask,
                                        int ImmValue, MachineInstr &OI) {
  if (((FlagI.getOpcode() == X86::CMP64rr && OI.getOpcode() == X86::SUB64rr) ||
       (FlagI.getOpcode() == X86::CMP32rr && OI.getOpcode() == X86::SUB32rr) ||
       (FlagI.getOpcode() == X86::CMP16rr && OI.getOpcode() == X86::SUB16rr) ||
       (FlagI.getOpcode() == X86::CMP8rr && OI.getOpcode() == X86::SUB8rr)) &&
      ((OI.getOperand(1).getReg() == SrcReg &&
        OI.getOperand(2).getReg() == SrcReg2) ||
       (OI.getOperand(1).getReg() == SrcReg2 &&
        OI.getOperand(2).getReg() == SrcReg)))
    return true;

  if (ImmMask != 0 &&
      ((FlagI.getOpcode() == X86::CMP64ri32 &&
        OI.getOpcode() == X86::SUB64ri32) ||
       (FlagI.getOpcode() == X86::CMP64ri8 &&
        OI.getOpcode() == X86::SUB64ri8) ||
       (FlagI.getOpcode() == X86::CMP32ri && OI.getOpcode() == X86::SUB32ri) ||
       (FlagI.getOpcode() == X86::CMP32ri8 &&
        OI.getOpcode() == X86::SUB32ri8) ||
       (FlagI.getOpcode() == X86::CMP16ri && OI.getOpcode() == X86::SUB16ri) ||
       (FlagI.getOpcode() == X86::CMP16ri8 &&
        OI.getOpcode() == X86::SUB16ri8) ||
       (FlagI.getOpcode() == X86::CMP8ri && OI.getOpcode() == X86::SUB8ri)) &&
      OI.getOperand(1).getReg() == SrcReg &&
      OI.getOperand(2).getImm() == ImmValue)
    return true;
  return false;
}

/// Check whether the definition can be converted
/// to remove a comparison against zero.
inline static bool isDefConvertible(MachineInstr &MI) {
  switch (MI.getOpcode()) {
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
static X86::CondCode isUseDefConvertible(MachineInstr &MI) {
  switch (MI.getOpcode()) {
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
bool X86InstrInfo::optimizeCompareInstr(MachineInstr &CmpInstr, unsigned SrcReg,
                                        unsigned SrcReg2, int CmpMask,
                                        int CmpValue,
                                        const MachineRegisterInfo *MRI) const {
  // Check whether we can replace SUB with CMP.
  unsigned NewOpcode = 0;
  switch (CmpInstr.getOpcode()) {
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
    if (!MRI->use_nodbg_empty(CmpInstr.getOperand(0).getReg()))
      return false;
    // There is no use of the destination register, we can replace SUB with CMP.
    switch (CmpInstr.getOpcode()) {
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
    CmpInstr.setDesc(get(NewOpcode));
    CmpInstr.RemoveOperand(0);
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
  bool IsCmpZero = (CmpMask != 0 && CmpValue == 0);
  if (IsCmpZero && MI->getParent() != CmpInstr.getParent())
    return false;

  // If we have a use of the source register between the def and our compare
  // instruction we can eliminate the compare iff the use sets EFLAGS in the
  // right way.
  bool ShouldUpdateCC = false;
  X86::CondCode NewCC = X86::COND_INVALID;
  if (IsCmpZero && !isDefConvertible(*MI)) {
    // Scan forward from the use until we hit the use we're looking for or the
    // compare instruction.
    for (MachineBasicBlock::iterator J = MI;; ++J) {
      // Do we have a convertible instruction?
      NewCC = isUseDefConvertible(*J);
      if (NewCC != X86::COND_INVALID && J->getOperand(1).isReg() &&
          J->getOperand(1).getReg() == SrcReg) {
        assert(J->definesRegister(X86::EFLAGS) && "Must be an EFLAGS def!");
        ShouldUpdateCC = true; // Update CC later on.
        // This is not a def of SrcReg, but still a def of EFLAGS. Keep going
        // with the new def.
        Def = J;
        MI = &*Def;
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
      RI = ++I.getReverse(),
      RE = CmpInstr.getParent() == MI->getParent()
               ? Def.getReverse() /* points to MI */
               : CmpInstr.getParent()->rend();
  MachineInstr *Movr0Inst = nullptr;
  for (; RI != RE; ++RI) {
    MachineInstr &Instr = *RI;
    // Check whether CmpInstr can be made redundant by the current instruction.
    if (!IsCmpZero && isRedundantFlagInstr(CmpInstr, SrcReg, SrcReg2, CmpMask,
                                           CmpValue, Instr)) {
      Sub = &Instr;
      break;
    }

    if (Instr.modifiesRegister(X86::EFLAGS, TRI) ||
        Instr.readsRegister(X86::EFLAGS, TRI)) {
      // This instruction modifies or uses EFLAGS.

      // MOV32r0 etc. are implemented with xor which clobbers condition code.
      // They are safe to move up, if the definition to EFLAGS is dead and
      // earlier instructions do not read or write EFLAGS.
      if (!Movr0Inst && Instr.getOpcode() == X86::MOV32r0 &&
          Instr.registerDefIsDead(X86::EFLAGS, TRI)) {
        Movr0Inst = &Instr;
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
  MachineBasicBlock::iterator E = CmpInstr.getParent()->end();
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
        const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
        NewOpc = getCMovFromCond(NewCC, TRI->getRegSizeInBits(*DstRC)/8,
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
    MachineBasicBlock *MBB = CmpInstr.getParent();
    for (MachineBasicBlock *Successor : MBB->successors())
      if (Successor->isLiveIn(X86::EFLAGS))
        return false;
  }

  // The instruction to be updated is either Sub or MI.
  Sub = IsCmpZero ? MI : Sub;
  // Move Movr0Inst to the appropriate place before Sub.
  if (Movr0Inst) {
    // Look backwards until we find a def that doesn't use the current EFLAGS.
    Def = Sub;
    MachineBasicBlock::reverse_iterator InsertI = Def.getReverse(),
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

  CmpInstr.eraseFromParent();

  // Modify the condition code of instructions in OpsToUpdate.
  for (auto &Op : OpsToUpdate)
    Op.first->setDesc(get(Op.second));
  return true;
}

/// Try to remove the load by folding it to a register
/// operand at the use. We fold the load instructions if load defines a virtual
/// register, the virtual register is used once in the same BB, and the
/// instructions in-between do not load or store, and have no side effects.
MachineInstr *X86InstrInfo::optimizeLoadInstr(MachineInstr &MI,
                                              const MachineRegisterInfo *MRI,
                                              unsigned &FoldAsLoadDefReg,
                                              MachineInstr *&DefMI) const {
  // Check whether we can move DefMI here.
  DefMI = MRI->getVRegDef(FoldAsLoadDefReg);
  assert(DefMI);
  bool SawStore = false;
  if (!DefMI->isSafeToMove(nullptr, SawStore))
    return nullptr;

  // Collect information about virtual register operands of MI.
  SmallVector<unsigned, 1> SrcOperandIds;
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg != FoldAsLoadDefReg)
      continue;
    // Do not fold if we have a subreg use or a def.
    if (MO.getSubReg() || MO.isDef())
      return nullptr;
    SrcOperandIds.push_back(i);
  }
  if (SrcOperandIds.empty())
    return nullptr;

  // Check whether we can fold the def into SrcOperandId.
  if (MachineInstr *FoldMI = foldMemoryOperand(MI, SrcOperandIds, *DefMI)) {
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

/// Expand a single-def pseudo instruction to a two-addr
/// instruction with two %k0 reads.
/// This is used for mapping:
///   %k4 = K_SET1
/// to:
///   %k4 = KXNORrr %k0, %k0
static bool Expand2AddrKreg(MachineInstrBuilder &MIB,
                            const MCInstrDesc &Desc, unsigned Reg) {
  assert(Desc.getNumOperands() == 3 && "Expected two-addr instruction.");
  MIB->setDesc(Desc);
  MIB.addReg(Reg, RegState::Undef).addReg(Reg, RegState::Undef);
  return true;
}

static bool expandMOV32r1(MachineInstrBuilder &MIB, const TargetInstrInfo &TII,
                          bool MinusOne) {
  MachineBasicBlock &MBB = *MIB->getParent();
  DebugLoc DL = MIB->getDebugLoc();
  unsigned Reg = MIB->getOperand(0).getReg();

  // Insert the XOR.
  BuildMI(MBB, MIB.getInstr(), DL, TII.get(X86::XOR32rr), Reg)
      .addReg(Reg, RegState::Undef)
      .addReg(Reg, RegState::Undef);

  // Turn the pseudo into an INC or DEC.
  MIB->setDesc(TII.get(MinusOne ? X86::DEC32r : X86::INC32r));
  MIB.addReg(Reg);

  return true;
}

static bool ExpandMOVImmSExti8(MachineInstrBuilder &MIB,
                               const TargetInstrInfo &TII,
                               const X86Subtarget &Subtarget) {
  MachineBasicBlock &MBB = *MIB->getParent();
  DebugLoc DL = MIB->getDebugLoc();
  int64_t Imm = MIB->getOperand(1).getImm();
  assert(Imm != 0 && "Using push/pop for 0 is not efficient.");
  MachineBasicBlock::iterator I = MIB.getInstr();

  int StackAdjustment;

  if (Subtarget.is64Bit()) {
    assert(MIB->getOpcode() == X86::MOV64ImmSExti8 ||
           MIB->getOpcode() == X86::MOV32ImmSExti8);

    // Can't use push/pop lowering if the function might write to the red zone.
    X86MachineFunctionInfo *X86FI =
        MBB.getParent()->getInfo<X86MachineFunctionInfo>();
    if (X86FI->getUsesRedZone()) {
      MIB->setDesc(TII.get(MIB->getOpcode() ==
                           X86::MOV32ImmSExti8 ? X86::MOV32ri : X86::MOV64ri));
      return true;
    }

    // 64-bit mode doesn't have 32-bit push/pop, so use 64-bit operations and
    // widen the register if necessary.
    StackAdjustment = 8;
    BuildMI(MBB, I, DL, TII.get(X86::PUSH64i8)).addImm(Imm);
    MIB->setDesc(TII.get(X86::POP64r));
    MIB->getOperand(0)
        .setReg(getX86SubSuperRegister(MIB->getOperand(0).getReg(), 64));
  } else {
    assert(MIB->getOpcode() == X86::MOV32ImmSExti8);
    StackAdjustment = 4;
    BuildMI(MBB, I, DL, TII.get(X86::PUSH32i8)).addImm(Imm);
    MIB->setDesc(TII.get(X86::POP32r));
  }

  // Build CFI if necessary.
  MachineFunction &MF = *MBB.getParent();
  const X86FrameLowering *TFL = Subtarget.getFrameLowering();
  bool IsWin64Prologue = MF.getTarget().getMCAsmInfo()->usesWindowsCFI();
  bool NeedsDwarfCFI =
      !IsWin64Prologue &&
      (MF.getMMI().hasDebugInfo() || MF.getFunction()->needsUnwindTableEntry());
  bool EmitCFI = !TFL->hasFP(MF) && NeedsDwarfCFI;
  if (EmitCFI) {
    TFL->BuildCFI(MBB, I, DL,
        MCCFIInstruction::createAdjustCfaOffset(nullptr, StackAdjustment));
    TFL->BuildCFI(MBB, std::next(I), DL,
        MCCFIInstruction::createAdjustCfaOffset(nullptr, -StackAdjustment));
  }

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
  auto Flags = MachineMemOperand::MOLoad |
               MachineMemOperand::MODereferenceable |
               MachineMemOperand::MOInvariant;
  MachineMemOperand *MMO = MBB.getParent()->getMachineMemOperand(
      MachinePointerInfo::getGOT(*MBB.getParent()), Flags, 8, 8);
  MachineBasicBlock::iterator I = MIB.getInstr();

  BuildMI(MBB, I, DL, TII.get(X86::MOV64rm), Reg).addReg(X86::RIP).addImm(1)
      .addReg(0).addGlobalAddress(GV, 0, X86II::MO_GOTPCREL).addReg(0)
      .addMemOperand(MMO);
  MIB->setDebugLoc(DL);
  MIB->setDesc(TII.get(X86::MOV64rm));
  MIB.addReg(Reg, RegState::Kill).addImm(1).addReg(0).addImm(0).addReg(0);
}

// This is used to handle spills for 128/256-bit registers when we have AVX512,
// but not VLX. If it uses an extended register we need to use an instruction
// that loads the lower 128/256-bit, but is available with only AVX512F.
static bool expandNOVLXLoad(MachineInstrBuilder &MIB,
                            const TargetRegisterInfo *TRI,
                            const MCInstrDesc &LoadDesc,
                            const MCInstrDesc &BroadcastDesc,
                            unsigned SubIdx) {
  unsigned DestReg = MIB->getOperand(0).getReg();
  // Check if DestReg is XMM16-31 or YMM16-31.
  if (TRI->getEncodingValue(DestReg) < 16) {
    // We can use a normal VEX encoded load.
    MIB->setDesc(LoadDesc);
  } else {
    // Use a 128/256-bit VBROADCAST instruction.
    MIB->setDesc(BroadcastDesc);
    // Change the destination to a 512-bit register.
    DestReg = TRI->getMatchingSuperReg(DestReg, SubIdx, &X86::VR512RegClass);
    MIB->getOperand(0).setReg(DestReg);
  }
  return true;
}

// This is used to handle spills for 128/256-bit registers when we have AVX512,
// but not VLX. If it uses an extended register we need to use an instruction
// that stores the lower 128/256-bit, but is available with only AVX512F.
static bool expandNOVLXStore(MachineInstrBuilder &MIB,
                             const TargetRegisterInfo *TRI,
                             const MCInstrDesc &StoreDesc,
                             const MCInstrDesc &ExtractDesc,
                             unsigned SubIdx) {
  unsigned SrcReg = MIB->getOperand(X86::AddrNumOperands).getReg();
  // Check if DestReg is XMM16-31 or YMM16-31.
  if (TRI->getEncodingValue(SrcReg) < 16) {
    // We can use a normal VEX encoded store.
    MIB->setDesc(StoreDesc);
  } else {
    // Use a VEXTRACTF instruction.
    MIB->setDesc(ExtractDesc);
    // Change the destination to a 512-bit register.
    SrcReg = TRI->getMatchingSuperReg(SrcReg, SubIdx, &X86::VR512RegClass);
    MIB->getOperand(X86::AddrNumOperands).setReg(SrcReg);
    MIB.addImm(0x0); // Append immediate to extract from the lower bits.
  }

  return true;
}
bool X86InstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  bool HasAVX = Subtarget.hasAVX();
  MachineInstrBuilder MIB(*MI.getParent()->getParent(), MI);
  switch (MI.getOpcode()) {
  case X86::MOV32r0:
    return Expand2AddrUndef(MIB, get(X86::XOR32rr));
  case X86::MOV32r1:
    return expandMOV32r1(MIB, *this, /*MinusOne=*/ false);
  case X86::MOV32r_1:
    return expandMOV32r1(MIB, *this, /*MinusOne=*/ true);
  case X86::MOV32ImmSExti8:
  case X86::MOV64ImmSExti8:
    return ExpandMOVImmSExti8(MIB, *this, Subtarget);
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
  case X86::AVX512_128_SET0:
  case X86::AVX512_FsFLD0SS:
  case X86::AVX512_FsFLD0SD: {
    bool HasVLX = Subtarget.hasVLX();
    unsigned SrcReg = MIB->getOperand(0).getReg();
    const TargetRegisterInfo *TRI = &getRegisterInfo();
    if (HasVLX || TRI->getEncodingValue(SrcReg) < 16)
      return Expand2AddrUndef(MIB,
                              get(HasVLX ? X86::VPXORDZ128rr : X86::VXORPSrr));
    // Extended register without VLX. Use a larger XOR.
    SrcReg = TRI->getMatchingSuperReg(SrcReg, X86::sub_xmm, &X86::VR512RegClass);
    MIB->getOperand(0).setReg(SrcReg);
    return Expand2AddrUndef(MIB, get(X86::VPXORDZrr));
  }
  case X86::AVX512_256_SET0: {
    bool HasVLX = Subtarget.hasVLX();
    unsigned SrcReg = MIB->getOperand(0).getReg();
    const TargetRegisterInfo *TRI = &getRegisterInfo();
    if (HasVLX || TRI->getEncodingValue(SrcReg) < 16)
      return Expand2AddrUndef(MIB,
                              get(HasVLX ? X86::VPXORDZ256rr : X86::VXORPSYrr));
    // Extended register without VLX. Use a larger XOR.
    SrcReg = TRI->getMatchingSuperReg(SrcReg, X86::sub_ymm, &X86::VR512RegClass);
    MIB->getOperand(0).setReg(SrcReg);
    return Expand2AddrUndef(MIB, get(X86::VPXORDZrr));
  }
  case X86::AVX512_512_SET0:
    return Expand2AddrUndef(MIB, get(X86::VPXORDZrr));
  case X86::V_SETALLONES:
    return Expand2AddrUndef(MIB, get(HasAVX ? X86::VPCMPEQDrr : X86::PCMPEQDrr));
  case X86::AVX2_SETALLONES:
    return Expand2AddrUndef(MIB, get(X86::VPCMPEQDYrr));
  case X86::AVX512_512_SETALLONES: {
    unsigned Reg = MIB->getOperand(0).getReg();
    MIB->setDesc(get(X86::VPTERNLOGDZrri));
    // VPTERNLOGD needs 3 register inputs and an immediate.
    // 0xff will return 1s for any input.
    MIB.addReg(Reg, RegState::Undef).addReg(Reg, RegState::Undef)
       .addReg(Reg, RegState::Undef).addImm(0xff);
    return true;
  }
  case X86::AVX512_512_SEXT_MASK_32:
  case X86::AVX512_512_SEXT_MASK_64: {
    unsigned Reg = MIB->getOperand(0).getReg();
    unsigned MaskReg = MIB->getOperand(1).getReg();
    unsigned MaskState = getRegState(MIB->getOperand(1));
    unsigned Opc = (MI.getOpcode() == X86::AVX512_512_SEXT_MASK_64) ?
                   X86::VPTERNLOGQZrrikz : X86::VPTERNLOGDZrrikz;
    MI.RemoveOperand(1);
    MIB->setDesc(get(Opc));
    // VPTERNLOG needs 3 register inputs and an immediate.
    // 0xff will return 1s for any input.
    MIB.addReg(Reg, RegState::Undef).addReg(MaskReg, MaskState)
       .addReg(Reg, RegState::Undef).addReg(Reg, RegState::Undef).addImm(0xff);
    return true;
  }
  case X86::VMOVAPSZ128rm_NOVLX:
    return expandNOVLXLoad(MIB, &getRegisterInfo(), get(X86::VMOVAPSrm),
                           get(X86::VBROADCASTF32X4rm), X86::sub_xmm);
  case X86::VMOVUPSZ128rm_NOVLX:
    return expandNOVLXLoad(MIB, &getRegisterInfo(), get(X86::VMOVUPSrm),
                           get(X86::VBROADCASTF32X4rm), X86::sub_xmm);
  case X86::VMOVAPSZ256rm_NOVLX:
    return expandNOVLXLoad(MIB, &getRegisterInfo(), get(X86::VMOVAPSYrm),
                           get(X86::VBROADCASTF64X4rm), X86::sub_ymm);
  case X86::VMOVUPSZ256rm_NOVLX:
    return expandNOVLXLoad(MIB, &getRegisterInfo(), get(X86::VMOVUPSYrm),
                           get(X86::VBROADCASTF64X4rm), X86::sub_ymm);
  case X86::VMOVAPSZ128mr_NOVLX:
    return expandNOVLXStore(MIB, &getRegisterInfo(), get(X86::VMOVAPSmr),
                            get(X86::VEXTRACTF32x4Zmr), X86::sub_xmm);
  case X86::VMOVUPSZ128mr_NOVLX:
    return expandNOVLXStore(MIB, &getRegisterInfo(), get(X86::VMOVUPSmr),
                            get(X86::VEXTRACTF32x4Zmr), X86::sub_xmm);
  case X86::VMOVAPSZ256mr_NOVLX:
    return expandNOVLXStore(MIB, &getRegisterInfo(), get(X86::VMOVAPSYmr),
                            get(X86::VEXTRACTF64x4Zmr), X86::sub_ymm);
  case X86::VMOVUPSZ256mr_NOVLX:
    return expandNOVLXStore(MIB, &getRegisterInfo(), get(X86::VMOVUPSYmr),
                            get(X86::VEXTRACTF64x4Zmr), X86::sub_ymm);
  case X86::TEST8ri_NOREX:
    MI.setDesc(get(X86::TEST8ri));
    return true;
  case X86::MOV32ri64:
    MI.setDesc(get(X86::MOV32ri));
    return true;

  // KNL does not recognize dependency-breaking idioms for mask registers,
  // so kxnor %k1, %k1, %k2 has a RAW dependence on %k1.
  // Using %k0 as the undef input register is a performance heuristic based
  // on the assumption that %k0 is used less frequently than the other mask
  // registers, since it is not usable as a write mask.
  // FIXME: A more advanced approach would be to choose the best input mask
  // register based on context.
  case X86::KSET0W: return Expand2AddrKreg(MIB, get(X86::KXORWrr), X86::K0);
  case X86::KSET0D: return Expand2AddrKreg(MIB, get(X86::KXORDrr), X86::K0);
  case X86::KSET0Q: return Expand2AddrKreg(MIB, get(X86::KXORQrr), X86::K0);
  case X86::KSET1W: return Expand2AddrKreg(MIB, get(X86::KXNORWrr), X86::K0);
  case X86::KSET1D: return Expand2AddrKreg(MIB, get(X86::KXNORDrr), X86::K0);
  case X86::KSET1Q: return Expand2AddrKreg(MIB, get(X86::KXNORQrr), X86::K0);
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
      MIB.add(MOs[i]);
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
        MIB.add(MO);
      }
    }
  }
}

static MachineInstr *FuseTwoAddrInst(MachineFunction &MF, unsigned Opcode,
                                     ArrayRef<MachineOperand> MOs,
                                     MachineBasicBlock::iterator InsertPt,
                                     MachineInstr &MI,
                                     const TargetInstrInfo &TII) {
  // Create the base instruction with the memory operand as the first part.
  // Omit the implicit operands, something BuildMI can't do.
  MachineInstr *NewMI =
      MF.CreateMachineInstr(TII.get(Opcode), MI.getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);
  addOperands(MIB, MOs);

  // Loop over the rest of the ri operands, converting them over.
  unsigned NumOps = MI.getDesc().getNumOperands() - 2;
  for (unsigned i = 0; i != NumOps; ++i) {
    MachineOperand &MO = MI.getOperand(i + 2);
    MIB.add(MO);
  }
  for (unsigned i = NumOps + 2, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    MIB.add(MO);
  }

  MachineBasicBlock *MBB = InsertPt->getParent();
  MBB->insert(InsertPt, NewMI);

  return MIB;
}

static MachineInstr *FuseInst(MachineFunction &MF, unsigned Opcode,
                              unsigned OpNo, ArrayRef<MachineOperand> MOs,
                              MachineBasicBlock::iterator InsertPt,
                              MachineInstr &MI, const TargetInstrInfo &TII,
                              int PtrOffset = 0) {
  // Omit the implicit operands, something BuildMI can't do.
  MachineInstr *NewMI =
      MF.CreateMachineInstr(TII.get(Opcode), MI.getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (i == OpNo) {
      assert(MO.isReg() && "Expected to fold into reg operand!");
      addOperands(MIB, MOs, PtrOffset);
    } else {
      MIB.add(MO);
    }
  }

  MachineBasicBlock *MBB = InsertPt->getParent();
  MBB->insert(InsertPt, NewMI);

  return MIB;
}

static MachineInstr *MakeM0Inst(const TargetInstrInfo &TII, unsigned Opcode,
                                ArrayRef<MachineOperand> MOs,
                                MachineBasicBlock::iterator InsertPt,
                                MachineInstr &MI) {
  MachineInstrBuilder MIB = BuildMI(*InsertPt->getParent(), InsertPt,
                                    MI.getDebugLoc(), TII.get(Opcode));
  addOperands(MIB, MOs);
  return MIB.addImm(0);
}

MachineInstr *X86InstrInfo::foldMemoryOperandCustom(
    MachineFunction &MF, MachineInstr &MI, unsigned OpNum,
    ArrayRef<MachineOperand> MOs, MachineBasicBlock::iterator InsertPt,
    unsigned Size, unsigned Align) const {
  switch (MI.getOpcode()) {
  case X86::INSERTPSrr:
  case X86::VINSERTPSrr:
  case X86::VINSERTPSZrr:
    // Attempt to convert the load of inserted vector into a fold load
    // of a single float.
    if (OpNum == 2) {
      unsigned Imm = MI.getOperand(MI.getNumOperands() - 1).getImm();
      unsigned ZMask = Imm & 15;
      unsigned DstIdx = (Imm >> 4) & 3;
      unsigned SrcIdx = (Imm >> 6) & 3;

      const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
      const TargetRegisterClass *RC = getRegClass(MI.getDesc(), OpNum, &RI, MF);
      unsigned RCSize = TRI.getRegSizeInBits(*RC) / 8;
      if (Size <= RCSize && 4 <= Align) {
        int PtrOffset = SrcIdx * 4;
        unsigned NewImm = (DstIdx << 4) | ZMask;
        unsigned NewOpCode =
            (MI.getOpcode() == X86::VINSERTPSZrr) ? X86::VINSERTPSZrm :
            (MI.getOpcode() == X86::VINSERTPSrr)  ? X86::VINSERTPSrm  :
                                                    X86::INSERTPSrm;
        MachineInstr *NewMI =
            FuseInst(MF, NewOpCode, OpNum, MOs, InsertPt, MI, *this, PtrOffset);
        NewMI->getOperand(NewMI->getNumOperands() - 1).setImm(NewImm);
        return NewMI;
      }
    }
    break;
  case X86::MOVHLPSrr:
  case X86::VMOVHLPSrr:
  case X86::VMOVHLPSZrr:
    // Move the upper 64-bits of the second operand to the lower 64-bits.
    // To fold the load, adjust the pointer to the upper and use (V)MOVLPS.
    // TODO: In most cases AVX doesn't have a 8-byte alignment requirement.
    if (OpNum == 2) {
      const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
      const TargetRegisterClass *RC = getRegClass(MI.getDesc(), OpNum, &RI, MF);
      unsigned RCSize = TRI.getRegSizeInBits(*RC) / 8;
      if (Size <= RCSize && 8 <= Align) {
        unsigned NewOpCode =
            (MI.getOpcode() == X86::VMOVHLPSZrr) ? X86::VMOVLPSZ128rm :
            (MI.getOpcode() == X86::VMOVHLPSrr)  ? X86::VMOVLPSrm     :
                                                   X86::MOVLPSrm;
        MachineInstr *NewMI =
            FuseInst(MF, NewOpCode, OpNum, MOs, InsertPt, MI, *this, 8);
        return NewMI;
      }
    }
    break;
  };

  return nullptr;
}

MachineInstr *X86InstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, unsigned OpNum,
    ArrayRef<MachineOperand> MOs, MachineBasicBlock::iterator InsertPt,
    unsigned Size, unsigned Align, bool AllowCommute) const {
  const DenseMap<unsigned,
                 std::pair<uint16_t, uint16_t> > *OpcodeTablePtr = nullptr;
  bool isCallRegIndirect = Subtarget.callRegIndirect();
  bool isTwoAddrFold = false;

  // For CPUs that favor the register form of a call or push,
  // do not fold loads into calls or pushes, unless optimizing for size
  // aggressively.
  if (isCallRegIndirect && !MF.getFunction()->optForMinSize() &&
      (MI.getOpcode() == X86::CALL32r || MI.getOpcode() == X86::CALL64r ||
       MI.getOpcode() == X86::PUSH16r || MI.getOpcode() == X86::PUSH32r ||
       MI.getOpcode() == X86::PUSH64r))
    return nullptr;

  unsigned NumOps = MI.getDesc().getNumOperands();
  bool isTwoAddr =
      NumOps > 1 && MI.getDesc().getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // FIXME: AsmPrinter doesn't know how to handle
  // X86II::MO_GOT_ABSOLUTE_ADDRESS after folding.
  if (MI.getOpcode() == X86::ADD32ri &&
      MI.getOperand(2).getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS)
    return nullptr;

  MachineInstr *NewMI = nullptr;

  // Attempt to fold any custom cases we have.
  if (MachineInstr *CustomMI =
          foldMemoryOperandCustom(MF, MI, OpNum, MOs, InsertPt, Size, Align))
    return CustomMI;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (isTwoAddr && NumOps >= 2 && OpNum < 2 && MI.getOperand(0).isReg() &&
      MI.getOperand(1).isReg() &&
      MI.getOperand(0).getReg() == MI.getOperand(1).getReg()) {
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
    isTwoAddrFold = true;
  } else if (OpNum == 0) {
    if (MI.getOpcode() == X86::MOV32r0) {
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
    auto I = OpcodeTablePtr->find(MI.getOpcode());
    if (I != OpcodeTablePtr->end()) {
      unsigned Opcode = I->second.first;
      unsigned MinAlign = (I->second.second & TB_ALIGN_MASK) >> TB_ALIGN_SHIFT;
      if (Align < MinAlign)
        return nullptr;
      bool NarrowToMOV32rm = false;
      if (Size) {
        const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
        const TargetRegisterClass *RC = getRegClass(MI.getDesc(), OpNum,
                                                    &RI, MF);
        unsigned RCSize = TRI.getRegSizeInBits(*RC) / 8;
        if (Size < RCSize) {
          // Check if it's safe to fold the load. If the size of the object is
          // narrower than the load width, then it's not.
          if (Opcode != X86::MOV64rm || RCSize != 8 || Size != 4)
            return nullptr;
          // If this is a 64-bit load, but the spill slot is 32, then we can do
          // a 32-bit load which is implicitly zero-extended. This likely is
          // due to live interval analysis remat'ing a load from stack slot.
          if (MI.getOperand(0).getSubReg() || MI.getOperand(1).getSubReg())
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
      bool HasDef = MI.getDesc().getNumDefs();
      unsigned Reg0 = HasDef ? MI.getOperand(0).getReg() : 0;
      unsigned Reg1 = MI.getOperand(CommuteOpIdx1).getReg();
      unsigned Reg2 = MI.getOperand(CommuteOpIdx2).getReg();
      bool Tied1 =
          0 == MI.getDesc().getOperandConstraint(CommuteOpIdx1, MCOI::TIED_TO);
      bool Tied2 =
          0 == MI.getDesc().getOperandConstraint(CommuteOpIdx2, MCOI::TIED_TO);

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
      if (CommutedMI != &MI) {
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
      if (UncommutedMI != &MI) {
        // New instruction. It doesn't need to be kept.
        UncommutedMI->eraseFromParent();
        return nullptr;
      }

      // Return here to prevent duplicate fuse failure report.
      return nullptr;
    }
  }

  // No fusion
  if (PrintFailedFusing && !MI.isCopy())
    dbgs() << "We failed to fuse operand " << OpNum << " in " << MI;
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
  case X86::CVTSS2SDrr:
  case X86::CVTSS2SDrm:
  case X86::MOVHPDrm:
  case X86::MOVHPSrm:
  case X86::MOVLPDrm:
  case X86::MOVLPSrm:
  case X86::RCPSSr:
  case X86::RCPSSm:
  case X86::RCPSSr_Int:
  case X86::RCPSSm_Int:
  case X86::ROUNDSDr:
  case X86::ROUNDSDm:
  case X86::ROUNDSSr:
  case X86::ROUNDSSm:
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

/// Inform the ExecutionDepsFix pass how many idle
/// instructions we would like before a partial register update.
unsigned X86InstrInfo::getPartialRegUpdateClearance(
    const MachineInstr &MI, unsigned OpNum,
    const TargetRegisterInfo *TRI) const {
  if (OpNum != 0 || !hasPartialRegUpdate(MI.getOpcode()))
    return 0;

  // If MI is marked as reading Reg, the partial register update is wanted.
  const MachineOperand &MO = MI.getOperand(0);
  unsigned Reg = MO.getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    if (MO.readsReg() || MI.readsVirtualRegister(Reg))
      return 0;
  } else {
    if (MI.readsRegister(Reg, TRI))
      return 0;
  }

  // If any instructions in the clearance range are reading Reg, insert a
  // dependency breaking instruction, which is inexpensive and is likely to
  // be hidden in other instruction's cycles.
  return PartialRegUpdateClearance;
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
  case X86::VRCPSSr_Int:
  case X86::VRCPSSm:
  case X86::VRCPSSm_Int:
  case X86::VROUNDSDr:
  case X86::VROUNDSDm:
  case X86::VROUNDSDr_Int:
  case X86::VROUNDSDm_Int:
  case X86::VROUNDSSr:
  case X86::VROUNDSSm:
  case X86::VROUNDSSr_Int:
  case X86::VROUNDSSm_Int:
  case X86::VRSQRTSSr:
  case X86::VRSQRTSSr_Int:
  case X86::VRSQRTSSm:
  case X86::VRSQRTSSm_Int:
  case X86::VSQRTSSr:
  case X86::VSQRTSSr_Int:
  case X86::VSQRTSSm:
  case X86::VSQRTSSm_Int:
  case X86::VSQRTSDr:
  case X86::VSQRTSDr_Int:
  case X86::VSQRTSDm:
  case X86::VSQRTSDm_Int:
  // AVX-512
  case X86::VCVTSI2SSZrr:
  case X86::VCVTSI2SSZrm:
  case X86::VCVTSI2SSZrr_Int:
  case X86::VCVTSI2SSZrrb_Int:
  case X86::VCVTSI2SSZrm_Int:
  case X86::VCVTSI642SSZrr:
  case X86::VCVTSI642SSZrm:
  case X86::VCVTSI642SSZrr_Int:
  case X86::VCVTSI642SSZrrb_Int:
  case X86::VCVTSI642SSZrm_Int:
  case X86::VCVTSI2SDZrr:
  case X86::VCVTSI2SDZrm:
  case X86::VCVTSI2SDZrr_Int:
  case X86::VCVTSI2SDZrrb_Int:
  case X86::VCVTSI2SDZrm_Int:
  case X86::VCVTSI642SDZrr:
  case X86::VCVTSI642SDZrm:
  case X86::VCVTSI642SDZrr_Int:
  case X86::VCVTSI642SDZrrb_Int:
  case X86::VCVTSI642SDZrm_Int:
  case X86::VCVTUSI2SSZrr:
  case X86::VCVTUSI2SSZrm:
  case X86::VCVTUSI2SSZrr_Int:
  case X86::VCVTUSI2SSZrrb_Int:
  case X86::VCVTUSI2SSZrm_Int:
  case X86::VCVTUSI642SSZrr:
  case X86::VCVTUSI642SSZrm:
  case X86::VCVTUSI642SSZrr_Int:
  case X86::VCVTUSI642SSZrrb_Int:
  case X86::VCVTUSI642SSZrm_Int:
  case X86::VCVTUSI2SDZrr:
  case X86::VCVTUSI2SDZrm:
  case X86::VCVTUSI2SDZrr_Int:
  case X86::VCVTUSI2SDZrm_Int:
  case X86::VCVTUSI642SDZrr:
  case X86::VCVTUSI642SDZrm:
  case X86::VCVTUSI642SDZrr_Int:
  case X86::VCVTUSI642SDZrrb_Int:
  case X86::VCVTUSI642SDZrm_Int:
  case X86::VCVTSD2SSZrr:
  case X86::VCVTSD2SSZrr_Int:
  case X86::VCVTSD2SSZrrb_Int:
  case X86::VCVTSD2SSZrm:
  case X86::VCVTSD2SSZrm_Int:
  case X86::VCVTSS2SDZrr:
  case X86::VCVTSS2SDZrr_Int:
  case X86::VCVTSS2SDZrrb_Int:
  case X86::VCVTSS2SDZrm:
  case X86::VCVTSS2SDZrm_Int:
  case X86::VRNDSCALESDr:
  case X86::VRNDSCALESDrb:
  case X86::VRNDSCALESDm:
  case X86::VRNDSCALESSr:
  case X86::VRNDSCALESSrb:
  case X86::VRNDSCALESSm:
  case X86::VRCP14SSrr:
  case X86::VRCP14SSrm:
  case X86::VRSQRT14SSrr:
  case X86::VRSQRT14SSrm:
  case X86::VSQRTSSZr:
  case X86::VSQRTSSZr_Int:
  case X86::VSQRTSSZrb_Int:
  case X86::VSQRTSSZm:
  case X86::VSQRTSSZm_Int:
  case X86::VSQRTSDZr:
  case X86::VSQRTSDZr_Int:
  case X86::VSQRTSDZrb_Int:
  case X86::VSQRTSDZm:
  case X86::VSQRTSDZm_Int:
    return true;
  }

  return false;
}

/// Inform the ExecutionDepsFix pass how many idle instructions we would like
/// before certain undef register reads.
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
unsigned
X86InstrInfo::getUndefRegClearance(const MachineInstr &MI, unsigned &OpNum,
                                   const TargetRegisterInfo *TRI) const {
  if (!hasUndefRegUpdate(MI.getOpcode()))
    return 0;

  // Set the OpNum parameter to the first source operand.
  OpNum = 1;

  const MachineOperand &MO = MI.getOperand(OpNum);
  if (MO.isUndef() && TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
    return UndefRegClearance;
  }
  return 0;
}

void X86InstrInfo::breakPartialRegDependency(
    MachineInstr &MI, unsigned OpNum, const TargetRegisterInfo *TRI) const {
  unsigned Reg = MI.getOperand(OpNum).getReg();
  // If MI kills this register, the false dependence is already broken.
  if (MI.killsRegister(Reg, TRI))
    return;

  if (X86::VR128RegClass.contains(Reg)) {
    // These instructions are all floating point domain, so xorps is the best
    // choice.
    unsigned Opc = Subtarget.hasAVX() ? X86::VXORPSrr : X86::XORPSrr;
    BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), get(Opc), Reg)
        .addReg(Reg, RegState::Undef)
        .addReg(Reg, RegState::Undef);
    MI.addRegisterKilled(Reg, TRI, true);
  } else if (X86::VR256RegClass.contains(Reg)) {
    // Use vxorps to clear the full ymm register.
    // It wants to read and write the xmm sub-register.
    unsigned XReg = TRI->getSubReg(Reg, X86::sub_xmm);
    BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), get(X86::VXORPSrr), XReg)
        .addReg(XReg, RegState::Undef)
        .addReg(XReg, RegState::Undef)
        .addReg(Reg, RegState::ImplicitDefine);
    MI.addRegisterKilled(Reg, TRI, true);
  }
}

MachineInstr *
X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF, MachineInstr &MI,
                                    ArrayRef<unsigned> Ops,
                                    MachineBasicBlock::iterator InsertPt,
                                    int FrameIndex, LiveIntervals *LIS) const {
  // Check switch flag
  if (NoFusing)
    return nullptr;

  // Unless optimizing for size, don't fold to avoid partial
  // register update stalls
  if (!MF.getFunction()->optForSize() && hasPartialRegUpdate(MI.getOpcode()))
    return nullptr;

  // Don't fold subreg spills, or reloads that use a high subreg.
  for (auto Op : Ops) {
    MachineOperand &MO = MI.getOperand(Op);
    auto SubReg = MO.getSubReg();
    if (SubReg && (MO.isDef() || SubReg == X86::sub_8bit_hi))
      return nullptr;
  }

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned Size = MFI.getObjectSize(FrameIndex);
  unsigned Alignment = MFI.getObjectAlignment(FrameIndex);
  // If the function stack isn't realigned we don't want to fold instructions
  // that need increased alignment.
  if (!RI.needsStackRealignment(MF))
    Alignment =
        std::min(Alignment, Subtarget.getFrameLowering()->getStackAlignment());
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    unsigned RCSize = 0;
    switch (MI.getOpcode()) {
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
    MI.setDesc(get(NewOpc));
    MI.getOperand(1).ChangeToImmediate(0);
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
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const TargetRegisterClass *RC = 
      MF.getRegInfo().getRegClass(LoadMI.getOperand(0).getReg());
  unsigned RegSize = TRI.getRegSizeInBits(*RC);

  if ((Opc == X86::MOVSSrm || Opc == X86::VMOVSSrm || Opc == X86::VMOVSSZrm) &&
      RegSize > 32) {
    // These instructions only load 32 bits, we can't fold them if the
    // destination register is wider than 32 bits (4 bytes), and its user
    // instruction isn't scalar (SS).
    switch (UserOpc) {
    case X86::ADDSSrr_Int: case X86::VADDSSrr_Int: case X86::VADDSSZrr_Int:
    case X86::Int_CMPSSrr: case X86::Int_VCMPSSrr: case X86::VCMPSSZrr_Int:
    case X86::DIVSSrr_Int: case X86::VDIVSSrr_Int: case X86::VDIVSSZrr_Int:
    case X86::MAXSSrr_Int: case X86::VMAXSSrr_Int: case X86::VMAXSSZrr_Int:
    case X86::MINSSrr_Int: case X86::VMINSSrr_Int: case X86::VMINSSZrr_Int:
    case X86::MULSSrr_Int: case X86::VMULSSrr_Int: case X86::VMULSSZrr_Int:
    case X86::SUBSSrr_Int: case X86::VSUBSSrr_Int: case X86::VSUBSSZrr_Int:
    case X86::VADDSSZrr_Intk: case X86::VADDSSZrr_Intkz:
    case X86::VDIVSSZrr_Intk: case X86::VDIVSSZrr_Intkz:
    case X86::VMAXSSZrr_Intk: case X86::VMAXSSZrr_Intkz:
    case X86::VMINSSZrr_Intk: case X86::VMINSSZrr_Intkz:
    case X86::VMULSSZrr_Intk: case X86::VMULSSZrr_Intkz:
    case X86::VSUBSSZrr_Intk: case X86::VSUBSSZrr_Intkz:
    case X86::VFMADDSS4rr_Int:   case X86::VFNMADDSS4rr_Int:
    case X86::VFMSUBSS4rr_Int:   case X86::VFNMSUBSS4rr_Int:
    case X86::VFMADD132SSr_Int:  case X86::VFNMADD132SSr_Int:
    case X86::VFMADD213SSr_Int:  case X86::VFNMADD213SSr_Int:
    case X86::VFMADD231SSr_Int:  case X86::VFNMADD231SSr_Int:
    case X86::VFMSUB132SSr_Int:  case X86::VFNMSUB132SSr_Int:
    case X86::VFMSUB213SSr_Int:  case X86::VFNMSUB213SSr_Int:
    case X86::VFMSUB231SSr_Int:  case X86::VFNMSUB231SSr_Int:
    case X86::VFMADD132SSZr_Int: case X86::VFNMADD132SSZr_Int:
    case X86::VFMADD213SSZr_Int: case X86::VFNMADD213SSZr_Int:
    case X86::VFMADD231SSZr_Int: case X86::VFNMADD231SSZr_Int:
    case X86::VFMSUB132SSZr_Int: case X86::VFNMSUB132SSZr_Int:
    case X86::VFMSUB213SSZr_Int: case X86::VFNMSUB213SSZr_Int:
    case X86::VFMSUB231SSZr_Int: case X86::VFNMSUB231SSZr_Int:
    case X86::VFMADD132SSZr_Intk: case X86::VFNMADD132SSZr_Intk:
    case X86::VFMADD213SSZr_Intk: case X86::VFNMADD213SSZr_Intk:
    case X86::VFMADD231SSZr_Intk: case X86::VFNMADD231SSZr_Intk:
    case X86::VFMSUB132SSZr_Intk: case X86::VFNMSUB132SSZr_Intk:
    case X86::VFMSUB213SSZr_Intk: case X86::VFNMSUB213SSZr_Intk:
    case X86::VFMSUB231SSZr_Intk: case X86::VFNMSUB231SSZr_Intk:
    case X86::VFMADD132SSZr_Intkz: case X86::VFNMADD132SSZr_Intkz:
    case X86::VFMADD213SSZr_Intkz: case X86::VFNMADD213SSZr_Intkz:
    case X86::VFMADD231SSZr_Intkz: case X86::VFNMADD231SSZr_Intkz:
    case X86::VFMSUB132SSZr_Intkz: case X86::VFNMSUB132SSZr_Intkz:
    case X86::VFMSUB213SSZr_Intkz: case X86::VFNMSUB213SSZr_Intkz:
    case X86::VFMSUB231SSZr_Intkz: case X86::VFNMSUB231SSZr_Intkz:
      return false;
    default:
      return true;
    }
  }

  if ((Opc == X86::MOVSDrm || Opc == X86::VMOVSDrm || Opc == X86::VMOVSDZrm) &&
      RegSize > 64) {
    // These instructions only load 64 bits, we can't fold them if the
    // destination register is wider than 64 bits (8 bytes), and its user
    // instruction isn't scalar (SD).
    switch (UserOpc) {
    case X86::ADDSDrr_Int: case X86::VADDSDrr_Int: case X86::VADDSDZrr_Int:
    case X86::Int_CMPSDrr: case X86::Int_VCMPSDrr: case X86::VCMPSDZrr_Int:
    case X86::DIVSDrr_Int: case X86::VDIVSDrr_Int: case X86::VDIVSDZrr_Int:
    case X86::MAXSDrr_Int: case X86::VMAXSDrr_Int: case X86::VMAXSDZrr_Int:
    case X86::MINSDrr_Int: case X86::VMINSDrr_Int: case X86::VMINSDZrr_Int:
    case X86::MULSDrr_Int: case X86::VMULSDrr_Int: case X86::VMULSDZrr_Int:
    case X86::SUBSDrr_Int: case X86::VSUBSDrr_Int: case X86::VSUBSDZrr_Int:
    case X86::VADDSDZrr_Intk: case X86::VADDSDZrr_Intkz:
    case X86::VDIVSDZrr_Intk: case X86::VDIVSDZrr_Intkz:
    case X86::VMAXSDZrr_Intk: case X86::VMAXSDZrr_Intkz:
    case X86::VMINSDZrr_Intk: case X86::VMINSDZrr_Intkz:
    case X86::VMULSDZrr_Intk: case X86::VMULSDZrr_Intkz:
    case X86::VSUBSDZrr_Intk: case X86::VSUBSDZrr_Intkz:
    case X86::VFMADDSD4rr_Int:   case X86::VFNMADDSD4rr_Int:
    case X86::VFMSUBSD4rr_Int:   case X86::VFNMSUBSD4rr_Int:
    case X86::VFMADD132SDr_Int:  case X86::VFNMADD132SDr_Int:
    case X86::VFMADD213SDr_Int:  case X86::VFNMADD213SDr_Int:
    case X86::VFMADD231SDr_Int:  case X86::VFNMADD231SDr_Int:
    case X86::VFMSUB132SDr_Int:  case X86::VFNMSUB132SDr_Int:
    case X86::VFMSUB213SDr_Int:  case X86::VFNMSUB213SDr_Int:
    case X86::VFMSUB231SDr_Int:  case X86::VFNMSUB231SDr_Int:
    case X86::VFMADD132SDZr_Int: case X86::VFNMADD132SDZr_Int:
    case X86::VFMADD213SDZr_Int: case X86::VFNMADD213SDZr_Int:
    case X86::VFMADD231SDZr_Int: case X86::VFNMADD231SDZr_Int:
    case X86::VFMSUB132SDZr_Int: case X86::VFNMSUB132SDZr_Int:
    case X86::VFMSUB213SDZr_Int: case X86::VFNMSUB213SDZr_Int:
    case X86::VFMSUB231SDZr_Int: case X86::VFNMSUB231SDZr_Int:
    case X86::VFMADD132SDZr_Intk: case X86::VFNMADD132SDZr_Intk:
    case X86::VFMADD213SDZr_Intk: case X86::VFNMADD213SDZr_Intk:
    case X86::VFMADD231SDZr_Intk: case X86::VFNMADD231SDZr_Intk:
    case X86::VFMSUB132SDZr_Intk: case X86::VFNMSUB132SDZr_Intk:
    case X86::VFMSUB213SDZr_Intk: case X86::VFNMSUB213SDZr_Intk:
    case X86::VFMSUB231SDZr_Intk: case X86::VFNMSUB231SDZr_Intk:
    case X86::VFMADD132SDZr_Intkz: case X86::VFNMADD132SDZr_Intkz:
    case X86::VFMADD213SDZr_Intkz: case X86::VFNMADD213SDZr_Intkz:
    case X86::VFMADD231SDZr_Intkz: case X86::VFNMADD231SDZr_Intkz:
    case X86::VFMSUB132SDZr_Intkz: case X86::VFNMSUB132SDZr_Intkz:
    case X86::VFMSUB213SDZr_Intkz: case X86::VFNMSUB213SDZr_Intkz:
    case X86::VFMSUB231SDZr_Intkz: case X86::VFNMSUB231SDZr_Intkz:
      return false;
    default:
      return true;
    }
  }

  return false;
}

MachineInstr *X86InstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, MachineInstr &LoadMI,
    LiveIntervals *LIS) const {

  // TODO: Support the case where LoadMI loads a wide register, but MI
  // only uses a subreg.
  for (auto Op : Ops) {
    if (MI.getOperand(Op).getSubReg())
      return nullptr;
  }

  // If loading from a FrameIndex, fold directly from the FrameIndex.
  unsigned NumOps = LoadMI.getDesc().getNumOperands();
  int FrameIndex;
  if (isLoadFromStackSlot(LoadMI, FrameIndex)) {
    if (isNonFoldablePartialRegisterLoad(LoadMI, MI, MF))
      return nullptr;
    return foldMemoryOperandImpl(MF, MI, Ops, InsertPt, FrameIndex, LIS);
  }

  // Check switch flag
  if (NoFusing) return nullptr;

  // Avoid partial register update stalls unless optimizing for size.
  if (!MF.getFunction()->optForSize() && hasPartialRegUpdate(MI.getOpcode()))
    return nullptr;

  // Determine the alignment of the load.
  unsigned Alignment = 0;
  if (LoadMI.hasOneMemOperand())
    Alignment = (*LoadMI.memoperands_begin())->getAlignment();
  else
    switch (LoadMI.getOpcode()) {
    case X86::AVX512_512_SET0:
    case X86::AVX512_512_SETALLONES:
      Alignment = 64;
      break;
    case X86::AVX2_SETALLONES:
    case X86::AVX_SET0:
    case X86::AVX512_256_SET0:
      Alignment = 32;
      break;
    case X86::V_SET0:
    case X86::V_SETALLONES:
    case X86::AVX512_128_SET0:
      Alignment = 16;
      break;
    case X86::FsFLD0SD:
    case X86::AVX512_FsFLD0SD:
      Alignment = 8;
      break;
    case X86::FsFLD0SS:
    case X86::AVX512_FsFLD0SS:
      Alignment = 4;
      break;
    default:
      return nullptr;
    }
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI.getOpcode()) {
    default: return nullptr;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; break;
    }
    // Change to CMPXXri r, 0 first.
    MI.setDesc(get(NewOpc));
    MI.getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return nullptr;

  // Make sure the subregisters match.
  // Otherwise we risk changing the size of the load.
  if (LoadMI.getOperand(0).getSubReg() != MI.getOperand(Ops[0]).getSubReg())
    return nullptr;

  SmallVector<MachineOperand,X86::AddrNumOperands> MOs;
  switch (LoadMI.getOpcode()) {
  case X86::V_SET0:
  case X86::V_SETALLONES:
  case X86::AVX2_SETALLONES:
  case X86::AVX_SET0:
  case X86::AVX512_128_SET0:
  case X86::AVX512_256_SET0:
  case X86::AVX512_512_SET0:
  case X86::AVX512_512_SETALLONES:
  case X86::FsFLD0SD:
  case X86::AVX512_FsFLD0SD:
  case X86::FsFLD0SS:
  case X86::AVX512_FsFLD0SS: {
    // Folding a V_SET0 or V_SETALLONES as a load, to ease register pressure.
    // Create a constant-pool entry and operands to load from it.

    // Medium and large mode can't fold loads this way.
    if (MF.getTarget().getCodeModel() != CodeModel::Small &&
        MF.getTarget().getCodeModel() != CodeModel::Kernel)
      return nullptr;

    // x86-32 PIC requires a PIC base register for constant pools.
    unsigned PICBase = 0;
    if (MF.getTarget().isPositionIndependent()) {
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
    unsigned Opc = LoadMI.getOpcode();
    if (Opc == X86::FsFLD0SS || Opc == X86::AVX512_FsFLD0SS)
      Ty = Type::getFloatTy(MF.getFunction()->getContext());
    else if (Opc == X86::FsFLD0SD || Opc == X86::AVX512_FsFLD0SD)
      Ty = Type::getDoubleTy(MF.getFunction()->getContext());
    else if (Opc == X86::AVX512_512_SET0 || Opc == X86::AVX512_512_SETALLONES)
      Ty = VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()),16);
    else if (Opc == X86::AVX2_SETALLONES || Opc == X86::AVX_SET0 ||
             Opc == X86::AVX512_256_SET0)
      Ty = VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()), 8);
    else
      Ty = VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()), 4);

    bool IsAllOnes = (Opc == X86::V_SETALLONES || Opc == X86::AVX2_SETALLONES ||
                      Opc == X86::AVX512_512_SETALLONES);
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
    if (isNonFoldablePartialRegisterLoad(LoadMI, MI, MF))
      return nullptr;

    // Folding a normal load. Just copy the load's address operands.
    MOs.append(LoadMI.operands_begin() + NumOps - X86::AddrNumOperands,
               LoadMI.operands_begin() + NumOps);
    break;
  }
  }
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, InsertPt,
                               /*Size=*/0, Alignment, /*AllowCommute=*/true);
}

bool X86InstrInfo::unfoldMemoryOperand(
    MachineFunction &MF, MachineInstr &MI, unsigned Reg, bool UnfoldLoad,
    bool UnfoldStore, SmallVectorImpl<MachineInstr *> &NewMIs) const {
  auto I = MemOp2RegOpTable.find(MI.getOpcode());
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
  if (!MI.hasOneMemOperand() && RC == &X86::VR128RegClass &&
      Subtarget.isUnalignedMem16Slow())
    // Without memoperands, loadRegFromAddr and storeRegToStackSlot will
    // conservatively assume the address is unaligned. That's bad for
    // performance.
    return false;
  SmallVector<MachineOperand, X86::AddrNumOperands> AddrOps;
  SmallVector<MachineOperand,2> BeforeOps;
  SmallVector<MachineOperand,2> AfterOps;
  SmallVector<MachineOperand,4> ImpOps;
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &Op = MI.getOperand(i);
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
    std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator> MMOs =
        MF.extractLoadMemRefs(MI.memoperands_begin(), MI.memoperands_end());
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
  MachineInstr *DataMI = MF.CreateMachineInstr(MCID, MI.getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, DataMI);

  if (FoldedStore)
    MIB.addReg(Reg, RegState::Define);
  for (MachineOperand &BeforeOp : BeforeOps)
    MIB.add(BeforeOp);
  if (FoldedLoad)
    MIB.addReg(Reg);
  for (MachineOperand &AfterOp : AfterOps)
    MIB.add(AfterOp);
  for (MachineOperand &ImpOp : ImpOps) {
    MIB.addReg(ImpOp.getReg(),
               getDefRegState(ImpOp.isDef()) |
               RegState::Implicit |
               getKillRegState(ImpOp.isKill()) |
               getDeadRegState(ImpOp.isDead()) |
               getUndefRegState(ImpOp.isUndef()));
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
    std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator> MMOs =
        MF.extractStoreMemRefs(MI.memoperands_begin(), MI.memoperands_end());
    storeRegToAddr(MF, Reg, true, AddrOps, DstRC, MMOs.first, MMOs.second, NewMIs);
  }

  return true;
}

bool
X86InstrInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                  SmallVectorImpl<SDNode*> &NewNodes) const {
  if (!N->isMachineOpcode())
    return false;

  auto I = MemOp2RegOpTable.find(N->getMachineOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & TB_INDEX_MASK;
  bool FoldedLoad = I->second.second & TB_FOLDED_LOAD;
  bool FoldedStore = I->second.second & TB_FOLDED_STORE;
  const MCInstrDesc &MCID = get(Opc);
  MachineFunction &MF = DAG.getMachineFunction();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
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
    EVT VT = *TRI.legalclasstypes_begin(*RC);
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
    unsigned Alignment = std::max<uint32_t>(TRI.getSpillSize(*RC), 16);
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
    VTs.push_back(*TRI.legalclasstypes_begin(*DstRC));
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
    unsigned Alignment = std::max<uint32_t>(TRI.getSpillSize(*RC), 16);
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
  auto I = MemOp2RegOpTable.find(Opc);
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
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVUPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  // AVX load instructions
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVUPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVUPDYrm:
  case X86::VMOVDQAYrm:
  case X86::VMOVDQUYrm:
  // AVX512 load instructions
  case X86::VMOVSSZrm:
  case X86::VMOVSDZrm:
  case X86::VMOVAPSZ128rm:
  case X86::VMOVUPSZ128rm:
  case X86::VMOVAPSZ128rm_NOVLX:
  case X86::VMOVUPSZ128rm_NOVLX:
  case X86::VMOVAPDZ128rm:
  case X86::VMOVUPDZ128rm:
  case X86::VMOVDQU8Z128rm:
  case X86::VMOVDQU16Z128rm:
  case X86::VMOVDQA32Z128rm:
  case X86::VMOVDQU32Z128rm:
  case X86::VMOVDQA64Z128rm:
  case X86::VMOVDQU64Z128rm:
  case X86::VMOVAPSZ256rm:
  case X86::VMOVUPSZ256rm:
  case X86::VMOVAPSZ256rm_NOVLX:
  case X86::VMOVUPSZ256rm_NOVLX:
  case X86::VMOVAPDZ256rm:
  case X86::VMOVUPDZ256rm:
  case X86::VMOVDQU8Z256rm:
  case X86::VMOVDQU16Z256rm:
  case X86::VMOVDQA32Z256rm:
  case X86::VMOVDQU32Z256rm:
  case X86::VMOVDQA64Z256rm:
  case X86::VMOVDQU64Z256rm:
  case X86::VMOVAPSZrm:
  case X86::VMOVUPSZrm:
  case X86::VMOVAPDZrm:
  case X86::VMOVUPDZrm:
  case X86::VMOVDQU8Zrm:
  case X86::VMOVDQU16Zrm:
  case X86::VMOVDQA32Zrm:
  case X86::VMOVDQU32Zrm:
  case X86::VMOVDQA64Zrm:
  case X86::VMOVDQU64Zrm:
  case X86::KMOVBkm:
  case X86::KMOVWkm:
  case X86::KMOVDkm:
  case X86::KMOVQkm:
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
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVUPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
  // AVX load instructions
  case X86::VMOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVAPSrm:
  case X86::VMOVUPSrm:
  case X86::VMOVAPDrm:
  case X86::VMOVUPDrm:
  case X86::VMOVDQArm:
  case X86::VMOVDQUrm:
  case X86::VMOVAPSYrm:
  case X86::VMOVUPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVUPDYrm:
  case X86::VMOVDQAYrm:
  case X86::VMOVDQUYrm:
  // AVX512 load instructions
  case X86::VMOVSSZrm:
  case X86::VMOVSDZrm:
  case X86::VMOVAPSZ128rm:
  case X86::VMOVUPSZ128rm:
  case X86::VMOVAPSZ128rm_NOVLX:
  case X86::VMOVUPSZ128rm_NOVLX:
  case X86::VMOVAPDZ128rm:
  case X86::VMOVUPDZ128rm:
  case X86::VMOVDQU8Z128rm:
  case X86::VMOVDQU16Z128rm:
  case X86::VMOVDQA32Z128rm:
  case X86::VMOVDQU32Z128rm:
  case X86::VMOVDQA64Z128rm:
  case X86::VMOVDQU64Z128rm:
  case X86::VMOVAPSZ256rm:
  case X86::VMOVUPSZ256rm:
  case X86::VMOVAPSZ256rm_NOVLX:
  case X86::VMOVUPSZ256rm_NOVLX:
  case X86::VMOVAPDZ256rm:
  case X86::VMOVUPDZ256rm:
  case X86::VMOVDQU8Z256rm:
  case X86::VMOVDQU16Z256rm:
  case X86::VMOVDQA32Z256rm:
  case X86::VMOVDQU32Z256rm:
  case X86::VMOVDQA64Z256rm:
  case X86::VMOVDQU64Z256rm:
  case X86::VMOVAPSZrm:
  case X86::VMOVUPSZrm:
  case X86::VMOVAPDZrm:
  case X86::VMOVUPDZrm:
  case X86::VMOVDQU8Zrm:
  case X86::VMOVDQU16Zrm:
  case X86::VMOVDQA32Zrm:
  case X86::VMOVDQU32Zrm:
  case X86::VMOVDQA64Zrm:
  case X86::VMOVDQU64Zrm:
  case X86::KMOVBkm:
  case X86::KMOVWkm:
  case X86::KMOVDkm:
  case X86::KMOVQkm:
    break;
  }

  // Lambda to check if both the loads have the same value for an operand index.
  auto HasSameOp = [&](int I) {
    return Load1->getOperand(I) == Load2->getOperand(I);
  };

  // All operands except the displacement should match.
  if (!HasSameOp(X86::AddrBaseReg) || !HasSameOp(X86::AddrScaleAmt) ||
      !HasSameOp(X86::AddrIndexReg) || !HasSameOp(X86::AddrSegmentReg))
    return false;

  // Chain Operand must be the same.
  if (!HasSameOp(5))
    return false;

  // Now let's examine if the displacements are constants.
  auto Disp1 = dyn_cast<ConstantSDNode>(Load1->getOperand(X86::AddrDisp));
  auto Disp2 = dyn_cast<ConstantSDNode>(Load2->getOperand(X86::AddrDisp));
  if (!Disp1 || !Disp2)
    return false;

  Offset1 = Disp1->getSExtValue();
  Offset2 = Disp2->getSExtValue();
  return true;
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

bool X86InstrInfo::
reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid X86 branch condition!");
  X86::CondCode CC = static_cast<X86::CondCode>(Cond[0].getImm());
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
  { X86::MOVLPSmr,   X86::MOVLPDmr,  X86::MOVPQI2QImr },
  { X86::MOVSDmr,    X86::MOVSDmr,   X86::MOVPQI2QImr },
  { X86::MOVSSmr,    X86::MOVSSmr,   X86::MOVPDI2DImr },
  { X86::MOVSDrm,    X86::MOVSDrm,   X86::MOVQI2PQIrm },
  { X86::MOVSSrm,    X86::MOVSSrm,   X86::MOVDI2PDIrm },
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
  { X86::VMOVLPSmr,  X86::VMOVLPDmr,  X86::VMOVPQI2QImr },
  { X86::VMOVSDmr,   X86::VMOVSDmr,   X86::VMOVPQI2QImr },
  { X86::VMOVSSmr,   X86::VMOVSSmr,   X86::VMOVPDI2DImr },
  { X86::VMOVSDrm,   X86::VMOVSDrm,   X86::VMOVQI2PQIrm },
  { X86::VMOVSSrm,   X86::VMOVSSrm,   X86::VMOVDI2PDIrm },
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
  { X86::VMOVNTPSYmr,  X86::VMOVNTPDYmr,  X86::VMOVNTDQYmr },
  // AVX512 support
  { X86::VMOVLPSZ128mr,  X86::VMOVLPDZ128mr,  X86::VMOVPQI2QIZmr  },
  { X86::VMOVNTPSZ128mr, X86::VMOVNTPDZ128mr, X86::VMOVNTDQZ128mr },
  { X86::VMOVNTPSZ256mr, X86::VMOVNTPDZ256mr, X86::VMOVNTDQZ256mr },
  { X86::VMOVNTPSZmr,    X86::VMOVNTPDZmr,    X86::VMOVNTDQZmr    },
  { X86::VMOVSDZmr,      X86::VMOVSDZmr,      X86::VMOVPQI2QIZmr  },
  { X86::VMOVSSZmr,      X86::VMOVSSZmr,      X86::VMOVPDI2DIZmr  },
  { X86::VMOVSDZrm,      X86::VMOVSDZrm,      X86::VMOVQI2PQIZrm  },
  { X86::VMOVSSZrm,      X86::VMOVSSZrm,      X86::VMOVDI2PDIZrm  },
  { X86::VBROADCASTSSZ128r, X86::VBROADCASTSSZ128r, X86::VPBROADCASTDZ128r },
  { X86::VBROADCASTSSZ128m, X86::VBROADCASTSSZ128m, X86::VPBROADCASTDZ128m },
  { X86::VBROADCASTSSZ256r, X86::VBROADCASTSSZ256r, X86::VPBROADCASTDZ256r },
  { X86::VBROADCASTSSZ256m, X86::VBROADCASTSSZ256m, X86::VPBROADCASTDZ256m },
  { X86::VBROADCASTSSZr,    X86::VBROADCASTSSZr,    X86::VPBROADCASTDZr },
  { X86::VBROADCASTSSZm,    X86::VBROADCASTSSZm,    X86::VPBROADCASTDZm },
  { X86::VBROADCASTSDZ256r, X86::VBROADCASTSDZ256r, X86::VPBROADCASTQZ256r },
  { X86::VBROADCASTSDZ256m, X86::VBROADCASTSDZ256m, X86::VPBROADCASTQZ256m },
  { X86::VBROADCASTSDZr,    X86::VBROADCASTSDZr,    X86::VPBROADCASTQZr },
  { X86::VBROADCASTSDZm,    X86::VBROADCASTSDZm,    X86::VPBROADCASTQZm },
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
  { X86::VPERM2F128rm,   X86::VPERM2F128rm,   X86::VPERM2I128rm },
  { X86::VPERM2F128rr,   X86::VPERM2F128rr,   X86::VPERM2I128rr },
  { X86::VBROADCASTSSrm, X86::VBROADCASTSSrm, X86::VPBROADCASTDrm},
  { X86::VBROADCASTSSrr, X86::VBROADCASTSSrr, X86::VPBROADCASTDrr},
  { X86::VBROADCASTSSYrr, X86::VBROADCASTSSYrr, X86::VPBROADCASTDYrr},
  { X86::VBROADCASTSSYrm, X86::VBROADCASTSSYrm, X86::VPBROADCASTDYrm},
  { X86::VBROADCASTSDYrr, X86::VBROADCASTSDYrr, X86::VPBROADCASTQYrr},
  { X86::VBROADCASTSDYrm, X86::VBROADCASTSDYrm, X86::VPBROADCASTQYrm},
  { X86::VBROADCASTF128,  X86::VBROADCASTF128,  X86::VBROADCASTI128 },
};

static const uint16_t ReplaceableInstrsAVX2InsertExtract[][3] = {
  //PackedSingle       PackedDouble       PackedInt
  { X86::VEXTRACTF128mr, X86::VEXTRACTF128mr, X86::VEXTRACTI128mr },
  { X86::VEXTRACTF128rr, X86::VEXTRACTF128rr, X86::VEXTRACTI128rr },
  { X86::VINSERTF128rm,  X86::VINSERTF128rm,  X86::VINSERTI128rm },
  { X86::VINSERTF128rr,  X86::VINSERTF128rr,  X86::VINSERTI128rr },
};

static const uint16_t ReplaceableInstrsAVX512[][4] = {
  // Two integer columns for 64-bit and 32-bit elements.
  //PackedSingle        PackedDouble        PackedInt             PackedInt
  { X86::VMOVAPSZ128mr, X86::VMOVAPDZ128mr, X86::VMOVDQA64Z128mr, X86::VMOVDQA32Z128mr  },
  { X86::VMOVAPSZ128rm, X86::VMOVAPDZ128rm, X86::VMOVDQA64Z128rm, X86::VMOVDQA32Z128rm  },
  { X86::VMOVAPSZ128rr, X86::VMOVAPDZ128rr, X86::VMOVDQA64Z128rr, X86::VMOVDQA32Z128rr  },
  { X86::VMOVUPSZ128mr, X86::VMOVUPDZ128mr, X86::VMOVDQU64Z128mr, X86::VMOVDQU32Z128mr  },
  { X86::VMOVUPSZ128rm, X86::VMOVUPDZ128rm, X86::VMOVDQU64Z128rm, X86::VMOVDQU32Z128rm  },
  { X86::VMOVAPSZ256mr, X86::VMOVAPDZ256mr, X86::VMOVDQA64Z256mr, X86::VMOVDQA32Z256mr  },
  { X86::VMOVAPSZ256rm, X86::VMOVAPDZ256rm, X86::VMOVDQA64Z256rm, X86::VMOVDQA32Z256rm  },
  { X86::VMOVAPSZ256rr, X86::VMOVAPDZ256rr, X86::VMOVDQA64Z256rr, X86::VMOVDQA32Z256rr  },
  { X86::VMOVUPSZ256mr, X86::VMOVUPDZ256mr, X86::VMOVDQU64Z256mr, X86::VMOVDQU32Z256mr  },
  { X86::VMOVUPSZ256rm, X86::VMOVUPDZ256rm, X86::VMOVDQU64Z256rm, X86::VMOVDQU32Z256rm  },
  { X86::VMOVAPSZmr,    X86::VMOVAPDZmr,    X86::VMOVDQA64Zmr,    X86::VMOVDQA32Zmr     },
  { X86::VMOVAPSZrm,    X86::VMOVAPDZrm,    X86::VMOVDQA64Zrm,    X86::VMOVDQA32Zrm     },
  { X86::VMOVAPSZrr,    X86::VMOVAPDZrr,    X86::VMOVDQA64Zrr,    X86::VMOVDQA32Zrr     },
  { X86::VMOVUPSZmr,    X86::VMOVUPDZmr,    X86::VMOVDQU64Zmr,    X86::VMOVDQU32Zmr     },
  { X86::VMOVUPSZrm,    X86::VMOVUPDZrm,    X86::VMOVDQU64Zrm,    X86::VMOVDQU32Zrm     },
};

static const uint16_t ReplaceableInstrsAVX512DQ[][4] = {
  // Two integer columns for 64-bit and 32-bit elements.
  //PackedSingle        PackedDouble        PackedInt           PackedInt
  { X86::VANDNPSZ128rm, X86::VANDNPDZ128rm, X86::VPANDNQZ128rm, X86::VPANDNDZ128rm },
  { X86::VANDNPSZ128rr, X86::VANDNPDZ128rr, X86::VPANDNQZ128rr, X86::VPANDNDZ128rr },
  { X86::VANDPSZ128rm,  X86::VANDPDZ128rm,  X86::VPANDQZ128rm,  X86::VPANDDZ128rm  },
  { X86::VANDPSZ128rr,  X86::VANDPDZ128rr,  X86::VPANDQZ128rr,  X86::VPANDDZ128rr  },
  { X86::VORPSZ128rm,   X86::VORPDZ128rm,   X86::VPORQZ128rm,   X86::VPORDZ128rm   },
  { X86::VORPSZ128rr,   X86::VORPDZ128rr,   X86::VPORQZ128rr,   X86::VPORDZ128rr   },
  { X86::VXORPSZ128rm,  X86::VXORPDZ128rm,  X86::VPXORQZ128rm,  X86::VPXORDZ128rm  },
  { X86::VXORPSZ128rr,  X86::VXORPDZ128rr,  X86::VPXORQZ128rr,  X86::VPXORDZ128rr  },
  { X86::VANDNPSZ256rm, X86::VANDNPDZ256rm, X86::VPANDNQZ256rm, X86::VPANDNDZ256rm },
  { X86::VANDNPSZ256rr, X86::VANDNPDZ256rr, X86::VPANDNQZ256rr, X86::VPANDNDZ256rr },
  { X86::VANDPSZ256rm,  X86::VANDPDZ256rm,  X86::VPANDQZ256rm,  X86::VPANDDZ256rm  },
  { X86::VANDPSZ256rr,  X86::VANDPDZ256rr,  X86::VPANDQZ256rr,  X86::VPANDDZ256rr  },
  { X86::VORPSZ256rm,   X86::VORPDZ256rm,   X86::VPORQZ256rm,   X86::VPORDZ256rm   },
  { X86::VORPSZ256rr,   X86::VORPDZ256rr,   X86::VPORQZ256rr,   X86::VPORDZ256rr   },
  { X86::VXORPSZ256rm,  X86::VXORPDZ256rm,  X86::VPXORQZ256rm,  X86::VPXORDZ256rm  },
  { X86::VXORPSZ256rr,  X86::VXORPDZ256rr,  X86::VPXORQZ256rr,  X86::VPXORDZ256rr  },
  { X86::VANDNPSZrm,    X86::VANDNPDZrm,    X86::VPANDNQZrm,    X86::VPANDNDZrm    },
  { X86::VANDNPSZrr,    X86::VANDNPDZrr,    X86::VPANDNQZrr,    X86::VPANDNDZrr    },
  { X86::VANDPSZrm,     X86::VANDPDZrm,     X86::VPANDQZrm,     X86::VPANDDZrm     },
  { X86::VANDPSZrr,     X86::VANDPDZrr,     X86::VPANDQZrr,     X86::VPANDDZrr     },
  { X86::VORPSZrm,      X86::VORPDZrm,      X86::VPORQZrm,      X86::VPORDZrm      },
  { X86::VORPSZrr,      X86::VORPDZrr,      X86::VPORQZrr,      X86::VPORDZrr      },
  { X86::VXORPSZrm,     X86::VXORPDZrm,     X86::VPXORQZrm,     X86::VPXORDZrm     },
  { X86::VXORPSZrr,     X86::VXORPDZrr,     X86::VPXORQZrr,     X86::VPXORDZrr     },
};

static const uint16_t ReplaceableInstrsAVX512DQMasked[][4] = {
  // Two integer columns for 64-bit and 32-bit elements.
  //PackedSingle          PackedDouble
  //PackedInt             PackedInt
  { X86::VANDNPSZ128rmk,  X86::VANDNPDZ128rmk,
    X86::VPANDNQZ128rmk,  X86::VPANDNDZ128rmk  },
  { X86::VANDNPSZ128rmkz, X86::VANDNPDZ128rmkz,
    X86::VPANDNQZ128rmkz, X86::VPANDNDZ128rmkz },
  { X86::VANDNPSZ128rrk,  X86::VANDNPDZ128rrk,
    X86::VPANDNQZ128rrk,  X86::VPANDNDZ128rrk  },
  { X86::VANDNPSZ128rrkz, X86::VANDNPDZ128rrkz,
    X86::VPANDNQZ128rrkz, X86::VPANDNDZ128rrkz },
  { X86::VANDPSZ128rmk,   X86::VANDPDZ128rmk,
    X86::VPANDQZ128rmk,   X86::VPANDDZ128rmk   },
  { X86::VANDPSZ128rmkz,  X86::VANDPDZ128rmkz,
    X86::VPANDQZ128rmkz,  X86::VPANDDZ128rmkz  },
  { X86::VANDPSZ128rrk,   X86::VANDPDZ128rrk,
    X86::VPANDQZ128rrk,   X86::VPANDDZ128rrk   },
  { X86::VANDPSZ128rrkz,  X86::VANDPDZ128rrkz,
    X86::VPANDQZ128rrkz,  X86::VPANDDZ128rrkz  },
  { X86::VORPSZ128rmk,    X86::VORPDZ128rmk,
    X86::VPORQZ128rmk,    X86::VPORDZ128rmk    },
  { X86::VORPSZ128rmkz,   X86::VORPDZ128rmkz,
    X86::VPORQZ128rmkz,   X86::VPORDZ128rmkz   },
  { X86::VORPSZ128rrk,    X86::VORPDZ128rrk,
    X86::VPORQZ128rrk,    X86::VPORDZ128rrk    },
  { X86::VORPSZ128rrkz,   X86::VORPDZ128rrkz,
    X86::VPORQZ128rrkz,   X86::VPORDZ128rrkz   },
  { X86::VXORPSZ128rmk,   X86::VXORPDZ128rmk,
    X86::VPXORQZ128rmk,   X86::VPXORDZ128rmk   },
  { X86::VXORPSZ128rmkz,  X86::VXORPDZ128rmkz,
    X86::VPXORQZ128rmkz,  X86::VPXORDZ128rmkz  },
  { X86::VXORPSZ128rrk,   X86::VXORPDZ128rrk,
    X86::VPXORQZ128rrk,   X86::VPXORDZ128rrk   },
  { X86::VXORPSZ128rrkz,  X86::VXORPDZ128rrkz,
    X86::VPXORQZ128rrkz,  X86::VPXORDZ128rrkz  },
  { X86::VANDNPSZ256rmk,  X86::VANDNPDZ256rmk,
    X86::VPANDNQZ256rmk,  X86::VPANDNDZ256rmk  },
  { X86::VANDNPSZ256rmkz, X86::VANDNPDZ256rmkz,
    X86::VPANDNQZ256rmkz, X86::VPANDNDZ256rmkz },
  { X86::VANDNPSZ256rrk,  X86::VANDNPDZ256rrk,
    X86::VPANDNQZ256rrk,  X86::VPANDNDZ256rrk  },
  { X86::VANDNPSZ256rrkz, X86::VANDNPDZ256rrkz,
    X86::VPANDNQZ256rrkz, X86::VPANDNDZ256rrkz },
  { X86::VANDPSZ256rmk,   X86::VANDPDZ256rmk,
    X86::VPANDQZ256rmk,   X86::VPANDDZ256rmk   },
  { X86::VANDPSZ256rmkz,  X86::VANDPDZ256rmkz,
    X86::VPANDQZ256rmkz,  X86::VPANDDZ256rmkz  },
  { X86::VANDPSZ256rrk,   X86::VANDPDZ256rrk,
    X86::VPANDQZ256rrk,   X86::VPANDDZ256rrk   },
  { X86::VANDPSZ256rrkz,  X86::VANDPDZ256rrkz,
    X86::VPANDQZ256rrkz,  X86::VPANDDZ256rrkz  },
  { X86::VORPSZ256rmk,    X86::VORPDZ256rmk,
    X86::VPORQZ256rmk,    X86::VPORDZ256rmk    },
  { X86::VORPSZ256rmkz,   X86::VORPDZ256rmkz,
    X86::VPORQZ256rmkz,   X86::VPORDZ256rmkz   },
  { X86::VORPSZ256rrk,    X86::VORPDZ256rrk,
    X86::VPORQZ256rrk,    X86::VPORDZ256rrk    },
  { X86::VORPSZ256rrkz,   X86::VORPDZ256rrkz,
    X86::VPORQZ256rrkz,   X86::VPORDZ256rrkz   },
  { X86::VXORPSZ256rmk,   X86::VXORPDZ256rmk,
    X86::VPXORQZ256rmk,   X86::VPXORDZ256rmk   },
  { X86::VXORPSZ256rmkz,  X86::VXORPDZ256rmkz,
    X86::VPXORQZ256rmkz,  X86::VPXORDZ256rmkz  },
  { X86::VXORPSZ256rrk,   X86::VXORPDZ256rrk,
    X86::VPXORQZ256rrk,   X86::VPXORDZ256rrk   },
  { X86::VXORPSZ256rrkz,  X86::VXORPDZ256rrkz,
    X86::VPXORQZ256rrkz,  X86::VPXORDZ256rrkz  },
  { X86::VANDNPSZrmk,     X86::VANDNPDZrmk,
    X86::VPANDNQZrmk,     X86::VPANDNDZrmk     },
  { X86::VANDNPSZrmkz,    X86::VANDNPDZrmkz,
    X86::VPANDNQZrmkz,    X86::VPANDNDZrmkz    },
  { X86::VANDNPSZrrk,     X86::VANDNPDZrrk,
    X86::VPANDNQZrrk,     X86::VPANDNDZrrk     },
  { X86::VANDNPSZrrkz,    X86::VANDNPDZrrkz,
    X86::VPANDNQZrrkz,    X86::VPANDNDZrrkz    },
  { X86::VANDPSZrmk,      X86::VANDPDZrmk,
    X86::VPANDQZrmk,      X86::VPANDDZrmk      },
  { X86::VANDPSZrmkz,     X86::VANDPDZrmkz,
    X86::VPANDQZrmkz,     X86::VPANDDZrmkz     },
  { X86::VANDPSZrrk,      X86::VANDPDZrrk,
    X86::VPANDQZrrk,      X86::VPANDDZrrk      },
  { X86::VANDPSZrrkz,     X86::VANDPDZrrkz,
    X86::VPANDQZrrkz,     X86::VPANDDZrrkz     },
  { X86::VORPSZrmk,       X86::VORPDZrmk,
    X86::VPORQZrmk,       X86::VPORDZrmk       },
  { X86::VORPSZrmkz,      X86::VORPDZrmkz,
    X86::VPORQZrmkz,      X86::VPORDZrmkz      },
  { X86::VORPSZrrk,       X86::VORPDZrrk,
    X86::VPORQZrrk,       X86::VPORDZrrk       },
  { X86::VORPSZrrkz,      X86::VORPDZrrkz,
    X86::VPORQZrrkz,      X86::VPORDZrrkz      },
  { X86::VXORPSZrmk,      X86::VXORPDZrmk,
    X86::VPXORQZrmk,      X86::VPXORDZrmk      },
  { X86::VXORPSZrmkz,     X86::VXORPDZrmkz,
    X86::VPXORQZrmkz,     X86::VPXORDZrmkz     },
  { X86::VXORPSZrrk,      X86::VXORPDZrrk,
    X86::VPXORQZrrk,      X86::VPXORDZrrk      },
  { X86::VXORPSZrrkz,     X86::VXORPDZrrkz,
    X86::VPXORQZrrkz,     X86::VPXORDZrrkz     },
  // Broadcast loads can be handled the same as masked operations to avoid
  // changing element size.
  { X86::VANDNPSZ128rmb,  X86::VANDNPDZ128rmb,
    X86::VPANDNQZ128rmb,  X86::VPANDNDZ128rmb  },
  { X86::VANDPSZ128rmb,   X86::VANDPDZ128rmb,
    X86::VPANDQZ128rmb,   X86::VPANDDZ128rmb   },
  { X86::VORPSZ128rmb,    X86::VORPDZ128rmb,
    X86::VPORQZ128rmb,    X86::VPORDZ128rmb    },
  { X86::VXORPSZ128rmb,   X86::VXORPDZ128rmb,
    X86::VPXORQZ128rmb,   X86::VPXORDZ128rmb   },
  { X86::VANDNPSZ256rmb,  X86::VANDNPDZ256rmb,
    X86::VPANDNQZ256rmb,  X86::VPANDNDZ256rmb  },
  { X86::VANDPSZ256rmb,   X86::VANDPDZ256rmb,
    X86::VPANDQZ256rmb,   X86::VPANDDZ256rmb   },
  { X86::VORPSZ256rmb,    X86::VORPDZ256rmb,
    X86::VPORQZ256rmb,    X86::VPORDZ256rmb    },
  { X86::VXORPSZ256rmb,   X86::VXORPDZ256rmb,
    X86::VPXORQZ256rmb,   X86::VPXORDZ256rmb   },
  { X86::VANDNPSZrmb,     X86::VANDNPDZrmb,
    X86::VPANDNQZrmb,     X86::VPANDNDZrmb     },
  { X86::VANDPSZrmb,      X86::VANDPDZrmb,
    X86::VPANDQZrmb,      X86::VPANDDZrmb      },
  { X86::VANDPSZrmb,      X86::VANDPDZrmb,
    X86::VPANDQZrmb,      X86::VPANDDZrmb      },
  { X86::VORPSZrmb,       X86::VORPDZrmb,
    X86::VPORQZrmb,       X86::VPORDZrmb       },
  { X86::VXORPSZrmb,      X86::VXORPDZrmb,
    X86::VPXORQZrmb,      X86::VPXORDZrmb      },
  { X86::VANDNPSZ128rmbk, X86::VANDNPDZ128rmbk,
    X86::VPANDNQZ128rmbk, X86::VPANDNDZ128rmbk },
  { X86::VANDPSZ128rmbk,  X86::VANDPDZ128rmbk,
    X86::VPANDQZ128rmbk,  X86::VPANDDZ128rmbk  },
  { X86::VORPSZ128rmbk,   X86::VORPDZ128rmbk,
    X86::VPORQZ128rmbk,   X86::VPORDZ128rmbk   },
  { X86::VXORPSZ128rmbk,  X86::VXORPDZ128rmbk,
    X86::VPXORQZ128rmbk,  X86::VPXORDZ128rmbk  },
  { X86::VANDNPSZ256rmbk, X86::VANDNPDZ256rmbk,
    X86::VPANDNQZ256rmbk, X86::VPANDNDZ256rmbk },
  { X86::VANDPSZ256rmbk,  X86::VANDPDZ256rmbk,
    X86::VPANDQZ256rmbk,  X86::VPANDDZ256rmbk  },
  { X86::VORPSZ256rmbk,   X86::VORPDZ256rmbk,
    X86::VPORQZ256rmbk,   X86::VPORDZ256rmbk   },
  { X86::VXORPSZ256rmbk,  X86::VXORPDZ256rmbk,
    X86::VPXORQZ256rmbk,  X86::VPXORDZ256rmbk  },
  { X86::VANDNPSZrmbk,    X86::VANDNPDZrmbk,
    X86::VPANDNQZrmbk,    X86::VPANDNDZrmbk    },
  { X86::VANDPSZrmbk,     X86::VANDPDZrmbk,
    X86::VPANDQZrmbk,     X86::VPANDDZrmbk     },
  { X86::VANDPSZrmbk,     X86::VANDPDZrmbk,
    X86::VPANDQZrmbk,     X86::VPANDDZrmbk     },
  { X86::VORPSZrmbk,      X86::VORPDZrmbk,
    X86::VPORQZrmbk,      X86::VPORDZrmbk      },
  { X86::VXORPSZrmbk,     X86::VXORPDZrmbk,
    X86::VPXORQZrmbk,     X86::VPXORDZrmbk     },
  { X86::VANDNPSZ128rmbkz,X86::VANDNPDZ128rmbkz,
    X86::VPANDNQZ128rmbkz,X86::VPANDNDZ128rmbkz},
  { X86::VANDPSZ128rmbkz, X86::VANDPDZ128rmbkz,
    X86::VPANDQZ128rmbkz, X86::VPANDDZ128rmbkz },
  { X86::VORPSZ128rmbkz,  X86::VORPDZ128rmbkz,
    X86::VPORQZ128rmbkz,  X86::VPORDZ128rmbkz  },
  { X86::VXORPSZ128rmbkz, X86::VXORPDZ128rmbkz,
    X86::VPXORQZ128rmbkz, X86::VPXORDZ128rmbkz },
  { X86::VANDNPSZ256rmbkz,X86::VANDNPDZ256rmbkz,
    X86::VPANDNQZ256rmbkz,X86::VPANDNDZ256rmbkz},
  { X86::VANDPSZ256rmbkz, X86::VANDPDZ256rmbkz,
    X86::VPANDQZ256rmbkz, X86::VPANDDZ256rmbkz },
  { X86::VORPSZ256rmbkz,  X86::VORPDZ256rmbkz,
    X86::VPORQZ256rmbkz,  X86::VPORDZ256rmbkz  },
  { X86::VXORPSZ256rmbkz, X86::VXORPDZ256rmbkz,
    X86::VPXORQZ256rmbkz, X86::VPXORDZ256rmbkz },
  { X86::VANDNPSZrmbkz,   X86::VANDNPDZrmbkz,
    X86::VPANDNQZrmbkz,   X86::VPANDNDZrmbkz   },
  { X86::VANDPSZrmbkz,    X86::VANDPDZrmbkz,
    X86::VPANDQZrmbkz,    X86::VPANDDZrmbkz    },
  { X86::VANDPSZrmbkz,    X86::VANDPDZrmbkz,
    X86::VPANDQZrmbkz,    X86::VPANDDZrmbkz    },
  { X86::VORPSZrmbkz,     X86::VORPDZrmbkz,
    X86::VPORQZrmbkz,     X86::VPORDZrmbkz     },
  { X86::VXORPSZrmbkz,    X86::VXORPDZrmbkz,
    X86::VPXORQZrmbkz,    X86::VPXORDZrmbkz    },
};

// FIXME: Some shuffle and unpack instructions have equivalents in different
// domains, but they require a bit more work than just switching opcodes.

static const uint16_t *lookup(unsigned opcode, unsigned domain,
                              ArrayRef<uint16_t[3]> Table) {
  for (const uint16_t (&Row)[3] : Table)
    if (Row[domain-1] == opcode)
      return Row;
  return nullptr;
}

static const uint16_t *lookupAVX512(unsigned opcode, unsigned domain,
                                    ArrayRef<uint16_t[4]> Table) {
  // If this is the integer domain make sure to check both integer columns.
  for (const uint16_t (&Row)[4] : Table)
    if (Row[domain-1] == opcode || (domain == 3 && Row[3] == opcode))
      return Row;
  return nullptr;
}

std::pair<uint16_t, uint16_t>
X86InstrInfo::getExecutionDomain(const MachineInstr &MI) const {
  uint16_t domain = (MI.getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  unsigned opcode = MI.getOpcode();
  uint16_t validDomains = 0;
  if (domain) {
    if (lookup(MI.getOpcode(), domain, ReplaceableInstrs)) {
      validDomains = 0xe;
    } else if (lookup(opcode, domain, ReplaceableInstrsAVX2)) {
      validDomains = Subtarget.hasAVX2() ? 0xe : 0x6;
    } else if (lookup(opcode, domain, ReplaceableInstrsAVX2InsertExtract)) {
      // Insert/extract instructions should only effect domain if AVX2
      // is enabled.
      if (!Subtarget.hasAVX2())
        return std::make_pair(0, 0);
      validDomains = 0xe;
    } else if (lookupAVX512(opcode, domain, ReplaceableInstrsAVX512)) {
      validDomains = 0xe;
    } else if (Subtarget.hasDQI() && lookupAVX512(opcode, domain,
                                                  ReplaceableInstrsAVX512DQ)) {
      validDomains = 0xe;
    } else if (Subtarget.hasDQI()) {
      if (const uint16_t *table = lookupAVX512(opcode, domain,
                                             ReplaceableInstrsAVX512DQMasked)) {
        if (domain == 1 || (domain == 3 && table[3] == opcode))
          validDomains = 0xa;
        else
          validDomains = 0xc;
      }
    }
  }
  return std::make_pair(domain, validDomains);
}

void X86InstrInfo::setExecutionDomain(MachineInstr &MI, unsigned Domain) const {
  assert(Domain>0 && Domain<4 && "Invalid execution domain");
  uint16_t dom = (MI.getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  assert(dom && "Not an SSE instruction");
  const uint16_t *table = lookup(MI.getOpcode(), dom, ReplaceableInstrs);
  if (!table) { // try the other table
    assert((Subtarget.hasAVX2() || Domain < 3) &&
           "256-bit vector operations only available in AVX2");
    table = lookup(MI.getOpcode(), dom, ReplaceableInstrsAVX2);
  }
  if (!table) { // try the other table
    assert(Subtarget.hasAVX2() &&
           "256-bit insert/extract only available in AVX2");
    table = lookup(MI.getOpcode(), dom, ReplaceableInstrsAVX2InsertExtract);
  }
  if (!table) { // try the AVX512 table
    assert(Subtarget.hasAVX512() && "Requires AVX-512");
    table = lookupAVX512(MI.getOpcode(), dom, ReplaceableInstrsAVX512);
    // Don't change integer Q instructions to D instructions.
    if (table && Domain == 3 && table[3] == MI.getOpcode())
      Domain = 4;
  }
  if (!table) { // try the AVX512DQ table
    assert((Subtarget.hasDQI() || Domain >= 3) && "Requires AVX-512DQ");
    table = lookupAVX512(MI.getOpcode(), dom, ReplaceableInstrsAVX512DQ);
    // Don't change integer Q instructions to D instructions and
    // use D intructions if we started with a PS instruction.
    if (table && Domain == 3 && (dom == 1 || table[3] == MI.getOpcode()))
      Domain = 4;
  }
  if (!table) { // try the AVX512DQMasked table
    assert((Subtarget.hasDQI() || Domain >= 3) && "Requires AVX-512DQ");
    table = lookupAVX512(MI.getOpcode(), dom, ReplaceableInstrsAVX512DQMasked);
    if (table && Domain == 3 && (dom == 1 || table[3] == MI.getOpcode()))
      Domain = 4;
  }
  assert(table && "Cannot change domain");
  MI.setDesc(get(table[Domain - 1]));
}

/// Return the noop instruction to use for a noop.
void X86InstrInfo::getNoop(MCInst &NopInst) const {
  NopInst.setOpcode(X86::NOOP);
}

bool X86InstrInfo::isHighLatencyDef(int opc) const {
  switch (opc) {
  default: return false;
  case X86::DIVPDrm:
  case X86::DIVPDrr:
  case X86::DIVPSrm:
  case X86::DIVPSrr:
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
  case X86::VDIVPDrm:
  case X86::VDIVPDrr:
  case X86::VDIVPDYrm:
  case X86::VDIVPDYrr:
  case X86::VDIVPSrm:
  case X86::VDIVPSrr:
  case X86::VDIVPSYrm:
  case X86::VDIVPSYrr:
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
  case X86::VSQRTPDYm:
  case X86::VSQRTPDYr:
  case X86::VSQRTPSm:
  case X86::VSQRTPSr:
  case X86::VSQRTPSYm:
  case X86::VSQRTPSYr:
  case X86::VSQRTSDm:
  case X86::VSQRTSDm_Int:
  case X86::VSQRTSDr:
  case X86::VSQRTSDr_Int:
  case X86::VSQRTSSm:
  case X86::VSQRTSSm_Int:
  case X86::VSQRTSSr:
  case X86::VSQRTSSr_Int:
  // AVX512 instructions with high latency
  case X86::VDIVPDZ128rm:
  case X86::VDIVPDZ128rmb:
  case X86::VDIVPDZ128rmbk:
  case X86::VDIVPDZ128rmbkz:
  case X86::VDIVPDZ128rmk:
  case X86::VDIVPDZ128rmkz:
  case X86::VDIVPDZ128rr:
  case X86::VDIVPDZ128rrk:
  case X86::VDIVPDZ128rrkz:
  case X86::VDIVPDZ256rm:
  case X86::VDIVPDZ256rmb:
  case X86::VDIVPDZ256rmbk:
  case X86::VDIVPDZ256rmbkz:
  case X86::VDIVPDZ256rmk:
  case X86::VDIVPDZ256rmkz:
  case X86::VDIVPDZ256rr:
  case X86::VDIVPDZ256rrk:
  case X86::VDIVPDZ256rrkz:
  case X86::VDIVPDZrb:
  case X86::VDIVPDZrbk:
  case X86::VDIVPDZrbkz:
  case X86::VDIVPDZrm:
  case X86::VDIVPDZrmb:
  case X86::VDIVPDZrmbk:
  case X86::VDIVPDZrmbkz:
  case X86::VDIVPDZrmk:
  case X86::VDIVPDZrmkz:
  case X86::VDIVPDZrr:
  case X86::VDIVPDZrrk:
  case X86::VDIVPDZrrkz:
  case X86::VDIVPSZ128rm:
  case X86::VDIVPSZ128rmb:
  case X86::VDIVPSZ128rmbk:
  case X86::VDIVPSZ128rmbkz:
  case X86::VDIVPSZ128rmk:
  case X86::VDIVPSZ128rmkz:
  case X86::VDIVPSZ128rr:
  case X86::VDIVPSZ128rrk:
  case X86::VDIVPSZ128rrkz:
  case X86::VDIVPSZ256rm:
  case X86::VDIVPSZ256rmb:
  case X86::VDIVPSZ256rmbk:
  case X86::VDIVPSZ256rmbkz:
  case X86::VDIVPSZ256rmk:
  case X86::VDIVPSZ256rmkz:
  case X86::VDIVPSZ256rr:
  case X86::VDIVPSZ256rrk:
  case X86::VDIVPSZ256rrkz:
  case X86::VDIVPSZrb:
  case X86::VDIVPSZrbk:
  case X86::VDIVPSZrbkz:
  case X86::VDIVPSZrm:
  case X86::VDIVPSZrmb:
  case X86::VDIVPSZrmbk:
  case X86::VDIVPSZrmbkz:
  case X86::VDIVPSZrmk:
  case X86::VDIVPSZrmkz:
  case X86::VDIVPSZrr:
  case X86::VDIVPSZrrk:
  case X86::VDIVPSZrrkz:
  case X86::VDIVSDZrm:
  case X86::VDIVSDZrr:
  case X86::VDIVSDZrm_Int:
  case X86::VDIVSDZrm_Intk:
  case X86::VDIVSDZrm_Intkz:
  case X86::VDIVSDZrr_Int:
  case X86::VDIVSDZrr_Intk:
  case X86::VDIVSDZrr_Intkz:
  case X86::VDIVSDZrrb:
  case X86::VDIVSDZrrbk:
  case X86::VDIVSDZrrbkz:
  case X86::VDIVSSZrm:
  case X86::VDIVSSZrr:
  case X86::VDIVSSZrm_Int:
  case X86::VDIVSSZrm_Intk:
  case X86::VDIVSSZrm_Intkz:
  case X86::VDIVSSZrr_Int:
  case X86::VDIVSSZrr_Intk:
  case X86::VDIVSSZrr_Intkz:
  case X86::VDIVSSZrrb:
  case X86::VDIVSSZrrbk:
  case X86::VDIVSSZrrbkz:
  case X86::VSQRTPDZ128m:
  case X86::VSQRTPDZ128mb:
  case X86::VSQRTPDZ128mbk:
  case X86::VSQRTPDZ128mbkz:
  case X86::VSQRTPDZ128mk:
  case X86::VSQRTPDZ128mkz:
  case X86::VSQRTPDZ128r:
  case X86::VSQRTPDZ128rk:
  case X86::VSQRTPDZ128rkz:
  case X86::VSQRTPDZ256m:
  case X86::VSQRTPDZ256mb:
  case X86::VSQRTPDZ256mbk:
  case X86::VSQRTPDZ256mbkz:
  case X86::VSQRTPDZ256mk:
  case X86::VSQRTPDZ256mkz:
  case X86::VSQRTPDZ256r:
  case X86::VSQRTPDZ256rk:
  case X86::VSQRTPDZ256rkz:
  case X86::VSQRTPDZm:
  case X86::VSQRTPDZmb:
  case X86::VSQRTPDZmbk:
  case X86::VSQRTPDZmbkz:
  case X86::VSQRTPDZmk:
  case X86::VSQRTPDZmkz:
  case X86::VSQRTPDZr:
  case X86::VSQRTPDZrb:
  case X86::VSQRTPDZrbk:
  case X86::VSQRTPDZrbkz:
  case X86::VSQRTPDZrk:
  case X86::VSQRTPDZrkz:
  case X86::VSQRTPSZ128m:
  case X86::VSQRTPSZ128mb:
  case X86::VSQRTPSZ128mbk:
  case X86::VSQRTPSZ128mbkz:
  case X86::VSQRTPSZ128mk:
  case X86::VSQRTPSZ128mkz:
  case X86::VSQRTPSZ128r:
  case X86::VSQRTPSZ128rk:
  case X86::VSQRTPSZ128rkz:
  case X86::VSQRTPSZ256m:
  case X86::VSQRTPSZ256mb:
  case X86::VSQRTPSZ256mbk:
  case X86::VSQRTPSZ256mbkz:
  case X86::VSQRTPSZ256mk:
  case X86::VSQRTPSZ256mkz:
  case X86::VSQRTPSZ256r:
  case X86::VSQRTPSZ256rk:
  case X86::VSQRTPSZ256rkz:
  case X86::VSQRTPSZm:
  case X86::VSQRTPSZmb:
  case X86::VSQRTPSZmbk:
  case X86::VSQRTPSZmbkz:
  case X86::VSQRTPSZmk:
  case X86::VSQRTPSZmkz:
  case X86::VSQRTPSZr:
  case X86::VSQRTPSZrb:
  case X86::VSQRTPSZrbk:
  case X86::VSQRTPSZrbkz:
  case X86::VSQRTPSZrk:
  case X86::VSQRTPSZrkz:
  case X86::VSQRTSDZm:
  case X86::VSQRTSDZm_Int:
  case X86::VSQRTSDZm_Intk:
  case X86::VSQRTSDZm_Intkz:
  case X86::VSQRTSDZr:
  case X86::VSQRTSDZr_Int:
  case X86::VSQRTSDZr_Intk:
  case X86::VSQRTSDZr_Intkz:
  case X86::VSQRTSDZrb_Int:
  case X86::VSQRTSDZrb_Intk:
  case X86::VSQRTSDZrb_Intkz:
  case X86::VSQRTSSZm:
  case X86::VSQRTSSZm_Int:
  case X86::VSQRTSSZm_Intk:
  case X86::VSQRTSSZm_Intkz:
  case X86::VSQRTSSZr:
  case X86::VSQRTSSZr_Int:
  case X86::VSQRTSSZr_Intk:
  case X86::VSQRTSSZr_Intkz:
  case X86::VSQRTSSZrb_Int:
  case X86::VSQRTSSZrb_Intk:
  case X86::VSQRTSSZrb_Intkz:

  case X86::VGATHERDPDYrm:
  case X86::VGATHERDPDZ128rm:
  case X86::VGATHERDPDZ256rm:
  case X86::VGATHERDPDZrm:
  case X86::VGATHERDPDrm:
  case X86::VGATHERDPSYrm:
  case X86::VGATHERDPSZ128rm:
  case X86::VGATHERDPSZ256rm:
  case X86::VGATHERDPSZrm:
  case X86::VGATHERDPSrm:
  case X86::VGATHERPF0DPDm:
  case X86::VGATHERPF0DPSm:
  case X86::VGATHERPF0QPDm:
  case X86::VGATHERPF0QPSm:
  case X86::VGATHERPF1DPDm:
  case X86::VGATHERPF1DPSm:
  case X86::VGATHERPF1QPDm:
  case X86::VGATHERPF1QPSm:
  case X86::VGATHERQPDYrm:
  case X86::VGATHERQPDZ128rm:
  case X86::VGATHERQPDZ256rm:
  case X86::VGATHERQPDZrm:
  case X86::VGATHERQPDrm:
  case X86::VGATHERQPSYrm:
  case X86::VGATHERQPSZ128rm:
  case X86::VGATHERQPSZ256rm:
  case X86::VGATHERQPSZrm:
  case X86::VGATHERQPSrm:
  case X86::VPGATHERDDYrm:
  case X86::VPGATHERDDZ128rm:
  case X86::VPGATHERDDZ256rm:
  case X86::VPGATHERDDZrm:
  case X86::VPGATHERDDrm:
  case X86::VPGATHERDQYrm:
  case X86::VPGATHERDQZ128rm:
  case X86::VPGATHERDQZ256rm:
  case X86::VPGATHERDQZrm:
  case X86::VPGATHERDQrm:
  case X86::VPGATHERQDYrm:
  case X86::VPGATHERQDZ128rm:
  case X86::VPGATHERQDZ256rm:
  case X86::VPGATHERQDZrm:
  case X86::VPGATHERQDrm:
  case X86::VPGATHERQQYrm:
  case X86::VPGATHERQQZ128rm:
  case X86::VPGATHERQQZ256rm:
  case X86::VPGATHERQQZrm:
  case X86::VPGATHERQQrm:
  case X86::VSCATTERDPDZ128mr:
  case X86::VSCATTERDPDZ256mr:
  case X86::VSCATTERDPDZmr:
  case X86::VSCATTERDPSZ128mr:
  case X86::VSCATTERDPSZ256mr:
  case X86::VSCATTERDPSZmr:
  case X86::VSCATTERPF0DPDm:
  case X86::VSCATTERPF0DPSm:
  case X86::VSCATTERPF0QPDm:
  case X86::VSCATTERPF0QPSm:
  case X86::VSCATTERPF1DPDm:
  case X86::VSCATTERPF1DPSm:
  case X86::VSCATTERPF1QPDm:
  case X86::VSCATTERPF1QPSm:
  case X86::VSCATTERQPDZ128mr:
  case X86::VSCATTERQPDZ256mr:
  case X86::VSCATTERQPDZmr:
  case X86::VSCATTERQPSZ128mr:
  case X86::VSCATTERQPSZ256mr:
  case X86::VSCATTERQPSZmr:
  case X86::VPSCATTERDDZ128mr:
  case X86::VPSCATTERDDZ256mr:
  case X86::VPSCATTERDDZmr:
  case X86::VPSCATTERDQZ128mr:
  case X86::VPSCATTERDQZ256mr:
  case X86::VPSCATTERDQZmr:
  case X86::VPSCATTERQDZ128mr:
  case X86::VPSCATTERQDZ256mr:
  case X86::VPSCATTERQDZmr:
  case X86::VPSCATTERQQZ128mr:
  case X86::VPSCATTERQQZ256mr:
  case X86::VPSCATTERQQZmr:
    return true;
  }
}

bool X86InstrInfo::hasHighOperandLatency(const TargetSchedModel &SchedModel,
                                         const MachineRegisterInfo *MRI,
                                         const MachineInstr &DefMI,
                                         unsigned DefIdx,
                                         const MachineInstr &UseMI,
                                         unsigned UseIdx) const {
  return isHighLatencyDef(DefMI.getOpcode());
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
  case X86::ANDPDrr:
  case X86::ANDPSrr:
  case X86::ORPDrr:
  case X86::ORPSrr:
  case X86::XORPDrr:
  case X86::XORPSrr:
  case X86::PADDBrr:
  case X86::PADDWrr:
  case X86::PADDDrr:
  case X86::PADDQrr:
  case X86::VPANDrr:
  case X86::VPANDYrr:
  case X86::VPANDDZ128rr:
  case X86::VPANDDZ256rr:
  case X86::VPANDDZrr:
  case X86::VPANDQZ128rr:
  case X86::VPANDQZ256rr:
  case X86::VPANDQZrr:
  case X86::VPORrr:
  case X86::VPORYrr:
  case X86::VPORDZ128rr:
  case X86::VPORDZ256rr:
  case X86::VPORDZrr:
  case X86::VPORQZ128rr:
  case X86::VPORQZ256rr:
  case X86::VPORQZrr:
  case X86::VPXORrr:
  case X86::VPXORYrr:
  case X86::VPXORDZ128rr:
  case X86::VPXORDZ256rr:
  case X86::VPXORDZrr:
  case X86::VPXORQZ128rr:
  case X86::VPXORQZ256rr:
  case X86::VPXORQZrr:
  case X86::VANDPDrr:
  case X86::VANDPSrr:
  case X86::VANDPDYrr:
  case X86::VANDPSYrr:
  case X86::VANDPDZ128rr:
  case X86::VANDPSZ128rr:
  case X86::VANDPDZ256rr:
  case X86::VANDPSZ256rr:
  case X86::VANDPDZrr:
  case X86::VANDPSZrr:
  case X86::VORPDrr:
  case X86::VORPSrr:
  case X86::VORPDYrr:
  case X86::VORPSYrr:
  case X86::VORPDZ128rr:
  case X86::VORPSZ128rr:
  case X86::VORPDZ256rr:
  case X86::VORPSZ256rr:
  case X86::VORPDZrr:
  case X86::VORPSZrr:
  case X86::VXORPDrr:
  case X86::VXORPSrr:
  case X86::VXORPDYrr:
  case X86::VXORPSYrr:
  case X86::VXORPDZ128rr:
  case X86::VXORPSZ128rr:
  case X86::VXORPDZ256rr:
  case X86::VXORPSZ256rr:
  case X86::VXORPDZrr:
  case X86::VXORPSZrr:
  case X86::KADDBrr:
  case X86::KADDWrr:
  case X86::KADDDrr:
  case X86::KADDQrr:
  case X86::KANDBrr:
  case X86::KANDWrr:
  case X86::KANDDrr:
  case X86::KANDQrr:
  case X86::KORBrr:
  case X86::KORWrr:
  case X86::KORDrr:
  case X86::KORQrr:
  case X86::KXORBrr:
  case X86::KXORWrr:
  case X86::KXORDrr:
  case X86::KXORQrr:
  case X86::VPADDBrr:
  case X86::VPADDWrr:
  case X86::VPADDDrr:
  case X86::VPADDQrr:
  case X86::VPADDBYrr:
  case X86::VPADDWYrr:
  case X86::VPADDDYrr:
  case X86::VPADDQYrr:
  case X86::VPADDBZ128rr:
  case X86::VPADDWZ128rr:
  case X86::VPADDDZ128rr:
  case X86::VPADDQZ128rr:
  case X86::VPADDBZ256rr:
  case X86::VPADDWZ256rr:
  case X86::VPADDDZ256rr:
  case X86::VPADDQZ256rr:
  case X86::VPADDBZrr:
  case X86::VPADDWZrr:
  case X86::VPADDDZrr:
  case X86::VPADDQZrr:
  case X86::VPMULLWrr:
  case X86::VPMULLWYrr:
  case X86::VPMULLWZ128rr:
  case X86::VPMULLWZ256rr:
  case X86::VPMULLWZrr:
  case X86::VPMULLDrr:
  case X86::VPMULLDYrr:
  case X86::VPMULLDZ128rr:
  case X86::VPMULLDZ256rr:
  case X86::VPMULLDZrr:
  case X86::VPMULLQZ128rr:
  case X86::VPMULLQZ256rr:
  case X86::VPMULLQZrr:
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
  case X86::VMAXCPDZ128rr:
  case X86::VMAXCPSZ128rr:
  case X86::VMAXCPDZ256rr:
  case X86::VMAXCPSZ256rr:
  case X86::VMAXCPDZrr:
  case X86::VMAXCPSZrr:
  case X86::VMAXCSDrr:
  case X86::VMAXCSSrr:
  case X86::VMAXCSDZrr:
  case X86::VMAXCSSZrr:
  case X86::VMINCPDrr:
  case X86::VMINCPSrr:
  case X86::VMINCPDYrr:
  case X86::VMINCPSYrr:
  case X86::VMINCPDZ128rr:
  case X86::VMINCPSZ128rr:
  case X86::VMINCPDZ256rr:
  case X86::VMINCPSZ256rr:
  case X86::VMINCPDZrr:
  case X86::VMINCPSZrr:
  case X86::VMINCSDrr:
  case X86::VMINCSSrr:
  case X86::VMINCSDZrr:
  case X86::VMINCSSZrr:
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
  case X86::VADDPDZ128rr:
  case X86::VADDPSZ128rr:
  case X86::VADDPDZ256rr:
  case X86::VADDPSZ256rr:
  case X86::VADDPDZrr:
  case X86::VADDPSZrr:
  case X86::VADDSDrr:
  case X86::VADDSSrr:
  case X86::VADDSDZrr:
  case X86::VADDSSZrr:
  case X86::VMULPDrr:
  case X86::VMULPSrr:
  case X86::VMULPDYrr:
  case X86::VMULPSYrr:
  case X86::VMULPDZ128rr:
  case X86::VMULPSZ128rr:
  case X86::VMULPDZ256rr:
  case X86::VMULPSZ256rr:
  case X86::VMULPDZrr:
  case X86::VMULPSZrr:
  case X86::VMULSDrr:
  case X86::VMULSSrr:
  case X86::VMULSDZrr:
  case X86::VMULSSZrr:
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
      {MO_DARWIN_NONLAZY, "x86-darwin-nonlazy"},
      {MO_DARWIN_NONLAZY_PIC_BASE, "x86-darwin-nonlazy-pic-base"},
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
      if (!TM->isPositionIndependent())
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

    StringRef getPassName() const override {
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
      if (skipFunction(*MF.getFunction()))
        return false;

      X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
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
              I = ReplaceTLSBaseAddrCall(*I, TLSBaseAddrReg);
            else
              I = SetRegister(*I, &TLSBaseAddrReg);
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
    MachineInstr *ReplaceTLSBaseAddrCall(MachineInstr &I,
                                         unsigned TLSBaseAddrReg) {
      MachineFunction *MF = I.getParent()->getParent();
      const X86Subtarget &STI = MF->getSubtarget<X86Subtarget>();
      const bool is64Bit = STI.is64Bit();
      const X86InstrInfo *TII = STI.getInstrInfo();

      // Insert a Copy from TLSBaseAddrReg to RAX/EAX.
      MachineInstr *Copy =
          BuildMI(*I.getParent(), I, I.getDebugLoc(),
                  TII->get(TargetOpcode::COPY), is64Bit ? X86::RAX : X86::EAX)
              .addReg(TLSBaseAddrReg);

      // Erase the TLS_base_addr instruction.
      I.eraseFromParent();

      return Copy;
    }

    // Create a virtal register in *TLSBaseAddrReg, and populate it by
    // inserting a copy instruction after I. Returns the new instruction.
    MachineInstr *SetRegister(MachineInstr &I, unsigned *TLSBaseAddrReg) {
      MachineFunction *MF = I.getParent()->getParent();
      const X86Subtarget &STI = MF->getSubtarget<X86Subtarget>();
      const bool is64Bit = STI.is64Bit();
      const X86InstrInfo *TII = STI.getInstrInfo();

      // Create a virtual register for the TLS base address.
      MachineRegisterInfo &RegInfo = MF->getRegInfo();
      *TLSBaseAddrReg = RegInfo.createVirtualRegister(is64Bit
                                                      ? &X86::GR64RegClass
                                                      : &X86::GR32RegClass);

      // Insert a copy from RAX/EAX to TLSBaseAddrReg.
      MachineInstr *Next = I.getNextNode();
      MachineInstr *Copy =
          BuildMI(*I.getParent(), Next, I.getDebugLoc(),
                  TII->get(TargetOpcode::COPY), *TLSBaseAddrReg)
              .addReg(is64Bit ? X86::RAX : X86::EAX);

      return Copy;
    }

    StringRef getPassName() const override {
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

unsigned X86InstrInfo::getOutliningBenefit(size_t SequenceSize,
                                           size_t Occurrences,
                                           bool CanBeTailCall) const {
  unsigned NotOutlinedSize = SequenceSize * Occurrences;
  unsigned OutlinedSize;

  // Is it a tail call?
  if (CanBeTailCall) {
    // If yes, we don't have to include a return instruction-- it's already in
    // our sequence. So we have one occurrence of the sequence + #Occurrences
    // calls.
    OutlinedSize = SequenceSize + Occurrences;
  } else {
    // If not, add one for the return instruction.
    OutlinedSize = (SequenceSize + 1) + Occurrences;
  }

  // Return the number of instructions saved by outlining this sequence.
  return NotOutlinedSize > OutlinedSize ? NotOutlinedSize - OutlinedSize : 0;
}

bool X86InstrInfo::isFunctionSafeToOutlineFrom(MachineFunction &MF) const {
  return MF.getFunction()->hasFnAttribute(Attribute::NoRedZone);
}

X86GenInstrInfo::MachineOutlinerInstrType
X86InstrInfo::getOutliningType(MachineInstr &MI) const {

  // Don't allow debug values to impact outlining type.
  if (MI.isDebugValue() || MI.isIndirectDebugValue())
    return MachineOutlinerInstrType::Invisible;

  // Is this a tail call? If yes, we can outline as a tail call.
  if (isTailCall(MI))
    return MachineOutlinerInstrType::Legal;

  // Is this the terminator of a basic block?
  if (MI.isTerminator() || MI.isReturn()) {

    // Does its parent have any successors in its MachineFunction?
    if (MI.getParent()->succ_empty())
        return MachineOutlinerInstrType::Legal;

    // It does, so we can't tail call it.
    return MachineOutlinerInstrType::Illegal;
  }

  // Don't outline anything that modifies or reads from the stack pointer.
  //
  // FIXME: There are instructions which are being manually built without
  // explicit uses/defs so we also have to check the MCInstrDesc. We should be
  // able to remove the extra checks once those are fixed up. For example,
  // sometimes we might get something like %RAX<def> = POP64r 1. This won't be
  // caught by modifiesRegister or readsRegister even though the instruction
  // really ought to be formed so that modifiesRegister/readsRegister would
  // catch it.
  if (MI.modifiesRegister(X86::RSP, &RI) || MI.readsRegister(X86::RSP, &RI) ||
      MI.getDesc().hasImplicitUseOfPhysReg(X86::RSP) ||
      MI.getDesc().hasImplicitDefOfPhysReg(X86::RSP)) 
    return MachineOutlinerInstrType::Illegal;

  // Outlined calls change the instruction pointer, so don't read from it.
  if (MI.readsRegister(X86::RIP, &RI) ||
      MI.getDesc().hasImplicitUseOfPhysReg(X86::RIP) ||
      MI.getDesc().hasImplicitDefOfPhysReg(X86::RIP))
    return MachineOutlinerInstrType::Illegal;

  // Positions can't safely be outlined.
  if (MI.isPosition())
    return MachineOutlinerInstrType::Illegal;

  // Make sure none of the operands of this instruction do anything tricky.
  for (const MachineOperand &MOP : MI.operands())
    if (MOP.isCPI() || MOP.isJTI() || MOP.isCFIIndex() || MOP.isFI() ||
        MOP.isTargetIndex())
      return MachineOutlinerInstrType::Illegal;

  return MachineOutlinerInstrType::Legal;
}

void X86InstrInfo::insertOutlinerEpilogue(MachineBasicBlock &MBB,
                                          MachineFunction &MF,
                                          bool IsTailCall) const {

  // If we're a tail call, we already have a return, so don't do anything.
  if (IsTailCall)
    return;

  // We're a normal call, so our sequence doesn't have a return instruction.
  // Add it in.
  MachineInstr *retq = BuildMI(MF, DebugLoc(), get(X86::RETQ));
  MBB.insert(MBB.end(), retq);
}

void X86InstrInfo::insertOutlinerPrologue(MachineBasicBlock &MBB,
                                          MachineFunction &MF,
                                          bool IsTailCall) const {
  return;
}

MachineBasicBlock::iterator
X86InstrInfo::insertOutlinedCall(Module &M, MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator &It,
                                 MachineFunction &MF,
                                 bool IsTailCall) const {
  // Is it a tail call?
  if (IsTailCall) {
    // Yes, just insert a JMP.
    It = MBB.insert(It,
                  BuildMI(MF, DebugLoc(), get(X86::JMP_1))
                      .addGlobalAddress(M.getNamedValue(MF.getName())));
  } else {
    // No, insert a call.
    It = MBB.insert(It,
                  BuildMI(MF, DebugLoc(), get(X86::CALL64pcrel32))
                      .addGlobalAddress(M.getNamedValue(MF.getName())));
  }

  return It;
}
