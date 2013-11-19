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
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include <limits>

#define GET_INSTRINFO_CTOR_DTOR
#include "X86GenInstrInfo.inc"

using namespace llvm;

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

struct X86OpTblEntry {
  uint16_t RegOp;
  uint16_t MemOp;
  uint16_t Flags;
};

// Pin the vtable to this file.
void X86InstrInfo::anchor() {}

X86InstrInfo::X86InstrInfo(X86TargetMachine &tm)
  : X86GenInstrInfo((tm.getSubtarget<X86Subtarget>().is64Bit()
                     ? X86::ADJCALLSTACKDOWN64
                     : X86::ADJCALLSTACKDOWN32),
                    (tm.getSubtarget<X86Subtarget>().is64Bit()
                     ? X86::ADJCALLSTACKUP64
                     : X86::ADJCALLSTACKUP32)),
    TM(tm), RI(tm) {

  static const X86OpTblEntry OpTbl2Addr[] = {
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
    { X86::DEC64_16r,   X86::DEC64_16m,  0 },
    { X86::DEC64_32r,   X86::DEC64_32m,  0 },
    { X86::DEC64r,      X86::DEC64m,     0 },
    { X86::DEC8r,       X86::DEC8m,      0 },
    { X86::INC16r,      X86::INC16m,     0 },
    { X86::INC32r,      X86::INC32m,     0 },
    { X86::INC64_16r,   X86::INC64_16m,  0 },
    { X86::INC64_32r,   X86::INC64_32m,  0 },
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

  for (unsigned i = 0, e = array_lengthof(OpTbl2Addr); i != e; ++i) {
    unsigned RegOp = OpTbl2Addr[i].RegOp;
    unsigned MemOp = OpTbl2Addr[i].MemOp;
    unsigned Flags = OpTbl2Addr[i].Flags;
    AddTableEntry(RegOp2MemOpTable2Addr, MemOp2RegOpTable,
                  RegOp, MemOp,
                  // Index 0, folded load and store, no alignment requirement.
                  Flags | TB_INDEX_0 | TB_FOLDED_LOAD | TB_FOLDED_STORE);
  }

  static const X86OpTblEntry OpTbl0[] = {
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
    // AVX 256-bit foldable instructions
    { X86::VEXTRACTI128rr, X86::VEXTRACTI128mr, TB_FOLDED_STORE | TB_ALIGN_16 },
    { X86::VMOVAPDYrr,  X86::VMOVAPDYmr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVAPSYrr,  X86::VMOVAPSYmr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVDQAYrr,  X86::VMOVDQAYmr,    TB_FOLDED_STORE | TB_ALIGN_32 },
    { X86::VMOVUPDYrr,  X86::VMOVUPDYmr,    TB_FOLDED_STORE },
    { X86::VMOVUPSYrr,  X86::VMOVUPSYmr,    TB_FOLDED_STORE },
    // AVX-512 foldable instructions
    { X86::VMOVPDI2DIZrr,X86::VMOVPDI2DIZmr,  TB_FOLDED_STORE }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl0); i != e; ++i) {
    unsigned RegOp      = OpTbl0[i].RegOp;
    unsigned MemOp      = OpTbl0[i].MemOp;
    unsigned Flags      = OpTbl0[i].Flags;
    AddTableEntry(RegOp2MemOpTable0, MemOp2RegOpTable,
                  RegOp, MemOp, TB_INDEX_0 | Flags);
  }

  static const X86OpTblEntry OpTbl1[] = {
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
    { X86::MOVZQI2PQIrr,    X86::MOVZQI2PQIrm,        0 },
    { X86::MOVZPQILo2PQIrr, X86::MOVZPQILo2PQIrm,     TB_ALIGN_16 },
    { X86::MOVZX16rr8,      X86::MOVZX16rm8,          0 },
    { X86::MOVZX32rr16,     X86::MOVZX32rm16,         0 },
    { X86::MOVZX32_NOREXrr8, X86::MOVZX32_NOREXrm8,   0 },
    { X86::MOVZX32rr8,      X86::MOVZX32rm8,          0 },
    { X86::PABSBrr128,      X86::PABSBrm128,          TB_ALIGN_16 },
    { X86::PABSDrr128,      X86::PABSDrm128,          TB_ALIGN_16 },
    { X86::PABSWrr128,      X86::PABSWrm128,          TB_ALIGN_16 },
    { X86::PSHUFDri,        X86::PSHUFDmi,            TB_ALIGN_16 },
    { X86::PSHUFHWri,       X86::PSHUFHWmi,           TB_ALIGN_16 },
    { X86::PSHUFLWri,       X86::PSHUFLWmi,           TB_ALIGN_16 },
    { X86::RCPPSr,          X86::RCPPSm,              TB_ALIGN_16 },
    { X86::RCPPSr_Int,      X86::RCPPSm_Int,          TB_ALIGN_16 },
    { X86::RSQRTPSr,        X86::RSQRTPSm,            TB_ALIGN_16 },
    { X86::RSQRTPSr_Int,    X86::RSQRTPSm_Int,        TB_ALIGN_16 },
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
    { X86::VMOV64toPQIrr,   X86::VMOVQI2PQIrm,        0 },
    { X86::VMOV64toSDrr,    X86::VMOV64toSDrm,        0 },
    { X86::VMOVAPDrr,       X86::VMOVAPDrm,           TB_ALIGN_16 },
    { X86::VMOVAPSrr,       X86::VMOVAPSrm,           TB_ALIGN_16 },
    { X86::VMOVDDUPrr,      X86::VMOVDDUPrm,          0 },
    { X86::VMOVDI2PDIrr,    X86::VMOVDI2PDIrm,        0 },
    { X86::VMOVDI2SSrr,     X86::VMOVDI2SSrm,         0 },
    { X86::VMOVDQArr,       X86::VMOVDQArm,           TB_ALIGN_16 },
    { X86::VMOVSLDUPrr,     X86::VMOVSLDUPrm,         TB_ALIGN_16 },
    { X86::VMOVSHDUPrr,     X86::VMOVSHDUPrm,         TB_ALIGN_16 },
    { X86::VMOVUPDrr,       X86::VMOVUPDrm,           0 },
    { X86::VMOVUPSrr,       X86::VMOVUPSrm,           0 },
    { X86::VMOVZQI2PQIrr,   X86::VMOVZQI2PQIrm,       0 },
    { X86::VMOVZPQILo2PQIrr,X86::VMOVZPQILo2PQIrm,    TB_ALIGN_16 },
    { X86::VPABSBrr128,     X86::VPABSBrm128,         0 },
    { X86::VPABSDrr128,     X86::VPABSDrm128,         0 },
    { X86::VPABSWrr128,     X86::VPABSWrm128,         0 },
    { X86::VPERMILPDri,     X86::VPERMILPDmi,         0 },
    { X86::VPERMILPSri,     X86::VPERMILPSmi,         0 },
    { X86::VPSHUFDri,       X86::VPSHUFDmi,           0 },
    { X86::VPSHUFHWri,      X86::VPSHUFHWmi,          0 },
    { X86::VPSHUFLWri,      X86::VPSHUFLWmi,          0 },
    { X86::VRCPPSr,         X86::VRCPPSm,             0 },
    { X86::VRCPPSr_Int,     X86::VRCPPSm_Int,         0 },
    { X86::VRSQRTPSr,       X86::VRSQRTPSm,           0 },
    { X86::VRSQRTPSr_Int,   X86::VRSQRTPSm_Int,       0 },
    { X86::VSQRTPDr,        X86::VSQRTPDm,            0 },
    { X86::VSQRTPSr,        X86::VSQRTPSm,            0 },
    { X86::VUCOMISDrr,      X86::VUCOMISDrm,          0 },
    { X86::VUCOMISSrr,      X86::VUCOMISSrm,          0 },
    { X86::VBROADCASTSSrr,  X86::VBROADCASTSSrm,      TB_NO_REVERSE },

    // AVX 256-bit foldable instructions
    { X86::VMOVAPDYrr,      X86::VMOVAPDYrm,          TB_ALIGN_32 },
    { X86::VMOVAPSYrr,      X86::VMOVAPSYrm,          TB_ALIGN_32 },
    { X86::VMOVDQAYrr,      X86::VMOVDQAYrm,          TB_ALIGN_32 },
    { X86::VMOVUPDYrr,      X86::VMOVUPDYrm,          0 },
    { X86::VMOVUPSYrr,      X86::VMOVUPSYrm,          0 },
    { X86::VPERMILPDYri,    X86::VPERMILPDYmi,        0 },
    { X86::VPERMILPSYri,    X86::VPERMILPSYmi,        0 },

    // AVX2 foldable instructions
    { X86::VPABSBrr256,     X86::VPABSBrm256,         0 },
    { X86::VPABSDrr256,     X86::VPABSDrm256,         0 },
    { X86::VPABSWrr256,     X86::VPABSWrm256,         0 },
    { X86::VPSHUFDYri,      X86::VPSHUFDYmi,          0 },
    { X86::VPSHUFHWYri,     X86::VPSHUFHWYmi,         0 },
    { X86::VPSHUFLWYri,     X86::VPSHUFLWYmi,         0 },
    { X86::VRCPPSYr,        X86::VRCPPSYm,            0 },
    { X86::VRCPPSYr_Int,    X86::VRCPPSYm_Int,        0 },
    { X86::VRSQRTPSYr,      X86::VRSQRTPSYm,          0 },
    { X86::VSQRTPDYr,       X86::VSQRTPDYm,           0 },
    { X86::VSQRTPSYr,       X86::VSQRTPSYm,           0 },
    { X86::VBROADCASTSSYrr, X86::VBROADCASTSSYrm,     TB_NO_REVERSE },
    { X86::VBROADCASTSDYrr, X86::VBROADCASTSDYrm,     TB_NO_REVERSE },

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
    { X86::VMOVDQA32rr,     X86::VMOVDQA32rm,         TB_ALIGN_64 },
    { X86::VMOVDQA64rr,     X86::VMOVDQA64rm,         TB_ALIGN_64 },
    { X86::VMOVDQU32rr,     X86::VMOVDQU32rm,         0 },
    { X86::VMOVDQU64rr,     X86::VMOVDQU64rm,         0 },

    // AES foldable instructions
    { X86::AESIMCrr,              X86::AESIMCrm,              TB_ALIGN_16 },
    { X86::AESKEYGENASSIST128rr,  X86::AESKEYGENASSIST128rm,  TB_ALIGN_16 },
    { X86::VAESIMCrr,             X86::VAESIMCrm,             TB_ALIGN_16 },
    { X86::VAESKEYGENASSIST128rr, X86::VAESKEYGENASSIST128rm, TB_ALIGN_16 },
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl1); i != e; ++i) {
    unsigned RegOp = OpTbl1[i].RegOp;
    unsigned MemOp = OpTbl1[i].MemOp;
    unsigned Flags = OpTbl1[i].Flags;
    AddTableEntry(RegOp2MemOpTable1, MemOp2RegOpTable,
                  RegOp, MemOp,
                  // Index 1, folded load
                  Flags | TB_INDEX_1 | TB_FOLDED_LOAD);
  }

  static const X86OpTblEntry OpTbl2[] = {
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
    { X86::ADDSSrr,         X86::ADDSSrm,       0 },
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
    { X86::DIVPDrr,         X86::DIVPDrm,       TB_ALIGN_16 },
    { X86::DIVPSrr,         X86::DIVPSrm,       TB_ALIGN_16 },
    { X86::DIVSDrr,         X86::DIVSDrm,       0 },
    { X86::DIVSSrr,         X86::DIVSSrm,       0 },
    { X86::FsANDNPDrr,      X86::FsANDNPDrm,    TB_ALIGN_16 },
    { X86::FsANDNPSrr,      X86::FsANDNPSrm,    TB_ALIGN_16 },
    { X86::FsANDPDrr,       X86::FsANDPDrm,     TB_ALIGN_16 },
    { X86::FsANDPSrr,       X86::FsANDPSrm,     TB_ALIGN_16 },
    { X86::FsORPDrr,        X86::FsORPDrm,      TB_ALIGN_16 },
    { X86::FsORPSrr,        X86::FsORPSrm,      TB_ALIGN_16 },
    { X86::FsXORPDrr,       X86::FsXORPDrm,     TB_ALIGN_16 },
    { X86::FsXORPSrr,       X86::FsXORPSrm,     TB_ALIGN_16 },
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
    { X86::MAXSSrr,         X86::MAXSSrm,       0 },
    { X86::MINPDrr,         X86::MINPDrm,       TB_ALIGN_16 },
    { X86::MINPSrr,         X86::MINPSrm,       TB_ALIGN_16 },
    { X86::MINSDrr,         X86::MINSDrm,       0 },
    { X86::MINSSrr,         X86::MINSSrm,       0 },
    { X86::MPSADBWrri,      X86::MPSADBWrmi,    TB_ALIGN_16 },
    { X86::MULPDrr,         X86::MULPDrm,       TB_ALIGN_16 },
    { X86::MULPSrr,         X86::MULPSrm,       TB_ALIGN_16 },
    { X86::MULSDrr,         X86::MULSDrm,       0 },
    { X86::MULSSrr,         X86::MULSSrm,       0 },
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
    { X86::PBLENDWrri,      X86::PBLENDWrmi,    TB_ALIGN_16 },
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
    { X86::PINSRWrri,       X86::PINSRWrmi,     TB_ALIGN_16 },
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
    { X86::PSUBSBrr,        X86::PSUBSBrm,      TB_ALIGN_16 },
    { X86::PSUBSWrr,        X86::PSUBSWrm,      TB_ALIGN_16 },
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
    { X86::SUBSSrr,         X86::SUBSSrm,       0 },
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
    { X86::VCVTTPD2DQrr,      X86::VCVTTPD2DQXrm,      0 },
    { X86::VCVTTPS2DQrr,      X86::VCVTTPS2DQrm,       0 },
    { X86::VRSQRTSSr,         X86::VRSQRTSSm,          0 },
    { X86::VSQRTSDr,          X86::VSQRTSDm,           0 },
    { X86::VSQRTSSr,          X86::VSQRTSSm,           0 },
    { X86::VADDPDrr,          X86::VADDPDrm,           0 },
    { X86::VADDPSrr,          X86::VADDPSrm,           0 },
    { X86::VADDSDrr,          X86::VADDSDrm,           0 },
    { X86::VADDSSrr,          X86::VADDSSrm,           0 },
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
    { X86::VDIVSSrr,          X86::VDIVSSrm,           0 },
    { X86::VFsANDNPDrr,       X86::VFsANDNPDrm,        TB_ALIGN_16 },
    { X86::VFsANDNPSrr,       X86::VFsANDNPSrm,        TB_ALIGN_16 },
    { X86::VFsANDPDrr,        X86::VFsANDPDrm,         TB_ALIGN_16 },
    { X86::VFsANDPSrr,        X86::VFsANDPSrm,         TB_ALIGN_16 },
    { X86::VFsORPDrr,         X86::VFsORPDrm,          TB_ALIGN_16 },
    { X86::VFsORPSrr,         X86::VFsORPSrm,          TB_ALIGN_16 },
    { X86::VFsXORPDrr,        X86::VFsXORPDrm,         TB_ALIGN_16 },
    { X86::VFsXORPSrr,        X86::VFsXORPSrm,         TB_ALIGN_16 },
    { X86::VHADDPDrr,         X86::VHADDPDrm,          0 },
    { X86::VHADDPSrr,         X86::VHADDPSrm,          0 },
    { X86::VHSUBPDrr,         X86::VHSUBPDrm,          0 },
    { X86::VHSUBPSrr,         X86::VHSUBPSrm,          0 },
    { X86::Int_VCMPSDrr,      X86::Int_VCMPSDrm,       0 },
    { X86::Int_VCMPSSrr,      X86::Int_VCMPSSrm,       0 },
    { X86::VMAXPDrr,          X86::VMAXPDrm,           0 },
    { X86::VMAXPSrr,          X86::VMAXPSrm,           0 },
    { X86::VMAXSDrr,          X86::VMAXSDrm,           0 },
    { X86::VMAXSSrr,          X86::VMAXSSrm,           0 },
    { X86::VMINPDrr,          X86::VMINPDrm,           0 },
    { X86::VMINPSrr,          X86::VMINPSrm,           0 },
    { X86::VMINSDrr,          X86::VMINSDrm,           0 },
    { X86::VMINSSrr,          X86::VMINSSrm,           0 },
    { X86::VMPSADBWrri,       X86::VMPSADBWrmi,        0 },
    { X86::VMULPDrr,          X86::VMULPDrm,           0 },
    { X86::VMULPSrr,          X86::VMULPSrm,           0 },
    { X86::VMULSDrr,          X86::VMULSDrm,           0 },
    { X86::VMULSSrr,          X86::VMULSSrm,           0 },
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
    { X86::VPBLENDWrri,       X86::VPBLENDWrmi,        0 },
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
    { X86::VPSUBSBrr,         X86::VPSUBSBrm,          0 },
    { X86::VPSUBSWrr,         X86::VPSUBSWrm,          0 },
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
    { X86::VSHUFPDrri,        X86::VSHUFPDrmi,         0 },
    { X86::VSHUFPSrri,        X86::VSHUFPSrmi,         0 },
    { X86::VSUBPDrr,          X86::VSUBPDrm,           0 },
    { X86::VSUBPSrr,          X86::VSUBPSrm,           0 },
    { X86::VSUBSDrr,          X86::VSUBSDrm,           0 },
    { X86::VSUBSSrr,          X86::VSUBSSrm,           0 },
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
    { X86::VPERMPDYri,        X86::VPERMPDYmi,         0 },
    { X86::VPERMPSYrr,        X86::VPERMPSYrm,         0 },
    { X86::VPERMQYri,         X86::VPERMQYmi,          0 },
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
    { X86::VPSUBSBYrr,        X86::VPSUBSBYrm,         0 },
    { X86::VPSUBSWYrr,        X86::VPSUBSWYrm,         0 },
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
    // FIXME: add AVX 256-bit foldable instructions

    // FMA4 foldable patterns
    { X86::VFMADDSS4rr,       X86::VFMADDSS4mr,        0           },
    { X86::VFMADDSD4rr,       X86::VFMADDSD4mr,        0           },
    { X86::VFMADDPS4rr,       X86::VFMADDPS4mr,        TB_ALIGN_16 },
    { X86::VFMADDPD4rr,       X86::VFMADDPD4mr,        TB_ALIGN_16 },
    { X86::VFMADDPS4rrY,      X86::VFMADDPS4mrY,       TB_ALIGN_32 },
    { X86::VFMADDPD4rrY,      X86::VFMADDPD4mrY,       TB_ALIGN_32 },
    { X86::VFNMADDSS4rr,      X86::VFNMADDSS4mr,       0           },
    { X86::VFNMADDSD4rr,      X86::VFNMADDSD4mr,       0           },
    { X86::VFNMADDPS4rr,      X86::VFNMADDPS4mr,       TB_ALIGN_16 },
    { X86::VFNMADDPD4rr,      X86::VFNMADDPD4mr,       TB_ALIGN_16 },
    { X86::VFNMADDPS4rrY,     X86::VFNMADDPS4mrY,      TB_ALIGN_32 },
    { X86::VFNMADDPD4rrY,     X86::VFNMADDPD4mrY,      TB_ALIGN_32 },
    { X86::VFMSUBSS4rr,       X86::VFMSUBSS4mr,        0           },
    { X86::VFMSUBSD4rr,       X86::VFMSUBSD4mr,        0           },
    { X86::VFMSUBPS4rr,       X86::VFMSUBPS4mr,        TB_ALIGN_16 },
    { X86::VFMSUBPD4rr,       X86::VFMSUBPD4mr,        TB_ALIGN_16 },
    { X86::VFMSUBPS4rrY,      X86::VFMSUBPS4mrY,       TB_ALIGN_32 },
    { X86::VFMSUBPD4rrY,      X86::VFMSUBPD4mrY,       TB_ALIGN_32 },
    { X86::VFNMSUBSS4rr,      X86::VFNMSUBSS4mr,       0           },
    { X86::VFNMSUBSD4rr,      X86::VFNMSUBSD4mr,       0           },
    { X86::VFNMSUBPS4rr,      X86::VFNMSUBPS4mr,       TB_ALIGN_16 },
    { X86::VFNMSUBPD4rr,      X86::VFNMSUBPD4mr,       TB_ALIGN_16 },
    { X86::VFNMSUBPS4rrY,     X86::VFNMSUBPS4mrY,      TB_ALIGN_32 },
    { X86::VFNMSUBPD4rrY,     X86::VFNMSUBPD4mrY,      TB_ALIGN_32 },
    { X86::VFMADDSUBPS4rr,    X86::VFMADDSUBPS4mr,     TB_ALIGN_16 },
    { X86::VFMADDSUBPD4rr,    X86::VFMADDSUBPD4mr,     TB_ALIGN_16 },
    { X86::VFMADDSUBPS4rrY,   X86::VFMADDSUBPS4mrY,    TB_ALIGN_32 },
    { X86::VFMADDSUBPD4rrY,   X86::VFMADDSUBPD4mrY,    TB_ALIGN_32 },
    { X86::VFMSUBADDPS4rr,    X86::VFMSUBADDPS4mr,     TB_ALIGN_16 },
    { X86::VFMSUBADDPD4rr,    X86::VFMSUBADDPD4mr,     TB_ALIGN_16 },
    { X86::VFMSUBADDPS4rrY,   X86::VFMSUBADDPS4mrY,    TB_ALIGN_32 },
    { X86::VFMSUBADDPD4rrY,   X86::VFMSUBADDPD4mrY,    TB_ALIGN_32 },

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
    { X86::VPADDDZrr,         X86::VPADDDZrm,           0 },
    { X86::VPADDQZrr,         X86::VPADDQZrm,           0 },
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
    { X86::VPERMPDZri,        X86::VPERMPDZmi,          0 },
    { X86::VPERMPSZrr,        X86::VPERMPSZrm,          0 },
    { X86::VPSLLVDZrr,        X86::VPSLLVDZrm,          0 },
    { X86::VPSLLVQZrr,        X86::VPSLLVQZrm,          0 },
    { X86::VPSRAVDZrr,        X86::VPSRAVDZrm,          0 },
    { X86::VPSRLVDZrr,        X86::VPSRLVDZrm,          0 },
    { X86::VPSRLVQZrr,        X86::VPSRLVQZrm,          0 },
    { X86::VSHUFPDZrri,       X86::VSHUFPDZrmi,         0 },
    { X86::VSHUFPSZrri,       X86::VSHUFPSZrmi,         0 },
    { X86::VALIGNQrri,        X86::VALIGNQrmi,          0 },
    { X86::VALIGNDrri,        X86::VALIGNDrmi,          0 },

    // AES foldable instructions
    { X86::AESDECLASTrr,      X86::AESDECLASTrm,        TB_ALIGN_16 },
    { X86::AESDECrr,          X86::AESDECrm,            TB_ALIGN_16 },
    { X86::AESENCLASTrr,      X86::AESENCLASTrm,        TB_ALIGN_16 },
    { X86::AESENCrr,          X86::AESENCrm,            TB_ALIGN_16 },
    { X86::VAESDECLASTrr,     X86::VAESDECLASTrm,       TB_ALIGN_16 },
    { X86::VAESDECrr,         X86::VAESDECrm,           TB_ALIGN_16 },
    { X86::VAESENCLASTrr,     X86::VAESENCLASTrm,       TB_ALIGN_16 },
    { X86::VAESENCrr,         X86::VAESENCrm,           TB_ALIGN_16 },

    // SHA foldable instructions
    { X86::SHA1MSG1rr,        X86::SHA1MSG1rm,          TB_ALIGN_16 },
    { X86::SHA1MSG2rr,        X86::SHA1MSG2rm,          TB_ALIGN_16 },
    { X86::SHA1NEXTErr,       X86::SHA1NEXTErm,         TB_ALIGN_16 },
    { X86::SHA1RNDS4rri,      X86::SHA1RNDS4rmi,        TB_ALIGN_16 },
    { X86::SHA256MSG1rr,      X86::SHA256MSG1rm,        TB_ALIGN_16 },
    { X86::SHA256MSG2rr,      X86::SHA256MSG2rm,        TB_ALIGN_16 },
    { X86::SHA256RNDS2rr,     X86::SHA256RNDS2rm,       TB_ALIGN_16 },
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2); i != e; ++i) {
    unsigned RegOp = OpTbl2[i].RegOp;
    unsigned MemOp = OpTbl2[i].MemOp;
    unsigned Flags = OpTbl2[i].Flags;
    AddTableEntry(RegOp2MemOpTable2, MemOp2RegOpTable,
                  RegOp, MemOp,
                  // Index 2, folded load
                  Flags | TB_INDEX_2 | TB_FOLDED_LOAD);
  }

  static const X86OpTblEntry OpTbl3[] = {
    // FMA foldable instructions
    { X86::VFMADDSSr231r,         X86::VFMADDSSr231m,         0 },
    { X86::VFMADDSDr231r,         X86::VFMADDSDr231m,         0 },
    { X86::VFMADDSSr132r,         X86::VFMADDSSr132m,         0 },
    { X86::VFMADDSDr132r,         X86::VFMADDSDr132m,         0 },
    { X86::VFMADDSSr213r,         X86::VFMADDSSr213m,         0 },
    { X86::VFMADDSDr213r,         X86::VFMADDSDr213m,         0 },
    { X86::VFMADDSSr213r_Int,     X86::VFMADDSSr213m_Int,     0 },
    { X86::VFMADDSDr213r_Int,     X86::VFMADDSDr213m_Int,     0 },

    { X86::VFMADDPSr231r,         X86::VFMADDPSr231m,         TB_ALIGN_16 },
    { X86::VFMADDPDr231r,         X86::VFMADDPDr231m,         TB_ALIGN_16 },
    { X86::VFMADDPSr132r,         X86::VFMADDPSr132m,         TB_ALIGN_16 },
    { X86::VFMADDPDr132r,         X86::VFMADDPDr132m,         TB_ALIGN_16 },
    { X86::VFMADDPSr213r,         X86::VFMADDPSr213m,         TB_ALIGN_16 },
    { X86::VFMADDPDr213r,         X86::VFMADDPDr213m,         TB_ALIGN_16 },
    { X86::VFMADDPSr231rY,        X86::VFMADDPSr231mY,        TB_ALIGN_32 },
    { X86::VFMADDPDr231rY,        X86::VFMADDPDr231mY,        TB_ALIGN_32 },
    { X86::VFMADDPSr132rY,        X86::VFMADDPSr132mY,        TB_ALIGN_32 },
    { X86::VFMADDPDr132rY,        X86::VFMADDPDr132mY,        TB_ALIGN_32 },
    { X86::VFMADDPSr213rY,        X86::VFMADDPSr213mY,        TB_ALIGN_32 },
    { X86::VFMADDPDr213rY,        X86::VFMADDPDr213mY,        TB_ALIGN_32 },

    { X86::VFNMADDSSr231r,        X86::VFNMADDSSr231m,        0 },
    { X86::VFNMADDSDr231r,        X86::VFNMADDSDr231m,        0 },
    { X86::VFNMADDSSr132r,        X86::VFNMADDSSr132m,        0 },
    { X86::VFNMADDSDr132r,        X86::VFNMADDSDr132m,        0 },
    { X86::VFNMADDSSr213r,        X86::VFNMADDSSr213m,        0 },
    { X86::VFNMADDSDr213r,        X86::VFNMADDSDr213m,        0 },
    { X86::VFNMADDSSr213r_Int,    X86::VFNMADDSSr213m_Int,    0 },
    { X86::VFNMADDSDr213r_Int,    X86::VFNMADDSDr213m_Int,    0 },

    { X86::VFNMADDPSr231r,        X86::VFNMADDPSr231m,        TB_ALIGN_16 },
    { X86::VFNMADDPDr231r,        X86::VFNMADDPDr231m,        TB_ALIGN_16 },
    { X86::VFNMADDPSr132r,        X86::VFNMADDPSr132m,        TB_ALIGN_16 },
    { X86::VFNMADDPDr132r,        X86::VFNMADDPDr132m,        TB_ALIGN_16 },
    { X86::VFNMADDPSr213r,        X86::VFNMADDPSr213m,        TB_ALIGN_16 },
    { X86::VFNMADDPDr213r,        X86::VFNMADDPDr213m,        TB_ALIGN_16 },
    { X86::VFNMADDPSr231rY,       X86::VFNMADDPSr231mY,       TB_ALIGN_32 },
    { X86::VFNMADDPDr231rY,       X86::VFNMADDPDr231mY,       TB_ALIGN_32 },
    { X86::VFNMADDPSr132rY,       X86::VFNMADDPSr132mY,       TB_ALIGN_32 },
    { X86::VFNMADDPDr132rY,       X86::VFNMADDPDr132mY,       TB_ALIGN_32 },
    { X86::VFNMADDPSr213rY,       X86::VFNMADDPSr213mY,       TB_ALIGN_32 },
    { X86::VFNMADDPDr213rY,       X86::VFNMADDPDr213mY,       TB_ALIGN_32 },

    { X86::VFMSUBSSr231r,         X86::VFMSUBSSr231m,         0 },
    { X86::VFMSUBSDr231r,         X86::VFMSUBSDr231m,         0 },
    { X86::VFMSUBSSr132r,         X86::VFMSUBSSr132m,         0 },
    { X86::VFMSUBSDr132r,         X86::VFMSUBSDr132m,         0 },
    { X86::VFMSUBSSr213r,         X86::VFMSUBSSr213m,         0 },
    { X86::VFMSUBSDr213r,         X86::VFMSUBSDr213m,         0 },
    { X86::VFMSUBSSr213r_Int,     X86::VFMSUBSSr213m_Int,     0 },
    { X86::VFMSUBSDr213r_Int,     X86::VFMSUBSDr213m_Int,     0 },

    { X86::VFMSUBPSr231r,         X86::VFMSUBPSr231m,         TB_ALIGN_16 },
    { X86::VFMSUBPDr231r,         X86::VFMSUBPDr231m,         TB_ALIGN_16 },
    { X86::VFMSUBPSr132r,         X86::VFMSUBPSr132m,         TB_ALIGN_16 },
    { X86::VFMSUBPDr132r,         X86::VFMSUBPDr132m,         TB_ALIGN_16 },
    { X86::VFMSUBPSr213r,         X86::VFMSUBPSr213m,         TB_ALIGN_16 },
    { X86::VFMSUBPDr213r,         X86::VFMSUBPDr213m,         TB_ALIGN_16 },
    { X86::VFMSUBPSr231rY,        X86::VFMSUBPSr231mY,        TB_ALIGN_32 },
    { X86::VFMSUBPDr231rY,        X86::VFMSUBPDr231mY,        TB_ALIGN_32 },
    { X86::VFMSUBPSr132rY,        X86::VFMSUBPSr132mY,        TB_ALIGN_32 },
    { X86::VFMSUBPDr132rY,        X86::VFMSUBPDr132mY,        TB_ALIGN_32 },
    { X86::VFMSUBPSr213rY,        X86::VFMSUBPSr213mY,        TB_ALIGN_32 },
    { X86::VFMSUBPDr213rY,        X86::VFMSUBPDr213mY,        TB_ALIGN_32 },

    { X86::VFNMSUBSSr231r,        X86::VFNMSUBSSr231m,        0 },
    { X86::VFNMSUBSDr231r,        X86::VFNMSUBSDr231m,        0 },
    { X86::VFNMSUBSSr132r,        X86::VFNMSUBSSr132m,        0 },
    { X86::VFNMSUBSDr132r,        X86::VFNMSUBSDr132m,        0 },
    { X86::VFNMSUBSSr213r,        X86::VFNMSUBSSr213m,        0 },
    { X86::VFNMSUBSDr213r,        X86::VFNMSUBSDr213m,        0 },
    { X86::VFNMSUBSSr213r_Int,    X86::VFNMSUBSSr213m_Int,    0 },
    { X86::VFNMSUBSDr213r_Int,    X86::VFNMSUBSDr213m_Int,    0 },

    { X86::VFNMSUBPSr231r,        X86::VFNMSUBPSr231m,        TB_ALIGN_16 },
    { X86::VFNMSUBPDr231r,        X86::VFNMSUBPDr231m,        TB_ALIGN_16 },
    { X86::VFNMSUBPSr132r,        X86::VFNMSUBPSr132m,        TB_ALIGN_16 },
    { X86::VFNMSUBPDr132r,        X86::VFNMSUBPDr132m,        TB_ALIGN_16 },
    { X86::VFNMSUBPSr213r,        X86::VFNMSUBPSr213m,        TB_ALIGN_16 },
    { X86::VFNMSUBPDr213r,        X86::VFNMSUBPDr213m,        TB_ALIGN_16 },
    { X86::VFNMSUBPSr231rY,       X86::VFNMSUBPSr231mY,       TB_ALIGN_32 },
    { X86::VFNMSUBPDr231rY,       X86::VFNMSUBPDr231mY,       TB_ALIGN_32 },
    { X86::VFNMSUBPSr132rY,       X86::VFNMSUBPSr132mY,       TB_ALIGN_32 },
    { X86::VFNMSUBPDr132rY,       X86::VFNMSUBPDr132mY,       TB_ALIGN_32 },
    { X86::VFNMSUBPSr213rY,       X86::VFNMSUBPSr213mY,       TB_ALIGN_32 },
    { X86::VFNMSUBPDr213rY,       X86::VFNMSUBPDr213mY,       TB_ALIGN_32 },

    { X86::VFMADDSUBPSr231r,      X86::VFMADDSUBPSr231m,      TB_ALIGN_16 },
    { X86::VFMADDSUBPDr231r,      X86::VFMADDSUBPDr231m,      TB_ALIGN_16 },
    { X86::VFMADDSUBPSr132r,      X86::VFMADDSUBPSr132m,      TB_ALIGN_16 },
    { X86::VFMADDSUBPDr132r,      X86::VFMADDSUBPDr132m,      TB_ALIGN_16 },
    { X86::VFMADDSUBPSr213r,      X86::VFMADDSUBPSr213m,      TB_ALIGN_16 },
    { X86::VFMADDSUBPDr213r,      X86::VFMADDSUBPDr213m,      TB_ALIGN_16 },
    { X86::VFMADDSUBPSr231rY,     X86::VFMADDSUBPSr231mY,     TB_ALIGN_32 },
    { X86::VFMADDSUBPDr231rY,     X86::VFMADDSUBPDr231mY,     TB_ALIGN_32 },
    { X86::VFMADDSUBPSr132rY,     X86::VFMADDSUBPSr132mY,     TB_ALIGN_32 },
    { X86::VFMADDSUBPDr132rY,     X86::VFMADDSUBPDr132mY,     TB_ALIGN_32 },
    { X86::VFMADDSUBPSr213rY,     X86::VFMADDSUBPSr213mY,     TB_ALIGN_32 },
    { X86::VFMADDSUBPDr213rY,     X86::VFMADDSUBPDr213mY,     TB_ALIGN_32 },

    { X86::VFMSUBADDPSr231r,      X86::VFMSUBADDPSr231m,      TB_ALIGN_16 },
    { X86::VFMSUBADDPDr231r,      X86::VFMSUBADDPDr231m,      TB_ALIGN_16 },
    { X86::VFMSUBADDPSr132r,      X86::VFMSUBADDPSr132m,      TB_ALIGN_16 },
    { X86::VFMSUBADDPDr132r,      X86::VFMSUBADDPDr132m,      TB_ALIGN_16 },
    { X86::VFMSUBADDPSr213r,      X86::VFMSUBADDPSr213m,      TB_ALIGN_16 },
    { X86::VFMSUBADDPDr213r,      X86::VFMSUBADDPDr213m,      TB_ALIGN_16 },
    { X86::VFMSUBADDPSr231rY,     X86::VFMSUBADDPSr231mY,     TB_ALIGN_32 },
    { X86::VFMSUBADDPDr231rY,     X86::VFMSUBADDPDr231mY,     TB_ALIGN_32 },
    { X86::VFMSUBADDPSr132rY,     X86::VFMSUBADDPSr132mY,     TB_ALIGN_32 },
    { X86::VFMSUBADDPDr132rY,     X86::VFMSUBADDPDr132mY,     TB_ALIGN_32 },
    { X86::VFMSUBADDPSr213rY,     X86::VFMSUBADDPSr213mY,     TB_ALIGN_32 },
    { X86::VFMSUBADDPDr213rY,     X86::VFMSUBADDPDr213mY,     TB_ALIGN_32 },

    // FMA4 foldable patterns
    { X86::VFMADDSS4rr,           X86::VFMADDSS4rm,           0           },
    { X86::VFMADDSD4rr,           X86::VFMADDSD4rm,           0           },
    { X86::VFMADDPS4rr,           X86::VFMADDPS4rm,           TB_ALIGN_16 },
    { X86::VFMADDPD4rr,           X86::VFMADDPD4rm,           TB_ALIGN_16 },
    { X86::VFMADDPS4rrY,          X86::VFMADDPS4rmY,          TB_ALIGN_32 },
    { X86::VFMADDPD4rrY,          X86::VFMADDPD4rmY,          TB_ALIGN_32 },
    { X86::VFNMADDSS4rr,          X86::VFNMADDSS4rm,          0           },
    { X86::VFNMADDSD4rr,          X86::VFNMADDSD4rm,          0           },
    { X86::VFNMADDPS4rr,          X86::VFNMADDPS4rm,          TB_ALIGN_16 },
    { X86::VFNMADDPD4rr,          X86::VFNMADDPD4rm,          TB_ALIGN_16 },
    { X86::VFNMADDPS4rrY,         X86::VFNMADDPS4rmY,         TB_ALIGN_32 },
    { X86::VFNMADDPD4rrY,         X86::VFNMADDPD4rmY,         TB_ALIGN_32 },
    { X86::VFMSUBSS4rr,           X86::VFMSUBSS4rm,           0           },
    { X86::VFMSUBSD4rr,           X86::VFMSUBSD4rm,           0           },
    { X86::VFMSUBPS4rr,           X86::VFMSUBPS4rm,           TB_ALIGN_16 },
    { X86::VFMSUBPD4rr,           X86::VFMSUBPD4rm,           TB_ALIGN_16 },
    { X86::VFMSUBPS4rrY,          X86::VFMSUBPS4rmY,          TB_ALIGN_32 },
    { X86::VFMSUBPD4rrY,          X86::VFMSUBPD4rmY,          TB_ALIGN_32 },
    { X86::VFNMSUBSS4rr,          X86::VFNMSUBSS4rm,          0           },
    { X86::VFNMSUBSD4rr,          X86::VFNMSUBSD4rm,          0           },
    { X86::VFNMSUBPS4rr,          X86::VFNMSUBPS4rm,          TB_ALIGN_16 },
    { X86::VFNMSUBPD4rr,          X86::VFNMSUBPD4rm,          TB_ALIGN_16 },
    { X86::VFNMSUBPS4rrY,         X86::VFNMSUBPS4rmY,         TB_ALIGN_32 },
    { X86::VFNMSUBPD4rrY,         X86::VFNMSUBPD4rmY,         TB_ALIGN_32 },
    { X86::VFMADDSUBPS4rr,        X86::VFMADDSUBPS4rm,        TB_ALIGN_16 },
    { X86::VFMADDSUBPD4rr,        X86::VFMADDSUBPD4rm,        TB_ALIGN_16 },
    { X86::VFMADDSUBPS4rrY,       X86::VFMADDSUBPS4rmY,       TB_ALIGN_32 },
    { X86::VFMADDSUBPD4rrY,       X86::VFMADDSUBPD4rmY,       TB_ALIGN_32 },
    { X86::VFMSUBADDPS4rr,        X86::VFMSUBADDPS4rm,        TB_ALIGN_16 },
    { X86::VFMSUBADDPD4rr,        X86::VFMSUBADDPD4rm,        TB_ALIGN_16 },
    { X86::VFMSUBADDPS4rrY,       X86::VFMSUBADDPS4rmY,       TB_ALIGN_32 },
    { X86::VFMSUBADDPD4rrY,       X86::VFMSUBADDPD4rmY,       TB_ALIGN_32 },
    // AVX-512 VPERMI instructions with 3 source operands.
    { X86::VPERMI2Drr,            X86::VPERMI2Drm,            0 },
    { X86::VPERMI2Qrr,            X86::VPERMI2Qrm,            0 },
    { X86::VPERMI2PSrr,           X86::VPERMI2PSrm,           0 },
    { X86::VPERMI2PDrr,           X86::VPERMI2PDrm,           0 },
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl3); i != e; ++i) {
    unsigned RegOp = OpTbl3[i].RegOp;
    unsigned MemOp = OpTbl3[i].MemOp;
    unsigned Flags = OpTbl3[i].Flags;
    AddTableEntry(RegOp2MemOpTable3, MemOp2RegOpTable,
                  RegOp, MemOp,
                  // Index 3, folded load
                  Flags | TB_INDEX_3 | TB_FOLDED_LOAD);
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
    if (!TM.getSubtarget<X86Subtarget>().is64Bit())
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

/// isFrameOperand - Return true and the FrameIndex if the specified
/// operand and follow operands form a reference to the stack frame.
bool X86InstrInfo::isFrameOperand(const MachineInstr *MI, unsigned int Op,
                                  int &FrameIndex) const {
  if (MI->getOperand(Op).isFI() && MI->getOperand(Op+1).isImm() &&
      MI->getOperand(Op+2).isReg() && MI->getOperand(Op+3).isImm() &&
      MI->getOperand(Op+1).getImm() == 1 &&
      MI->getOperand(Op+2).getReg() == 0 &&
      MI->getOperand(Op+3).getImm() == 0) {
    FrameIndex = MI->getOperand(Op).getIndex();
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
  case X86::VMOVAPSYrm:
  case X86::VMOVAPDYrm:
  case X86::VMOVDQAYrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::VMOVDQA32rm:
  case X86::VMOVDQA64rm:
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
  case X86::VMOVAPSYmr:
  case X86::VMOVAPDYmr:
  case X86::VMOVDQAYmr:
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

/// regIsPICBase - Return true if register is PIC base (i.e.g defined by
/// X86::MOVPC32r.
static bool regIsPICBase(unsigned BaseReg, const MachineRegisterInfo &MRI) {
  // Don't waste compile time scanning use-def chains of physregs.
  if (!TargetRegisterInfo::isVirtualRegister(BaseReg))
    return false;
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
  case X86::FsMOVAPDrm: {
    // Loads from constant pools are trivially rematerializable.
    if (MI->getOperand(1).isReg() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(3).isReg() && MI->getOperand(3).getReg() == 0 &&
        MI->isInvariantLoad(AA)) {
      unsigned BaseReg = MI->getOperand(1).getReg();
      if (BaseReg == 0 || BaseReg == X86::RIP)
        return true;
      // Allow re-materialization of PIC load.
      if (!ReMatPICStubLoad && MI->getOperand(4).isGlobal())
        return false;
      const MachineFunction &MF = *MI->getParent()->getParent();
      const MachineRegisterInfo &MRI = MF.getRegInfo();
      return regIsPICBase(BaseReg, MRI);
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
/// a few instructions in each direction it assumes it's not safe.
static bool isSafeToClobberEFLAGS(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) {
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

  MachineInstr *NewMI = prior(I);
  NewMI->substituteRegister(Orig->getOperand(0).getReg(), DestReg, SubIdx, TRI);
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

/// getTruncatedShiftCount - check whether the shift count for a machine operand
/// is non-zero.
inline static unsigned getTruncatedShiftCount(MachineInstr *MI,
                                              unsigned ShiftAmtOperandIdx) {
  // The shift count is six bits with the REX.W prefix and five bits without.
  unsigned ShiftCountMask = (MI->getDesc().TSFlags & X86II::REX_W) ? 63 : 31;
  unsigned Imm = MI->getOperand(ShiftAmtOperandIdx).getImm();
  return Imm & ShiftCountMask;
}

/// isTruncatedShiftCountForLEA - check whether the given shift count is appropriate
/// can be represented by a LEA instruction.
inline static bool isTruncatedShiftCountForLEA(unsigned ShAmt) {
  // Left shift instructions can be transformed into load-effective-address
  // instructions if we can encode them appropriately.
  // A LEA instruction utilizes a SIB byte to encode it's scale factor.
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

/// convertToThreeAddressWithLEA - Helper for convertToThreeAddress when
/// 16-bit LEA is disabled, use 32-bit LEA to form 3-address code by promoting
/// to a 32-bit superregister and then truncating back down to a 16-bit
/// subregister.
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
  if (TM.getSubtarget<X86Subtarget>().is64Bit()) {
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
  case X86::INC64_16r:
    addRegOffset(MIB, leaInReg, true, 1);
    break;
  case X86::DEC16r:
  case X86::DEC64_16r:
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
    MachineInstr *InsMI2 = 0;
    if (Src == Src2) {
      // ADD16rr %reg1028<kill>, %reg1028
      // just a single insert_subreg.
      addRegReg(MIB, leaInReg, true, leaInReg, false);
    } else {
      if (TM.getSubtarget<X86Subtarget>().is64Bit())
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

  // The following opcodes also sets the condition code register(s). Only
  // convert them to equivalent lea if the condition code register def's
  // are dead!
  if (hasLiveCondCodeDef(MI))
    return 0;

  MachineFunction &MF = *MI->getParent()->getParent();
  // All instructions input are two-addr instructions.  Get the known operands.
  const MachineOperand &Dest = MI->getOperand(0);
  const MachineOperand &Src = MI->getOperand(1);

  MachineInstr *NewMI = NULL;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  // 16-bit LEA is also slow on Core2.
  bool DisableLEA16 = true;
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();

  unsigned MIOpc = MI->getOpcode();
  switch (MIOpc) {
  case X86::SHUFPSrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufps instruction!");
    if (!TM.getSubtarget<X86Subtarget>().hasSSE2()) return 0;

    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    if (B != C) return 0;
    unsigned M = MI->getOperand(3).getImm();
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::PSHUFDri))
      .addOperand(Dest).addOperand(Src).addImm(M);
    break;
  }
  case X86::SHUFPDrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufpd instruction!");
    if (!TM.getSubtarget<X86Subtarget>().hasSSE2()) return 0;

    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    if (B != C) return 0;
    unsigned M = MI->getOperand(3).getImm();

    // Convert to PSHUFD mask.
    M = ((M & 1) << 1) | ((M & 1) << 3) | ((M & 2) << 4) | ((M & 2) << 6)| 0x44;

    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::PSHUFDri))
      .addOperand(Dest).addOperand(Src).addImm(M);
    break;
  }
  case X86::SHL64ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return 0;

    // LEA can't handle RSP.
    if (TargetRegisterInfo::isVirtualRegister(Src.getReg()) &&
        !MF.getRegInfo().constrainRegClass(Src.getReg(),
                                           &X86::GR64_NOSPRegClass))
      return 0;

    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
      .addOperand(Dest)
      .addReg(0).addImm(1 << ShAmt).addOperand(Src).addImm(0).addReg(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    unsigned ShAmt = getTruncatedShiftCount(MI, 2);
    if (!isTruncatedShiftCountForLEA(ShAmt)) return 0;

    unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;

    // LEA can't handle ESP.
    bool isKill, isUndef;
    unsigned SrcReg;
    MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
    if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                        SrcReg, isKill, isUndef, ImplicitOp))
      return 0;

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
    if (!isTruncatedShiftCountForLEA(ShAmt)) return 0;

    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
      .addOperand(Dest)
      .addReg(0).addImm(1 << ShAmt).addOperand(Src).addImm(0).addReg(0);
    break;
  }
  default: {

    switch (MIOpc) {
    default: return 0;
    case X86::INC64r:
    case X86::INC32r:
    case X86::INC64_32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      unsigned Opc = MIOpc == X86::INC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      bool isKill, isUndef;
      unsigned SrcReg;
      MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
      if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                          SrcReg, isKill, isUndef, ImplicitOp))
        return 0;

      MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
          .addOperand(Dest)
          .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef));
      if (ImplicitOp.getReg() != 0)
        MIB.addOperand(ImplicitOp);

      NewMI = addOffset(MIB, 1);
      break;
    }
    case X86::INC16r:
    case X86::INC64_16r:
      if (DisableLEA16)
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      NewMI = addOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                        .addOperand(Dest).addOperand(Src), 1);
      break;
    case X86::DEC64r:
    case X86::DEC32r:
    case X86::DEC64_32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      unsigned Opc = MIOpc == X86::DEC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);

      bool isKill, isUndef;
      unsigned SrcReg;
      MachineOperand ImplicitOp = MachineOperand::CreateReg(0, false);
      if (!classifyLEAReg(MI, Src, Opc, /*AllowSP=*/ false,
                          SrcReg, isKill, isUndef, ImplicitOp))
        return 0;

      MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), get(Opc))
          .addOperand(Dest)
          .addReg(SrcReg, getUndefRegState(isUndef) | getKillRegState(isKill));
      if (ImplicitOp.getReg() != 0)
        MIB.addOperand(ImplicitOp);

      NewMI = addOffset(MIB, -1);

      break;
    }
    case X86::DEC16r:
    case X86::DEC64_16r:
      if (DisableLEA16)
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
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
        return 0;

      const MachineOperand &Src2 = MI->getOperand(2);
      bool isKill2, isUndef2;
      unsigned SrcReg2;
      MachineOperand ImplicitOp2 = MachineOperand::CreateReg(0, false);
      if (!classifyLEAReg(MI, Src2, Opc, /*AllowSP=*/ false,
                          SrcReg2, isKill2, isUndef2, ImplicitOp2))
        return 0;

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
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
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
        return 0;

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
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      NewMI = addOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                        .addOperand(Dest).addOperand(Src),
                        MI->getOperand(2).getImm());
      break;
    }
  }
  }

  if (!NewMI) return 0;

  if (LV) {  // Update live variables
    if (Src.isKill())
      LV->replaceKillInstruction(Src.getReg(), MI, NewMI);
    if (Dest.isDead())
      LV->replaceKillInstruction(Dest.getReg(), MI, NewMI);
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
    return TargetInstrInfo::commuteInstruction(MI, NewMI);
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
    return TargetInstrInfo::commuteInstruction(MI, NewMI);
  }
}

static X86::CondCode getCondFromBranchOpc(unsigned BrOpc) {
  switch (BrOpc) {
  default: return X86::COND_INVALID;
  case X86::JE_4:  return X86::COND_E;
  case X86::JNE_4: return X86::COND_NE;
  case X86::JL_4:  return X86::COND_L;
  case X86::JLE_4: return X86::COND_LE;
  case X86::JG_4:  return X86::COND_G;
  case X86::JGE_4: return X86::COND_GE;
  case X86::JB_4:  return X86::COND_B;
  case X86::JBE_4: return X86::COND_BE;
  case X86::JA_4:  return X86::COND_A;
  case X86::JAE_4: return X86::COND_AE;
  case X86::JS_4:  return X86::COND_S;
  case X86::JNS_4: return X86::COND_NS;
  case X86::JP_4:  return X86::COND_P;
  case X86::JNP_4: return X86::COND_NP;
  case X86::JO_4:  return X86::COND_O;
  case X86::JNO_4: return X86::COND_NO;
  }
}

/// getCondFromSETOpc - return condition code of a SET opcode.
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

/// getCondFromCmovOpc - return condition code of a CMov opcode.
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
  case X86::COND_E:  return X86::JE_4;
  case X86::COND_NE: return X86::JNE_4;
  case X86::COND_L:  return X86::JL_4;
  case X86::COND_LE: return X86::JLE_4;
  case X86::COND_G:  return X86::JG_4;
  case X86::COND_GE: return X86::JGE_4;
  case X86::COND_B:  return X86::JB_4;
  case X86::COND_BE: return X86::JBE_4;
  case X86::COND_A:  return X86::JA_4;
  case X86::COND_AE: return X86::JAE_4;
  case X86::COND_S:  return X86::JS_4;
  case X86::COND_NS: return X86::JNS_4;
  case X86::COND_P:  return X86::JP_4;
  case X86::COND_NP: return X86::JNP_4;
  case X86::COND_O:  return X86::JO_4;
  case X86::COND_NO: return X86::JNO_4;
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

/// getSwappedCondition - assume the flags are set by MI(a,b), return
/// the condition code if we modify the instructions such that flags are
/// set by MI(b,a).
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

/// getSETFromCond - Return a set opcode for the given condition and
/// whether it has memory operand.
static unsigned getSETFromCond(X86::CondCode CC,
                               bool HasMemoryOperand) {
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

  assert(CC < 16 && "Can only handle standard cond codes");
  return Opc[CC][HasMemoryOperand ? 1 : 0];
}

/// getCMovFromCond - Return a cmov opcode for the given condition,
/// register size in bytes, and operand type.
static unsigned getCMovFromCond(X86::CondCode CC, unsigned RegBytes,
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

bool X86InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
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
    if (I->getOpcode() == X86::JMP_4) {
      UnCondBrIter = I;

      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      while (llvm::next(I) != MBB.end())
        llvm::next(I)->eraseFromParent();

      Cond.clear();
      FBB = 0;

      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = 0;
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
        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(X86::JMP_4))
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
  }

  return false;
}

unsigned X86InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (I->getOpcode() != X86::JMP_4 &&
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
                           MachineBasicBlock *FBB,
                           const SmallVectorImpl<MachineOperand> &Cond,
                           DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(X86::JMP_4)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  X86::CondCode CC = (X86::CondCode)Cond[0].getImm();
  switch (CC) {
  case X86::COND_NP_OR_E:
    // Synthesize NP_OR_E with two branches.
    BuildMI(&MBB, DL, get(X86::JNP_4)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JE_4)).addMBB(TBB);
    ++Count;
    break;
  case X86::COND_NE_OR_P:
    // Synthesize NE_OR_P with two branches.
    BuildMI(&MBB, DL, get(X86::JNE_4)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JP_4)).addMBB(TBB);
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
    BuildMI(&MBB, DL, get(X86::JMP_4)).addMBB(FBB);
    ++Count;
  }
  return Count;
}

bool X86InstrInfo::
canInsertSelect(const MachineBasicBlock &MBB,
                const SmallVectorImpl<MachineOperand> &Cond,
                unsigned TrueReg, unsigned FalseReg,
                int &CondCycles, int &TrueCycles, int &FalseCycles) const {
  // Not all subtargets have cmov instructions.
  if (!TM.getSubtarget<X86Subtarget>().hasCMov())
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
                                unsigned DstReg,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                unsigned TrueReg, unsigned FalseReg) const {
   MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
   assert(Cond.size() == 1 && "Invalid Cond array");
   unsigned Opc = getCMovFromCond((X86::CondCode)Cond[0].getImm(),
                                  MRI.getRegClass(DstReg)->getSize(),
                                  false/*HasMemoryOperand*/);
   BuildMI(MBB, I, DL, get(Opc), DstReg).addReg(FalseReg).addReg(TrueReg);
}

/// isHReg - Test if the given register is a physical h register.
static bool isHReg(unsigned Reg) {
  return X86::GR8_ABCD_HRegClass.contains(Reg);
}

// Try and copy between VR128/VR64 and GR64 registers.
static unsigned CopyToFromAsymmetricReg(unsigned DestReg, unsigned SrcReg,
                                        const X86Subtarget& Subtarget) {


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
      return X86::MOVSDto64rr;
  } else if (X86::GR64RegClass.contains(SrcReg)) {
    // Copy from a GR64 register to a VR128 register.
    if (X86::VR128XRegClass.contains(DestReg))
      return HasAVX512 ? X86::VMOV64toPQIZrr: (HasAVX ? X86::VMOV64toPQIrr :
                                               X86::MOV64toPQIrr);
    // Copy from a GR64 register to a VR64 register.
    if (X86::VR64RegClass.contains(DestReg))
      return X86::MOV64toSDrr;
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

static
unsigned copyPhysRegOpcode_AVX512(unsigned& DestReg, unsigned& SrcReg) {
  if (X86::VR128XRegClass.contains(DestReg, SrcReg) ||
      X86::VR256XRegClass.contains(DestReg, SrcReg) ||
      X86::VR512RegClass.contains(DestReg, SrcReg)) {
     DestReg = get512BitSuperRegister(DestReg);
     SrcReg = get512BitSuperRegister(SrcReg);
     return X86::VMOVAPSZrr;
  }
  if ((X86::VK8RegClass.contains(DestReg) ||
       X86::VK16RegClass.contains(DestReg)) &&
      (X86::VK8RegClass.contains(SrcReg) ||
       X86::VK16RegClass.contains(SrcReg)))
    return X86::KMOVWkk;
  return 0;
}

void X86InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI, DebugLoc DL,
                               unsigned DestReg, unsigned SrcReg,
                               bool KillSrc) const {
  // First deal with the normal symmetric copies.
  bool HasAVX = TM.getSubtarget<X86Subtarget>().hasAVX();
  bool HasAVX512 = TM.getSubtarget<X86Subtarget>().hasAVX512();
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
        TM.getSubtarget<X86Subtarget>().is64Bit()) {
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
    Opc = copyPhysRegOpcode_AVX512(DestReg, SrcReg);
  else if (X86::VR128RegClass.contains(DestReg, SrcReg))
    Opc = HasAVX ? X86::VMOVAPSrr : X86::MOVAPSrr;
  else if (X86::VR256RegClass.contains(DestReg, SrcReg))
    Opc = X86::VMOVAPSYrr;
  if (!Opc)
    Opc = CopyToFromAsymmetricReg(DestReg, SrcReg, TM.getSubtarget<X86Subtarget>());

  if (Opc) {
    BuildMI(MBB, MI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Moving EFLAGS to / from another register requires a push and a pop.
  // Notice that we have to adjust the stack if we don't want to clobber the
  // first frame index. See X86FrameLowering.cpp - colobbersTheStack.
  if (SrcReg == X86::EFLAGS) {
    if (X86::GR64RegClass.contains(DestReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSHF64));
      BuildMI(MBB, MI, DL, get(X86::POP64r), DestReg);
      return;
    }
    if (X86::GR32RegClass.contains(DestReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSHF32));
      BuildMI(MBB, MI, DL, get(X86::POP32r), DestReg);
      return;
    }
  }
  if (DestReg == X86::EFLAGS) {
    if (X86::GR64RegClass.contains(SrcReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSH64r))
        .addReg(SrcReg, getKillRegState(KillSrc));
      BuildMI(MBB, MI, DL, get(X86::POPF64));
      return;
    }
    if (X86::GR32RegClass.contains(SrcReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSH32r))
        .addReg(SrcReg, getKillRegState(KillSrc));
      BuildMI(MBB, MI, DL, get(X86::POPF32));
      return;
    }
  }

  DEBUG(dbgs() << "Cannot copy " << RI.getName(SrcReg)
               << " to " << RI.getName(DestReg) << '\n');
  llvm_unreachable("Cannot emit physreg copy instruction");
}

static unsigned getLoadStoreRegOpcode(unsigned Reg,
                                      const TargetRegisterClass *RC,
                                      bool isStackAligned,
                                      const TargetMachine &TM,
                                      bool load) {
  if (TM.getSubtarget<X86Subtarget>().hasAVX512()) {
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

  bool HasAVX = TM.getSubtarget<X86Subtarget>().hasAVX();
  switch (RC->getSize()) {
  default:
    llvm_unreachable("Unknown spill size");
  case 1:
    assert(X86::GR8RegClass.hasSubClassEq(RC) && "Unknown 1-byte regclass");
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
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

static unsigned getStoreRegOpcode(unsigned SrcReg,
                                  const TargetRegisterClass *RC,
                                  bool isStackAligned,
                                  TargetMachine &TM) {
  return getLoadStoreRegOpcode(SrcReg, RC, isStackAligned, TM, false);
}


static unsigned getLoadRegOpcode(unsigned DestReg,
                                 const TargetRegisterClass *RC,
                                 bool isStackAligned,
                                 const TargetMachine &TM) {
  return getLoadStoreRegOpcode(DestReg, RC, isStackAligned, TM, true);
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
  bool isAligned = (TM.getFrameLowering()->getStackAlignment() >= Alignment) ||
    RI.canRealignStack(MF);
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, TM);
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
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, TM);
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
  bool isAligned = (TM.getFrameLowering()->getStackAlignment() >= Alignment) ||
    RI.canRealignStack(MF);
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, TM);
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
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, TM);
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

/// isRedundantFlagInstr - check whether the first instruction, whose only
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

/// isDefConvertible - check whether the definition can be converted
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
  case X86::DEC64_32r: case X86::DEC64_16r:
  case X86::ADD64ri32: case X86::ADD64ri8: case X86::ADD32ri:
  case X86::ADD32ri8:  case X86::ADD16ri:  case X86::ADD16ri8:
  case X86::ADD8ri:    case X86::ADD64rr:  case X86::ADD32rr:
  case X86::ADD16rr:   case X86::ADD8rr:   case X86::ADD64rm:
  case X86::ADD32rm:   case X86::ADD16rm:  case X86::ADD8rm:
  case X86::INC64r:    case X86::INC32r:   case X86::INC16r: case X86::INC8r:
  case X86::INC64_32r: case X86::INC64_16r:
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

/// optimizeCompareInstr - Check if there exists an earlier instruction that
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
  if (IsCmpZero && (MI->getParent() != CmpInstr->getParent() ||
      !isDefConvertible(MI)))
    return false;

  // We are searching for an earlier instruction that can make CmpInstr
  // redundant and that instruction will be saved in Sub.
  MachineInstr *Sub = NULL;
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
  MachineInstr *Movr0Inst = 0;
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
    X86::CondCode OldCC;
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
    } else if (IsSwapped) {
      // If we have SUB(r1, r2) and CMP(r2, r1), the condition code needs
      // to be changed from r2 > r1 to r1 < r2, from r2 < r1 to r1 > r2, etc.
      // We swap the condition code and synthesize the new opcode.
      X86::CondCode NewCC = getSwappedCondition(OldCC);
      if (NewCC == X86::COND_INVALID) return false;

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

/// optimizeLoadInstr - Try to remove the load by folding it to a register
/// operand at the use. We fold the load instructions if load defines a virtual
/// register, the virtual register is used once in the same BB, and the
/// instructions in-between do not load or store, and have no side effects.
MachineInstr* X86InstrInfo::
optimizeLoadInstr(MachineInstr *MI, const MachineRegisterInfo *MRI,
                  unsigned &FoldAsLoadDefReg,
                  MachineInstr *&DefMI) const {
  if (FoldAsLoadDefReg == 0)
    return 0;
  // To be conservative, if there exists another load, clear the load candidate.
  if (MI->mayLoad()) {
    FoldAsLoadDefReg = 0;
    return 0;
  }

  // Check whether we can move DefMI here.
  DefMI = MRI->getVRegDef(FoldAsLoadDefReg);
  assert(DefMI);
  bool SawStore = false;
  if (!DefMI->isSafeToMove(this, 0, SawStore))
    return 0;

  // We try to commute MI if possible.
  unsigned IdxEnd = (MI->isCommutable()) ? 2 : 1;
  for (unsigned Idx = 0; Idx < IdxEnd; Idx++) {
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
        return 0;

      SrcOperandId = i;
      FoundSrcOperand = true;
    }
    if (!FoundSrcOperand) return 0;

    // Check whether we can fold the def into SrcOperandId.
    SmallVector<unsigned, 8> Ops;
    Ops.push_back(SrcOperandId);
    MachineInstr *FoldMI = foldMemoryOperand(MI, Ops, DefMI);
    if (FoldMI) {
      FoldAsLoadDefReg = 0;
      return FoldMI;
    }

    if (Idx == 1) {
      // MI was changed but it didn't help, commute it back!
      commuteInstruction(MI, false);
      return 0;
    }

    // Check whether we can commute MI and enable folding.
    if (MI->isCommutable()) {
      MachineInstr *NewMI = commuteInstruction(MI, false);
      // Unable to commute.
      if (!NewMI) return 0;
      if (NewMI != MI) {
        // New instruction. It doesn't need to be kept.
        NewMI->eraseFromParent();
        return 0;
      }
    }
  }
  return 0;
}

/// Expand2AddrUndef - Expand a single-def pseudo instruction to a two-addr
/// instruction with two undef reads of the register being defined.  This is
/// used for mapping:
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

bool X86InstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  bool HasAVX = TM.getSubtarget<X86Subtarget>().hasAVX();
  MachineInstrBuilder MIB(*MI->getParent()->getParent(), MI);
  switch (MI->getOpcode()) {
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
  case X86::KSET0W: return Expand2AddrUndef(MIB, get(X86::KXORWrr));
  case X86::KSET1B:
  case X86::KSET1W: return Expand2AddrUndef(MIB, get(X86::KXNORWrr));
  }
  return false;
}

static MachineInstr *FuseTwoAddrInst(MachineFunction &MF, unsigned Opcode,
                                     const SmallVectorImpl<MachineOperand> &MOs,
                                     MachineInstr *MI,
                                     const TargetInstrInfo &TII) {
  // Create the base instruction with the memory operand as the first part.
  // Omit the implicit operands, something BuildMI can't do.
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);
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
  // Omit the implicit operands, something BuildMI can't do.
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);

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
                                    unsigned Size, unsigned Align) const {
  const DenseMap<unsigned, std::pair<unsigned,unsigned> > *OpcodeTablePtr = 0;
  bool isCallRegIndirect = TM.getSubtarget<X86Subtarget>().callRegIndirect();
  bool isTwoAddrFold = false;

  // Atom favors register form of call. So, we do not fold loads into calls
  // when X86Subtarget is Atom.
  if (isCallRegIndirect &&
    (MI->getOpcode() == X86::CALL32r || MI->getOpcode() == X86::CALL64r)) {
    return NULL;
  }

  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // FIXME: AsmPrinter doesn't know how to handle
  // X86II::MO_GOT_ABSOLUTE_ADDRESS after folding.
  if (MI->getOpcode() == X86::ADD32ri &&
      MI->getOperand(2).getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS)
    return NULL;

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
    if (MI->getOpcode() == X86::MOV32r0) {
      NewMI = MakeM0Inst(*this, X86::MOV32mi, MOs, MI);
      if (NewMI)
        return NewMI;
    }

    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (i == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (i == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  } else if (i == 3) {
    OpcodeTablePtr = &RegOp2MemOpTable3;
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
        return NULL;
      bool NarrowToMOV32rm = false;
      if (Size) {
        unsigned RCSize = getRegClass(MI->getDesc(), i, &RI, MF)->getSize();
        if (Size < RCSize) {
          // Check if it's safe to fold the load. If the size of the object is
          // narrower than the load width, then it's not.
          if (Opcode != X86::MOV64rm || RCSize != 8 || Size != 4)
            return NULL;
          // If this is a 64-bit load, but the spill slot is 32, then we can do
          // a 32-bit load which is implicitly zero-extended. This likely is due
          // to liveintervalanalysis remat'ing a load from stack slot.
          if (MI->getOperand(0).getSubReg() || MI->getOperand(1).getSubReg())
            return NULL;
          Opcode = X86::MOV32rm;
          NarrowToMOV32rm = true;
        }
      }

      if (isTwoAddrFold)
        NewMI = FuseTwoAddrInst(MF, Opcode, MOs, MI, *this);
      else
        NewMI = FuseInst(MF, Opcode, i, MOs, MI, *this);

      if (NarrowToMOV32rm) {
        // If this is the special case where we use a MOV32rm to load a 32-bit
        // value and zero-extend the top bits. Change the destination register
        // to a 32-bit one.
        unsigned DstReg = NewMI->getOperand(0).getReg();
        if (TargetRegisterInfo::isPhysicalRegister(DstReg))
          NewMI->getOperand(0).setReg(RI.getSubReg(DstReg,
                                                   X86::sub_32bit));
        else
          NewMI->getOperand(0).setSubReg(X86::sub_32bit);
      }
      return NewMI;
    }
  }

  // No fusion
  if (PrintFailedFusing && !MI->isCopy())
    dbgs() << "We failed to fuse operand " << i << " in " << *MI;
  return NULL;
}

/// hasPartialRegUpdate - Return true for all instructions that only update
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
  case X86::CVTSI2SS64rr:
  case X86::CVTSI2SDrr:
  case X86::CVTSI2SD64rr:
  case X86::CVTSD2SSrr:
  case X86::Int_CVTSD2SSrr:
  case X86::CVTSS2SDrr:
  case X86::Int_CVTSS2SDrr:
  case X86::RCPSSr:
  case X86::RCPSSr_Int:
  case X86::ROUNDSDr:
  case X86::ROUNDSDr_Int:
  case X86::ROUNDSSr:
  case X86::ROUNDSSr_Int:
  case X86::RSQRTSSr:
  case X86::RSQRTSSr_Int:
  case X86::SQRTSSr:
  case X86::SQRTSSr_Int:
    return true;
  }

  return false;
}

/// getPartialRegUpdateClearance - Inform the ExeDepsFix pass how many idle
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
  case X86::Int_VCVTSI2SSrr:
  case X86::VCVTSI2SS64rr:
  case X86::Int_VCVTSI2SS64rr:
  case X86::VCVTSI2SDrr:
  case X86::Int_VCVTSI2SDrr:
  case X86::VCVTSI2SD64rr:
  case X86::Int_VCVTSI2SD64rr:
  case X86::VCVTSD2SSrr:
  case X86::Int_VCVTSD2SSrr:
  case X86::VCVTSS2SDrr:
  case X86::Int_VCVTSS2SDrr:
  case X86::VRCPSSr:
  case X86::VROUNDSDr:
  case X86::VROUNDSDr_Int:
  case X86::VROUNDSSr:
  case X86::VROUNDSSr_Int:
  case X86::VRSQRTSSr:
  case X86::VSQRTSSr:

  // AVX-512
  case X86::VCVTSD2SSZrr:
  case X86::VCVTSS2SDZrr:
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
    bool HasAVX = TM.getSubtarget<X86Subtarget>().hasAVX();
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

static MachineInstr* foldPatchpoint(MachineFunction &MF,
                                    MachineInstr *MI,
                                    const SmallVectorImpl<unsigned> &Ops,
                                    int FrameIndex,
                                    const TargetInstrInfo &TII) {
  unsigned StartIdx = 0;
  switch (MI->getOpcode()) {
  case TargetOpcode::STACKMAP:
    StartIdx = 2; // Skip ID, nShadowBytes.
    break;
  case TargetOpcode::PATCHPOINT: {
    // For PatchPoint, the call args are not foldable.
    PatchPointOpers opers(MI);
    StartIdx = opers.getVarIdx();
    break;
  }
  default:
    llvm_unreachable("unexpected stackmap opcode");
  }

  // Return false if any operands requested for folding are not foldable (not
  // part of the stackmap's live values).
  for (SmallVectorImpl<unsigned>::const_iterator I = Ops.begin(), E = Ops.end();
       I != E; ++I) {
    if (*I < StartIdx)
      return 0;
  }

  MachineInstr *NewMI =
    MF.CreateMachineInstr(TII.get(MI->getOpcode()), MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);

  // No need to fold return, the meta data, and function arguments
  for (unsigned i = 0; i < StartIdx; ++i)
    MIB.addOperand(MI->getOperand(i));

  for (unsigned i = StartIdx; i < MI->getNumOperands(); ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (std::find(Ops.begin(), Ops.end(), i) != Ops.end()) {
      assert(MO.getReg() && "patchpoint can only fold a vreg operand");
      // Compute the spill slot size and offset.
      const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(MO.getReg());
      unsigned SpillSize;
      unsigned SpillOffset;
      bool Valid = TII.getStackSlotRange(RC, MO.getSubReg(), SpillSize,
                                         SpillOffset, &MF.getTarget());
      if (!Valid)
        report_fatal_error("cannot spill patchpoint subregister operand");

      MIB.addOperand(MachineOperand::CreateImm(StackMaps::IndirectMemRefOp));
      MIB.addOperand(MachineOperand::CreateImm(SpillSize));
      MIB.addOperand(MachineOperand::CreateFI(FrameIndex));
      addOffset(MIB, SpillOffset);
    }
    else
      MIB.addOperand(MO);
  }
  return NewMI;
}

MachineInstr*
X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                                    const SmallVectorImpl<unsigned> &Ops,
                                    int FrameIndex) const {
  // Special case stack map and patch point intrinsics.
  if (MI->getOpcode() == TargetOpcode::STACKMAP
      || MI->getOpcode() == TargetOpcode::PATCHPOINT) {
    return foldPatchpoint(MF, MI, Ops, FrameIndex, *this);
  }
  // Check switch flag
  if (NoFusing) return NULL;

  // Unless optimizing for size, don't fold to avoid partial
  // register update stalls
  if (!MF.getFunction()->getAttributes().
        hasAttribute(AttributeSet::FunctionIndex, Attribute::OptimizeForSize) &&
      hasPartialRegUpdate(MI->getOpcode()))
    return 0;

  const MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned Size = MFI->getObjectSize(FrameIndex);
  unsigned Alignment = MFI->getObjectAlignment(FrameIndex);
  // If the function stack isn't realigned we don't want to fold instructions
  // that need increased alignment.
  if (!RI.needsStackRealignment(MF))
    Alignment = std::min(Alignment, TM.getFrameLowering()->getStackAlignment());
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    unsigned RCSize = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; RCSize = 1; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; RCSize = 2; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; RCSize = 4; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; RCSize = 8; break;
    }
    // Check if it's safe to fold the load. If the size of the object is
    // narrower than the load width, then it's not.
    if (Size < RCSize)
      return NULL;
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  SmallVector<MachineOperand,4> MOs;
  MOs.push_back(MachineOperand::CreateFI(FrameIndex));
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, Size, Alignment);
}

MachineInstr* X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                                  MachineInstr *LoadMI) const {
  // If loading from a FrameIndex, fold directly from the FrameIndex.
  unsigned NumOps = LoadMI->getDesc().getNumOperands();
  int FrameIndex;
  if (isLoadFromStackSlot(LoadMI, FrameIndex))
    return foldMemoryOperandImpl(MF, MI, Ops, FrameIndex);

  // Check switch flag
  if (NoFusing) return NULL;

  // Unless optimizing for size, don't fold to avoid partial
  // register update stalls
  if (!MF.getFunction()->getAttributes().
        hasAttribute(AttributeSet::FunctionIndex, Attribute::OptimizeForSize) &&
      hasPartialRegUpdate(MI->getOpcode()))
    return 0;

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
      return 0;
    }
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  // Make sure the subregisters match.
  // Otherwise we risk changing the size of the load.
  if (LoadMI->getOperand(0).getSubReg() != MI->getOperand(Ops[0]).getSubReg())
    return NULL;

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
    if (TM.getCodeModel() != CodeModel::Small &&
        TM.getCodeModel() != CodeModel::Kernel)
      return NULL;

    // x86-32 PIC requires a PIC base register for constant pools.
    unsigned PICBase = 0;
    if (TM.getRelocationModel() == Reloc::PIC_) {
      if (TM.getSubtarget<X86Subtarget>().is64Bit())
        PICBase = X86::RIP;
      else
        // FIXME: PICBase = getGlobalBaseReg(&MF);
        // This doesn't work for several reasons.
        // 1. GlobalBaseReg may have been spilled.
        // 2. It may not be live at MI.
        return NULL;
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
    if ((LoadMI->getOpcode() == X86::MOVSSrm ||
         LoadMI->getOpcode() == X86::VMOVSSrm) &&
        MF.getRegInfo().getRegClass(LoadMI->getOperand(0).getReg())->getSize()
          > 4)
      // These instructions only load 32 bits, we can't fold them if the
      // destination register is wider than 32 bits (4 bytes).
      return NULL;
    if ((LoadMI->getOpcode() == X86::MOVSDrm ||
         LoadMI->getOpcode() == X86::VMOVSDrm) &&
        MF.getRegInfo().getRegClass(LoadMI->getOperand(0).getReg())->getSize()
          > 8)
      // These instructions only load 64 bits, we can't fold them if the
      // destination register is wider than 64 bits (8 bytes).
      return NULL;

    // Folding a normal load. Just copy the load's address operands.
    for (unsigned i = NumOps - X86::AddrNumOperands; i != NumOps; ++i)
      MOs.push_back(LoadMI->getOperand(i));
    break;
  }
  }
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, 0, Alignment);
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
    case X86::ADD32ri:
      // FIXME: AsmPrinter doesn't know how to handle
      // X86II::MO_GOT_ABSOLUTE_ADDRESS after folding.
      if (MI->getOperand(2).getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS)
        return false;
      break;
    }
  }

  if (Ops.size() != 1)
    return false;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  const DenseMap<unsigned, std::pair<unsigned,unsigned> > *OpcodeTablePtr = 0;
  if (isTwoAddr && NumOps >= 2 && OpNum < 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
  } else if (OpNum == 0) { // If operand 0
    if (Opc == X86::MOV32r0)
      return true;

    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (OpNum == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (OpNum == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  } else if (OpNum == 3) {
    OpcodeTablePtr = &RegOp2MemOpTable3;
  }

  if (OpcodeTablePtr && OpcodeTablePtr->count(Opc))
    return true;
  return TargetInstrInfo::canFoldMemoryOperand(MI, Ops);
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
  if (!MI->hasOneMemOperand() &&
      RC == &X86::VR128RegClass &&
      !TM.getSubtarget<X86Subtarget>().isUnalignedMemAccessFast())
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
  SDNode *Load = 0;
  if (FoldedLoad) {
    EVT VT = *RC->vt_begin();
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractLoadMemRefs(cast<MachineSDNode>(N)->memoperands_begin(),
                            cast<MachineSDNode>(N)->memoperands_end());
    if (!(*MMOs.first) &&
        RC == &X86::VR128RegClass &&
        !TM.getSubtarget<X86Subtarget>().isUnalignedMemAccessFast())
      // Do not introduce a slow unaligned load.
      return false;
    unsigned Alignment = RC->getSize() == 32 ? 32 : 16;
    bool isAligned = (*MMOs.first) &&
                     (*MMOs.first)->getAlignment() >= Alignment;
    Load = DAG.getMachineNode(getLoadRegOpcode(0, RC, isAligned, TM), dl,
                              VT, MVT::Other, AddrOps);
    NewNodes.push_back(Load);

    // Preserve memory reference information.
    cast<MachineSDNode>(Load)->setMemRefs(MMOs.first, MMOs.second);
  }

  // Emit the data processing instruction.
  std::vector<EVT> VTs;
  const TargetRegisterClass *DstRC = 0;
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
  std::copy(AfterOps.begin(), AfterOps.end(), std::back_inserter(BeforeOps));
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
        !TM.getSubtarget<X86Subtarget>().isUnalignedMemAccessFast())
      // Do not introduce a slow unaligned store.
      return false;
    unsigned Alignment = RC->getSize() == 32 ? 32 : 16;
    bool isAligned = (*MMOs.first) &&
                     (*MMOs.first)->getAlignment() >= Alignment;
    SDNode *Store = DAG.getMachineNode(getStoreRegOpcode(0, DstRC,
                                                         isAligned, TM),
                                       dl, MVT::Other, AddrOps);
    NewNodes.push_back(Store);

    // Preserve memory reference information.
    cast<MachineSDNode>(Load)->setMemRefs(MMOs.first, MMOs.second);
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
    if (TM.getSubtargetImpl()->is64Bit()) {
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
  if (!TM.getSubtarget<X86Subtarget>().hasAVX())
    return false;

  enum {
    FuseTest,
    FuseCmp,
    FuseInc
  } FuseKind;

  switch(Second->getOpcode()) {
  default:
    return false;
  case X86::JE_4:
  case X86::JNE_4:
  case X86::JL_4:
  case X86::JLE_4:
  case X86::JG_4:
  case X86::JGE_4:
    FuseKind = FuseInc;
    break;
  case X86::JB_4:
  case X86::JBE_4:
  case X86::JA_4:
  case X86::JAE_4:
    FuseKind = FuseCmp;
    break;
  case X86::JS_4:
  case X86::JNS_4:
  case X86::JP_4:
  case X86::JNP_4:
  case X86::JO_4:
  case X86::JNO_4:
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
  case X86::INC64_16r:
  case X86::INC64_32r:
  case X86::INC64r:
  case X86::INC8r:
  case X86::DEC16r:
  case X86::DEC32r:
  case X86::DEC64_16r:
  case X86::DEC64_32r:
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

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
/// TODO: Eliminate this and move the code to X86MachineFunctionInfo.
///
unsigned X86InstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  assert(!TM.getSubtarget<X86Subtarget>().is64Bit() &&
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
  { X86::VPERM2F128rr,   X86::VPERM2F128rr,   X86::VPERM2I128rr }
};

// FIXME: Some shuffle and unpack instructions have equivalents in different
// domains, but they require a bit more work than just switching opcodes.

static const uint16_t *lookup(unsigned opcode, unsigned domain) {
  for (unsigned i = 0, e = array_lengthof(ReplaceableInstrs); i != e; ++i)
    if (ReplaceableInstrs[i][domain-1] == opcode)
      return ReplaceableInstrs[i];
  return 0;
}

static const uint16_t *lookupAVX2(unsigned opcode, unsigned domain) {
  for (unsigned i = 0, e = array_lengthof(ReplaceableInstrsAVX2); i != e; ++i)
    if (ReplaceableInstrsAVX2[i][domain-1] == opcode)
      return ReplaceableInstrsAVX2[i];
  return 0;
}

std::pair<uint16_t, uint16_t>
X86InstrInfo::getExecutionDomain(const MachineInstr *MI) const {
  uint16_t domain = (MI->getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  bool hasAVX2 = TM.getSubtarget<X86Subtarget>().hasAVX2();
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
    assert((TM.getSubtarget<X86Subtarget>().hasAVX2() || Domain < 3) &&
           "256-bit vector operations only available in AVX2");
    table = lookupAVX2(MI->getOpcode(), dom);
  }
  assert(table && "Cannot change domain");
  MI->setDesc(get(table[Domain-1]));
}

/// getNoopForMachoTarget - Return the noop instruction to use for a noop.
void X86InstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  NopInst.setOpcode(X86::NOOP);
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
  case X86::VSQRTPDZrm:
  case X86::VSQRTPDZrr:
  case X86::VSQRTPSZrm:
  case X86::VSQRTPSZrr:
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
hasHighOperandLatency(const InstrItineraryData *ItinData,
                      const MachineRegisterInfo *MRI,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx) const {
  return isHighLatencyDef(DefMI->getOpcode());
}

namespace {
  /// CGBR - Create Global Base Reg pass. This initializes the PIC
  /// global base register for x86-32.
  struct CGBR : public MachineFunctionPass {
    static char ID;
    CGBR() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      const X86TargetMachine *TM =
        static_cast<const X86TargetMachine *>(&MF.getTarget());

      assert(!TM->getSubtarget<X86Subtarget>().is64Bit() &&
             "X86-64 PIC uses RIP relative addressing");

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
      const X86InstrInfo *TII = TM->getInstrInfo();

      unsigned PC;
      if (TM->getSubtarget<X86Subtarget>().isPICStyleGOT())
        PC = RegInfo.createVirtualRegister(&X86::GR32RegClass);
      else
        PC = GlobalBaseReg;

      // Operand of MovePCtoStack is completely ignored by asm printer. It's
      // only used in JIT code emission as displacement to pc.
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::MOVPC32r), PC).addImm(0);

      // If we're using vanilla 'GOT' PIC style, we should use relative addressing
      // not to pc, but to _GLOBAL_OFFSET_TABLE_ external.
      if (TM->getSubtarget<X86Subtarget>().isPICStyleGOT()) {
        // Generate addl $__GLOBAL_OFFSET_TABLE_ + [.-piclabel], %some_register
        BuildMI(FirstMBB, MBBI, DL, TII->get(X86::ADD32ri), GlobalBaseReg)
          .addReg(PC).addExternalSymbol("_GLOBAL_OFFSET_TABLE_",
                                        X86II::MO_GOT_ABSOLUTE_ADDRESS);
      }

      return true;
    }

    virtual const char *getPassName() const {
      return "X86 PIC Global Base Reg Initialization";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char CGBR::ID = 0;
FunctionPass*
llvm::createGlobalBaseRegPass() { return new CGBR(); }

namespace {
  struct LDTLSCleanup : public MachineFunctionPass {
    static char ID;
    LDTLSCleanup() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
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
      const X86TargetMachine *TM =
          static_cast<const X86TargetMachine *>(&MF->getTarget());
      const bool is64Bit = TM->getSubtarget<X86Subtarget>().is64Bit();
      const X86InstrInfo *TII = TM->getInstrInfo();

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
      const X86TargetMachine *TM =
          static_cast<const X86TargetMachine *>(&MF->getTarget());
      const bool is64Bit = TM->getSubtarget<X86Subtarget>().is64Bit();
      const X86InstrInfo *TII = TM->getInstrInfo();

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

    virtual const char *getPassName() const {
      return "Local Dynamic TLS Access Clean-up";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char LDTLSCleanup::ID = 0;
FunctionPass*
llvm::createCleanupLocalDynamicTLSPass() { return new LDTLSCleanup(); }
