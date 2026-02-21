//===-- NVPTX.h - Top-level interface for NVPTX representation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM NVPTX back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTX_H
#define LLVM_LIB_TARGET_NVPTX_NVPTX_H

#include "llvm/CodeGen/ISDOpcodes.h"

#include "llvm/ADT/Bitfields.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class Function;
class FunctionPass;
class MachineMemOperand;
class MachineFunctionPass;
class NVPTXTargetMachine;
class PassRegistry;
class Value;

namespace NVPTXCC {
enum CondCodes {
  EQ,
  NE,
  LT,
  LE,
  GT,
  GE
};
}

FunctionPass *createNVPTXISelDag(NVPTXTargetMachine &TM,
                                 llvm::CodeGenOptLevel OptLevel);
ModulePass *createNVPTXAssignValidGlobalNamesPass();
ModulePass *createGenericToNVVMLegacyPass();
ModulePass *createNVPTXCtorDtorLoweringLegacyPass();
FunctionPass *createNVVMIntrRangePass();
ModulePass *createNVVMReflectPass(unsigned int SmVersion);
MachineFunctionPass *createNVPTXPrologEpilogPass();
MachineFunctionPass *createNVPTXReplaceImageHandlesPass();
FunctionPass *createNVPTXImageOptimizerPass();
FunctionPass *createNVPTXLowerArgsPass();
FunctionPass *createNVPTXLowerAllocaPass();
FunctionPass *createNVPTXLowerUnreachablePass(bool TrapUnreachable,
                                              bool NoTrapAfterNoreturn);
FunctionPass *createNVPTXTagInvariantLoadsPass();
FunctionPass *createNVPTXIRPeepholePass();
MachineFunctionPass *createNVPTXPeephole();
MachineFunctionPass *createNVPTXProxyRegErasurePass();
MachineFunctionPass *createNVPTXForwardParamsPass();

void initializeNVVMReflectLegacyPassPass(PassRegistry &);
void initializeGenericToNVVMLegacyPassPass(PassRegistry &);
void initializeNVPTXAllocaHoistingPass(PassRegistry &);
void initializeNVPTXAsmPrinterPass(PassRegistry &);
void initializeNVPTXAssignValidGlobalNamesPass(PassRegistry &);
void initializeNVPTXAtomicLowerPass(PassRegistry &);
void initializeNVPTXCtorDtorLoweringLegacyPass(PassRegistry &);
void initializeNVPTXLowerAggrCopiesPass(PassRegistry &);
void initializeNVPTXLowerAllocaPass(PassRegistry &);
void initializeNVPTXLowerUnreachablePass(PassRegistry &);
void initializeNVPTXLowerArgsLegacyPassPass(PassRegistry &);
void initializeNVPTXProxyRegErasurePass(PassRegistry &);
void initializeNVPTXForwardParamsPassPass(PassRegistry &);
void initializeNVVMIntrRangePass(PassRegistry &);
void initializeNVVMReflectPass(PassRegistry &);
void initializeNVPTXAAWrapperPassPass(PassRegistry &);
void initializeNVPTXExternalAAWrapperPass(PassRegistry &);
void initializeNVPTXPeepholePass(PassRegistry &);
void initializeNVPTXTagInvariantLoadLegacyPassPass(PassRegistry &);
void initializeNVPTXIRPeepholePass(PassRegistry &);
void initializeNVPTXPrologEpilogPassPass(PassRegistry &);

struct NVVMIntrRangePass : PassInfoMixin<NVVMIntrRangePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct NVPTXIRPeepholePass : PassInfoMixin<NVPTXIRPeepholePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct NVVMReflectPass : PassInfoMixin<NVVMReflectPass> {
  NVVMReflectPass() : SmVersion(0) {}
  NVVMReflectPass(unsigned SmVersion) : SmVersion(SmVersion) {}
  PreservedAnalyses run(Module &F, ModuleAnalysisManager &AM);

private:
  unsigned SmVersion;
};

struct GenericToNVVMPass : PassInfoMixin<GenericToNVVMPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

struct NVPTXCopyByValArgsPass : PassInfoMixin<NVPTXCopyByValArgsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct NVPTXLowerArgsPass : PassInfoMixin<NVPTXLowerArgsPass> {
private:
  TargetMachine &TM;

public:
  NVPTXLowerArgsPass(TargetMachine &TM) : TM(TM) {};
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct NVPTXTagInvariantLoadsPass : PassInfoMixin<NVPTXTagInvariantLoadsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

namespace NVPTX {
enum DrvInterface {
  NVCL,
  CUDA
};

// A field inside TSFlags needs a shift and a mask. The usage is
// always as follows :
// ((TSFlags & fieldMask) >> fieldShift)
// The enum keeps the mask, the shift, and all valid values of the
// field in one place.
enum VecInstType {
  VecInstTypeShift = 0,
  VecInstTypeMask = 0xF,

  VecNOP = 0,
  VecLoad = 1,
  VecStore = 2,
  VecBuild = 3,
  VecShuffle = 4,
  VecExtract = 5,
  VecInsert = 6,
  VecDest = 7,
  VecOther = 15
};

enum SimpleMove {
  SimpleMoveMask = 0x10,
  SimpleMoveShift = 4
};
enum LoadStore {
  isLoadMask = 0x20,
  isLoadShift = 5,
  isStoreMask = 0x40,
  isStoreShift = 6
};

// Extends LLVM AtomicOrdering with PTX Orderings:
using OrderingUnderlyingType = unsigned int;
enum Ordering : OrderingUnderlyingType {
  NotAtomic = (OrderingUnderlyingType)
      AtomicOrdering::NotAtomic, // PTX calls these: "Weak"
  // Unordered = 1, // NVPTX maps LLVM Unorderd to Relaxed
  Relaxed = (OrderingUnderlyingType)AtomicOrdering::Monotonic,
  // Consume = 3,   // Unimplemented in LLVM; NVPTX would map to "Acquire"
  Acquire = (OrderingUnderlyingType)AtomicOrdering::Acquire,
  Release = (OrderingUnderlyingType)AtomicOrdering::Release,
  AcquireRelease = (OrderingUnderlyingType)AtomicOrdering::AcquireRelease,
  SequentiallyConsistent =
      (OrderingUnderlyingType)AtomicOrdering::SequentiallyConsistent,
  Volatile = SequentiallyConsistent + 1,
  RelaxedMMIO = Volatile + 1,
};

using ScopeUnderlyingType = unsigned int;
enum Scope : ScopeUnderlyingType {
  Thread = 0,
  Block = 1,
  Cluster = 2,
  Device = 3,
  System = 4,
  DefaultDevice = 5, //  For SM < 70: denotes PTX op implicit/default .gpu scope
  LASTSCOPE = DefaultDevice
};

using AddressSpaceUnderlyingType = unsigned int;
enum AddressSpace : AddressSpaceUnderlyingType {
  Generic = 0,
  Global = 1,
  Shared = 3,
  Const = 4,
  Local = 5,
  SharedCluster = 7,

  // NVPTX Backend Private:
  Param = 101
};

// Cache hint enums for !mem.cache_hint metadata
// These correspond to PTX cache control qualifiers

// L1 Eviction Policy - maps to PTX L1::evict_* qualifiers
enum class L1Eviction : uint8_t {
  Normal = 0,     // Default behavior (no qualifier)
  Unchanged = 1,  // L1::evict_unchanged
  First = 2,      // L1::evict_first
  Last = 3,       // L1::evict_last
  NoAllocate = 4, // L1::no_allocate
};

// L2 Eviction Policy - maps to PTX L2::evict_* qualifiers
enum class L2Eviction : uint8_t {
  Normal = 0, // Default behavior (no qualifier)
  First = 1,  // L2::evict_first
  Last = 2,   // L2::evict_last
};

// L2 Prefetch Size - maps to PTX L2::*B qualifiers
enum class L2Prefetch : uint8_t {
  None = 0,     // No prefetch hint
  Bytes64 = 1,  // L2::64B
  Bytes128 = 2, // L2::128B
  Bytes256 = 3, // L2::256B
};

// Bitfield layout for encoded cache control hints (stored in unsigned):
// Bits 0-2:  L1 Eviction (3 bits, 5 values)
// Bits 3-4:  L2 Eviction (2 bits, 3 values)
// Bits 5-6:  L2 Prefetch (2 bits, 4 values)
// Bit 7:    L2::cache_hint mode flag (when set, use CachePolicy operand)
// Bits 8-31: Reserved
//
// Using llvm::Bitfield for type-safe access with compile-time validation.
using L1EvictionBits =
    Bitfield::Element<L1Eviction, 0, 3, L1Eviction::NoAllocate>;
using L2EvictionBits = Bitfield::Element<L2Eviction, 3, 2, L2Eviction::Last>;
using L2PrefetchBits =
    Bitfield::Element<L2Prefetch, 5, 2, L2Prefetch::Bytes256>;
using L2CacheHintBit = Bitfield::Element<bool, 7, 1>;

inline unsigned encodeCacheControlHint(L1Eviction L1, L2Eviction L2,
                                       L2Prefetch P) {
  unsigned Hint = 0;
  Bitfield::set<L1EvictionBits>(Hint, L1);
  Bitfield::set<L2EvictionBits>(Hint, L2);
  Bitfield::set<L2PrefetchBits>(Hint, P);
  return Hint;
}

inline L1Eviction decodeL1Eviction(unsigned Hint) {
  return Bitfield::get<L1EvictionBits>(Hint);
}

inline L2Eviction decodeL2Eviction(unsigned Hint) {
  return Bitfield::get<L2EvictionBits>(Hint);
}

inline L2Prefetch decodeL2Prefetch(unsigned Hint) {
  return Bitfield::get<L2PrefetchBits>(Hint);
}

inline bool isL2CacheHintMode(unsigned Hint) {
  return Bitfield::get<L2CacheHintBit>(Hint);
}

namespace PTXLdStInstCode {
enum FromType { Unsigned = 0, Signed, Float, Untyped };
} // namespace PTXLdStInstCode

/// PTXCvtMode - Conversion code enumeration
namespace PTXCvtMode {
enum CvtMode {
  NONE = 0,
  RNI,
  RZI,
  RMI,
  RPI,
  RN,
  RZ,
  RM,
  RP,
  RNA,
  RS,

  BASE_MASK = 0x0F,
  FTZ_FLAG = 0x10,
  SAT_FLAG = 0x20,
  RELU_FLAG = 0x40
};
}

/// PTXCmpMode - Comparison mode enumeration
namespace PTXCmpMode {
enum CmpMode {
  EQ = 0,
  NE,
  LT,
  LE,
  GT,
  GE,
  EQU,
  NEU,
  LTU,
  LEU,
  GTU,
  GEU,
  NUM,
  // NAN is a MACRO
  NotANumber,
};
}

namespace PTXPrmtMode {
enum PrmtMode {
  NONE,
  F4E,
  B4E,
  RC8,
  ECL,
  ECR,
  RC16,
};
}

enum class DivPrecisionLevel : unsigned {
  Approx = 0,
  Full = 1,
  IEEE754 = 2,
  IEEE754_NoFTZ = 3,
};

} // namespace NVPTX
void initializeNVPTXDAGToDAGISelLegacyPass(PassRegistry &);
} // namespace llvm

// Defines symbolic names for NVPTX registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "NVPTXGenRegisterInfo.inc"

// Defines symbolic names for the NVPTX instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "NVPTXGenInstrInfo.inc"

// Pull in OpName enum and getNamedOperandIdx() for LD/ST instructions.
// Generated from UseNamedOperandTable=1 in NVPTXInstrInfo.td.
#define GET_INSTRINFO_OPERAND_ENUM
#include "NVPTXGenInstrInfo.inc"

#endif
