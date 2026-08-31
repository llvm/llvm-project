//===- GCNPreRAAntiHints.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNPRERAANTIHINTS_H
#define LLVM_LIB_TARGET_AMDGPU_GCNPRERAANTIHINTS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

class MachineFunction;
class MachineInstr;
class MachineRegisterInfo;
class LiveIntervals;
class SIRegisterInfo;
class SIInstrInfo;
class GCNSubtarget;
class TargetSchedModel;

namespace AMDGPU {

using HazardClassMask = uint32_t;
namespace HazardClass {

// Coherent with CoExec instruction types
enum : HazardClassMask {
  None = 0,
  CTRL = 1u << 0,
  VALU = 1u << 1,
  TRANS = 1u << 2,
  SALU = 1u << 3,
  DS = 1u << 4,
  VMEM = 1u << 5,
  SMEM = 1u << 6,
  WMMA = 1u << 7,
  MFMA = 1u << 8,
  EXP = 1u << 9,
};
} // namespace HazardClass

enum class HazardOperand : uint8_t {
  None,
  Def,
  Src0,
  Src1,
  Src2,
  Idx,
  Vaddr,
  AnySrc,
  AnyUse,
};

enum class ConsumerHint : uint8_t {
  OneDirectional,
  Symmetric,
};

struct HazardContext {
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  const GCNSubtarget *ST;
  const TargetSchedModel *SchedModel;
};

struct ClassMatch {
  HazardClassMask AnyOf = 0;
  bool matches(HazardClassMask Mask) const { return !AnyOf || (Mask & AnyOf); }
};

// To enable or disable the whole rule.
using RulePredicate = bool (*)(const HazardContext &);

// To gate the rule for producer or consumer.
using InstPredicate = bool (*)(const MachineInstr &, const HazardContext &);

// This helps advancing a window for the WAW/WAR case where a RAW sits between
// the producer and the consumer.
using AdvanceForRawWindowFn = unsigned (*)(const MachineInstr &Producer,
                                           HazardClassMask ReaderClass,
                                           const HazardContext &Ctx);

struct HazardSide {
  ClassMatch Match;
  HazardOperand Op = HazardOperand::None;
  InstPredicate Predicate = nullptr;
};

// Length of a hazard window in wait states.
// Precedence: OptWindowLength --> Fn --> WindowLength.
struct WindowSpec {
  unsigned WindowLength = 0;
  const cl::opt<unsigned> *OptWindowLength = nullptr;
  unsigned (*Fn)(const MachineInstr &Producer,
                 const HazardContext &Ctx) = nullptr;
};

struct ConsumerTarget {
  HazardSide Side;
  WindowSpec Window;
  HazardClassMask CounterMask = HazardClass::None;
  ConsumerHint Hint = ConsumerHint::OneDirectional;
};

struct HazardAntiHintRule {
  RulePredicate Predicate = nullptr;
  HazardSide Producer;
  SmallVector<ConsumerTarget, 3> Consumers;
  AdvanceForRawWindowFn AdvanceForRawWindow = nullptr;
};

void applyAntiHintRules(MachineFunction &MF, const HazardContext &Ctx);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNPRERAANTIHINTS_H
