//===- llvm/MC/MCInstrAnalysis.h - InstrDesc target hooks -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MCInstrAnalysis class which the MCTargetDescs can
// derive from to give additional information to MC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTRANALYSIS_H
#define LLVM_MC_MCINSTRANALYSIS_H

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include <cstdint>

namespace llvm {

class MCRegisterInfo;

class MCInstrAnalysis {
protected:
  friend class Target;

  const MCInstrInfo *Info;

public:
  MCInstrAnalysis(const MCInstrInfo *Info) : Info(Info) {}
  virtual ~MCInstrAnalysis() = default;

  virtual bool isBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isBranch();
  }

  virtual bool isConditionalBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isConditionalBranch();
  }

  virtual bool isUnconditionalBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isUnconditionalBranch();
  }

  virtual bool isIndirectBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isIndirectBranch();
  }

  virtual bool isCall(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isCall();
  }

  virtual bool isReturn(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isReturn();
  }

  virtual bool isTerminator(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isTerminator();
  }

  /// Returns true if at least one of the register writes performed by
  /// \param Inst implicitly clears the upper portion of all super-registers.
  ///
  /// Example: on X86-64, a write to EAX implicitly clears the upper half of
  /// RAX. Also (still on x86) an XMM write perfomed by an AVX 128-bit
  /// instruction implicitly clears the upper portion of the correspondent
  /// YMM register.
  ///
  /// This method also updates an APInt which is used as mask of register
  /// writes. There is one bit for every explicit/implicit write performed by
  /// the instruction. If a write implicitly clears its super-registers, then
  /// the corresponding bit is set (vic. the corresponding bit is cleared).
  ///
  /// The first bits in the APint are related to explicit writes. The remaining
  /// bits are related to implicit writes. The sequence of writes follows the
  /// machine operand sequence. For implicit writes, the sequence is defined by
  /// the MCInstrDesc.
  ///
  /// The assumption is that the bit-width of the APInt is correctly set by
  /// the caller. The default implementation conservatively assumes that none of
  /// the writes clears the upper portion of a super-register.
  virtual bool clearsSuperRegisters(const MCRegisterInfo &MRI,
                                    const MCInst &Inst,
                                    APInt &Writes) const;

  /// Returns true if \param Inst is a dependency breaking instruction for the
  /// given subtarget.
  ///
  /// The value computed by a dependency breaking instruction is not dependent
  /// on the inputs. An example of dependency breaking instruction on X86 is
  /// `XOR %eax, %eax`.
  /// TODO: In future, we could implement an alternative approach where this
  /// method returns `true` if the input instruction is not dependent on
  /// some/all of its input operands. An APInt mask could then be used to
  /// identify independent operands.
  virtual bool isDependencyBreaking(const MCSubtargetInfo &STI,
                                    const MCInst &Inst) const;

  /// Given a branch instruction try to get the address the branch
  /// targets. Return true on success, and the address in Target.
  virtual bool
  evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                 uint64_t &Target) const;
};

} // end namespace llvm

#endif // LLVM_MC_MCINSTRANALYSIS_H
