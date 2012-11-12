//==-- llvm/Target/TargetSubtargetInfo.h - Target Information ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSUBTARGETINFO_H
#define LLVM_TARGET_TARGETSUBTARGETINFO_H

#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class MachineInstr;
class SDep;
class SUnit;
class TargetRegisterClass;
class TargetSchedModel;
template <typename T> class SmallVectorImpl;

//===----------------------------------------------------------------------===//
///
/// TargetSubtargetInfo - Generic base class for all target subtargets.  All
/// Target-specific options that control code generation and printing should
/// be exposed through a TargetSubtargetInfo-derived class.
///
class TargetSubtargetInfo : public MCSubtargetInfo {
  TargetSubtargetInfo(const TargetSubtargetInfo&) LLVM_DELETED_FUNCTION;
  void operator=(const TargetSubtargetInfo&) LLVM_DELETED_FUNCTION;
protected: // Can only create subclasses...
  TargetSubtargetInfo();
public:
  // AntiDepBreakMode - Type of anti-dependence breaking that should
  // be performed before post-RA scheduling.
  typedef enum { ANTIDEP_NONE, ANTIDEP_CRITICAL, ANTIDEP_ALL } AntiDepBreakMode;
  typedef SmallVectorImpl<const TargetRegisterClass*> RegClassVector;

  virtual ~TargetSubtargetInfo();

  /// Resolve a SchedClass at runtime, where SchedClass identifies an
  /// MCSchedClassDesc with the isVariant property. This may return the ID of
  /// another variant SchedClass, but repeated invocation must quickly terminate
  /// in a nonvariant SchedClass.
  virtual unsigned resolveSchedClass(unsigned SchedClass, const MachineInstr *MI,
                                     const TargetSchedModel* SchedModel) const {
    return 0;
  }

  // enablePostRAScheduler - If the target can benefit from post-regalloc
  // scheduling and the specified optimization level meets the requirement
  // return true to enable post-register-allocation scheduling. In
  // CriticalPathRCs return any register classes that should only be broken
  // if on the critical path.
  virtual bool enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                                     AntiDepBreakMode& Mode,
                                     RegClassVector& CriticalPathRCs) const;
  // adjustSchedDependency - Perform target specific adjustments to
  // the latency of a schedule dependency.
  virtual void adjustSchedDependency(SUnit *def, SUnit *use,
                                     SDep& dep) const { }
};

} // End llvm namespace

#endif
