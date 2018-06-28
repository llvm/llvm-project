//===---------------------- Stage.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a stage.
/// A chain of stages compose an instruction pipeline.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_STAGE_H
#define LLVM_TOOLS_LLVM_MCA_STAGE_H

#include "HWEventListener.h"
#include <set>

namespace mca {

class InstRef;

class Stage {
  Stage(const Stage &Other) = delete;
  Stage &operator=(const Stage &Other) = delete;
  std::set<HWEventListener *> Listeners;

protected:
  const std::set<HWEventListener *> &getListeners() const { return Listeners; }

public:
  Stage();
  virtual ~Stage() = default;

  /// Called prior to preExecute to ensure that the stage has items that it
  /// is to process.  For example, a FetchStage might have more instructions
  /// that need to be processed, or a RCU might have items that have yet to
  /// retire.
  virtual bool hasWorkToComplete() const = 0;

  /// Called as a setup phase to prepare for the main stage execution.
  virtual void preExecute(const InstRef &IR) {}

  /// Called as a cleanup and finalization phase after main stage execution.
  virtual void postExecute(const InstRef &IR) {}

  /// The primary action that this stage performs.
  /// Returning false prevents successor stages from having their 'execute'
  /// routine called.
  virtual bool execute(InstRef &IR) = 0;

  /// Add a listener to receive callbacks during the execution of this stage.
  void addListener(HWEventListener *Listener);

  virtual void notifyInstructionEvent(const HWInstructionEvent &Event);
};

} // namespace mca
#endif // LLVM_TOOLS_LLVM_MCA_STAGE_H
