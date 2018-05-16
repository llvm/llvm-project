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

#include <set>

namespace mca {

class HWEventListener;
class InstRef;

class Stage {
  std::set<HWEventListener *> Listeners;
  Stage(const Stage &Other) = delete;
  Stage &operator=(const Stage &Other) = delete;

public:
  Stage();
  virtual ~Stage() = default;

  /// Called prior to preExecute to ensure that the stage can operate.
  /// TODO: Remove this logic once backend::run and backend::runCycle become
  /// one routine.
  virtual bool isReady() const { return true; }

  /// Called as a setup phase to prepare for the main stage execution.
  virtual void preExecute(const InstRef &IR) {}

  /// Called as a cleanup and finalization phase after main stage execution.
  virtual void postExecute(const InstRef &IR) {}

  /// The primary action that this stage performs.
  virtual bool execute(InstRef &IR) = 0;

  /// Add a listener to receive callbaks during the execution of this stage.
  void addListener(HWEventListener *Listener);
};

} // namespace mca
#endif // LLVM_TOOLS_LLVM_MCA_STAGE_H
