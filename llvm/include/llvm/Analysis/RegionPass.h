//===- RegionPass.h - RegionPass class ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegionPass class. All region based analysis,
// optimization and transformation passes are derived from RegionPass.
// This class is implemented following the some ideas of the LoopPass.h class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REGION_PASS_H
#define LLVM_REGION_PASS_H

#include "llvm/Analysis/RegionInfo.h"

#include "llvm/Pass.h"
#include "llvm/PassManagers.h"
#include "llvm/Function.h"

#include <deque>

namespace llvm {

class RGPassManager;
class Function;

//===----------------------------------------------------------------------===//
/// @brief A pass that runs on each Region in a function.
///
/// RegionPass is managed by RGPassManager.
class RegionPass : public Pass {
public:
  explicit RegionPass(char &pid) : Pass(PT_Region, pid) {}

  //===--------------------------------------------------------------------===//
  /// @name To be implemented by every RegionPass
  ///
  //@{
  /// @brief Run the pass on a specific Region
  ///
  /// Accessing regions not contained in the current region is not allowed.
  ///
  /// @param R The region this pass is run on.
  /// @param RGM The RegionPassManager that manages this Pass.
  ///
  /// @return True if the pass modifies this Region.
  virtual bool runOnRegion(Region *R, RGPassManager &RGM) = 0;

  /// @brief Get a pass to print the LLVM IR in the region.
  ///
  /// @param O      The ouput stream to print the Region.
  /// @param Banner The banner to separate different printed passes.
  ///
  /// @return The pass to print the LLVM IR in the region.
  Pass *createPrinterPass(raw_ostream &O, const std::string &Banner) const;

  virtual bool doInitialization(Region *R, RGPassManager &RGM) { return false; }
  virtual bool doFinalization() { return false; }
  //@}

  //===--------------------------------------------------------------------===//
  /// @name PassManager API
  ///
  //@{
  void preparePassManager(PMStack &PMS);

  virtual void assignPassManager(PMStack &PMS,
    PassManagerType PMT = PMT_RegionPassManager);

  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_RegionPassManager;
  }
  //@}
};

/// @brief The pass manager to schedule RegionPasses.
class RGPassManager : public FunctionPass, public PMDataManager {
  std::deque<Region*> RQ;
  bool skipThisRegion;
  bool redoThisRegion;
  RegionInfo *RI;
  Region *CurrentRegion;

public:
  static char ID;
  explicit RGPassManager();

  /// @brief Execute all of the passes scheduled for execution.
  ///
  /// @return True if any of the passes modifies the function.
  bool runOnFunction(Function &F);

  /// Pass Manager itself does not invalidate any analysis info.
  /// RGPassManager needs RegionInfo.
  void getAnalysisUsage(AnalysisUsage &Info) const;

  virtual const char *getPassName() const {
    return "Region Pass Manager";
  }

  virtual PMDataManager *getAsPMDataManager() { return this; }
  virtual Pass *getAsPass() { return this; }

  /// @brief Print passes managed by this manager.
  void dumpPassStructure(unsigned Offset);

  /// @brief Get passes contained by this manager.
  Pass *getContainedPass(unsigned N) {
    assert(N < PassVector.size() && "Pass number out of range!");
    Pass *FP = static_cast<Pass *>(PassVector[N]);
    return FP;
  }

  virtual PassManagerType getPassManagerType() const {
    return PMT_RegionPassManager;
  }
};

} // End llvm namespace

#endif
