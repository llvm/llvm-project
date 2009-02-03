//===- llvm/PassAnalysisSupport.h - Analysis Pass Support code --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines stuff that is used to define and "use" Analysis Passes.
// This file is automatically #included by Pass.h, so:
//
//           NO .CPP FILES SHOULD INCLUDE THIS FILE DIRECTLY
//
// Instead, #include Pass.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_ANALYSIS_SUPPORT_H
#define LLVM_PASS_ANALYSIS_SUPPORT_H

#include <vector>
#include "llvm/ADT/SmallVector.h"

namespace llvm {

// No need to include Pass.h, we are being included by it!

//===----------------------------------------------------------------------===//
// AnalysisUsage - Represent the analysis usage information of a pass.  This
// tracks analyses that the pass REQUIRES (must be available when the pass
// runs), REQUIRES TRANSITIVE (must be available throughout the lifetime of the
// pass), and analyses that the pass PRESERVES (the pass does not invalidate the
// results of these analyses).  This information is provided by a pass to the
// Pass infrastructure through the getAnalysisUsage virtual function.
//
class AnalysisUsage {
public:
  typedef SmallVector<AnalysisID, 32> VectorType;

private:
  // Sets of analyses required and preserved by a pass
  VectorType Required, RequiredTransitive, Preserved;
  bool PreservesAll;

public:
  AnalysisUsage() : PreservesAll(false) {}

  // addRequired - Add the specified ID to the required set of the usage info
  // for a pass.
  //
  AnalysisUsage &addRequiredID(AnalysisID ID) {
    assert(ID && "Pass class not registered!");
    Required.push_back(ID);
    return *this;
  }
  template<class PassClass>
  AnalysisUsage &addRequired() {
    return addRequiredID(Pass::getClassPassInfo<PassClass>());
  }

  AnalysisUsage &addRequiredTransitiveID(AnalysisID ID) {
    assert(ID && "Pass class not registered!");
    Required.push_back(ID);
    RequiredTransitive.push_back(ID);
    return *this;
  }
  template<class PassClass>
  AnalysisUsage &addRequiredTransitive() {
    AnalysisID ID = Pass::getClassPassInfo<PassClass>();
    return addRequiredTransitiveID(ID);
  }

  // addPreserved - Add the specified ID to the set of analyses preserved by
  // this pass
  //
  AnalysisUsage &addPreservedID(AnalysisID ID) {
    Preserved.push_back(ID);
    return *this;
  }

  template<class PassClass>
  AnalysisUsage &addPreserved() {
    assert(Pass::getClassPassInfo<PassClass>() && "Pass class not registered!");
    Preserved.push_back(Pass::getClassPassInfo<PassClass>());
    return *this;
  }

  // setPreservesAll - Set by analyses that do not transform their input at all
  void setPreservesAll() { PreservesAll = true; }
  bool getPreservesAll() const { return PreservesAll; }

  /// setPreservesCFG - This function should be called by the pass, iff they do
  /// not:
  ///
  ///  1. Add or remove basic blocks from the function
  ///  2. Modify terminator instructions in any way.
  ///
  /// This function annotates the AnalysisUsage info object to say that analyses
  /// that only depend on the CFG are preserved by this pass.
  ///
  void setPreservesCFG();

  const VectorType &getRequiredSet() const { return Required; }
  const VectorType &getRequiredTransitiveSet() const {
    return RequiredTransitive;
  }
  const VectorType &getPreservedSet() const { return Preserved; }
};

//===----------------------------------------------------------------------===//
// AnalysisResolver - Simple interface used by Pass objects to pull all
// analysis information out of pass manager that is responsible to manage
// the pass.
//
class PMDataManager;
class AnalysisResolver {
private:
  AnalysisResolver();  // DO NOT IMPLEMENT

public:
  explicit AnalysisResolver(PMDataManager &P) : PM(P) { }
  
  inline PMDataManager &getPMDataManager() { return PM; }

  // Find pass that is implementing PI.
  Pass *findImplPass(const PassInfo *PI) {
    Pass *ResultPass = 0;
    for (unsigned i = 0; i < AnalysisImpls.size() ; ++i) {
      if (AnalysisImpls[i].first == PI) {
        ResultPass = AnalysisImpls[i].second;
        break;
      }
    }
    return ResultPass;
  }

  // Find pass that is implementing PI. Initialize pass for Function F.
  Pass *findImplPass(Pass *P, const PassInfo *PI, Function &F);

  void addAnalysisImplsPair(const PassInfo *PI, Pass *P) {
    std::pair<const PassInfo*, Pass*> pir = std::make_pair(PI,P);
    AnalysisImpls.push_back(pir);
  }

  // getAnalysisIfAvailable - Return analysis result or null if it doesn't exist
  Pass *getAnalysisIfAvailable(AnalysisID ID, bool Direction) const;

  // AnalysisImpls - This keeps track of which passes implements the interfaces
  // that are required by the current pass (to implement getAnalysis()).
  // NOTE : Remove AnalysisImpls from class Pass, when AnalysisResolver
  // replaces AnalysisResolver
  std::vector<std::pair<const PassInfo*, Pass*> > AnalysisImpls;

private:
  // PassManager that is used to resolve analysis info
  PMDataManager &PM;
};

/// getAnalysisIfAvailable<AnalysisType>() - Subclasses use this function to
/// get analysis information that might be around, for example to update it.
/// This is different than getAnalysis in that it can fail (if the analysis
/// results haven't been computed), so should only be used if you can handle
/// the case when the analysis is not available.  This method is often used by
/// transformation APIs to update analysis results for a pass automatically as
/// the transform is performed.
///
template<typename AnalysisType>
AnalysisType *Pass::getAnalysisIfAvailable() const {
  assert(Resolver && "Pass not resident in a PassManager object!");

  const PassInfo *PI = getClassPassInfo<AnalysisType>();
  if (PI == 0) return 0;
  return dynamic_cast<AnalysisType*>
    (Resolver->getAnalysisIfAvailable(PI, true));
}

/// getAnalysis<AnalysisType>() - This function is used by subclasses to get
/// to the analysis information that they claim to use by overriding the
/// getAnalysisUsage function.
///
template<typename AnalysisType>
AnalysisType &Pass::getAnalysis() const {
  assert(Resolver &&"Pass has not been inserted into a PassManager object!");

  return getAnalysisID<AnalysisType>(getClassPassInfo<AnalysisType>());
}

template<typename AnalysisType>
AnalysisType &Pass::getAnalysisID(const PassInfo *PI) const {
  assert(PI && "getAnalysis for unregistered pass!");
  assert(Resolver&&"Pass has not been inserted into a PassManager object!");
  // PI *must* appear in AnalysisImpls.  Because the number of passes used
  // should be a small number, we just do a linear search over a (dense)
  // vector.
  Pass *ResultPass = Resolver->findImplPass(PI);
  assert (ResultPass && 
          "getAnalysis*() called on an analysis that was not "
          "'required' by pass!");

  // Because the AnalysisType may not be a subclass of pass (for
  // AnalysisGroups), we must use dynamic_cast here to potentially adjust the
  // return pointer (because the class may multiply inherit, once from pass,
  // once from AnalysisType).
  //
  AnalysisType *Result = dynamic_cast<AnalysisType*>(ResultPass);
  assert(Result && "Pass does not implement interface required!");
  return *Result;
}

/// getAnalysis<AnalysisType>() - This function is used by subclasses to get
/// to the analysis information that they claim to use by overriding the
/// getAnalysisUsage function.
///
template<typename AnalysisType>
AnalysisType &Pass::getAnalysis(Function &F) {
  assert(Resolver &&"Pass has not been inserted into a PassManager object!");

  return getAnalysisID<AnalysisType>(getClassPassInfo<AnalysisType>(), F);
}

template<typename AnalysisType>
AnalysisType &Pass::getAnalysisID(const PassInfo *PI, Function &F) {
  assert(PI && "getAnalysis for unregistered pass!");
   assert(Resolver&&"Pass has not been inserted into a PassManager object!");
   // PI *must* appear in AnalysisImpls.  Because the number of passes used
   // should be a small number, we just do a linear search over a (dense)
   // vector.
   Pass *ResultPass = Resolver->findImplPass(this, PI, F);
   assert (ResultPass && 
           "getAnalysis*() called on an analysis that was not "
           "'required' by pass!");
 
   // Because the AnalysisType may not be a subclass of pass (for
   // AnalysisGroups), we must use dynamic_cast here to potentially adjust the
   // return pointer (because the class may multiply inherit, once from pass,
   // once from AnalysisType).
   //
   AnalysisType *Result = dynamic_cast<AnalysisType*>(ResultPass);
   assert(Result && "Pass does not implement interface required!");
   return *Result;
}

} // End llvm namespace

#endif
