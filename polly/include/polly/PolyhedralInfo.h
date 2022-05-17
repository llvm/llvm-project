//===- polly/PolyhedralInfo.h - PolyhedralInfo class definition -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the declaration of the PolyhedralInfo class, which will
/// provide an interface to expose polyhedral analysis information of Polly.
///
/// This is work in progress. We will add more API's as and when deemed
/// required.
//===----------------------------------------------------------------------===///

#ifndef POLLY_POLYHEDRAL_INFO_H
#define POLLY_POLYHEDRAL_INFO_H

#include "llvm/Pass.h"
#include "isl/aff_type.h"
#include "isl/ctx.h"
#include "isl/union_map_type.h"

namespace llvm {
class Loop;
} // namespace llvm

namespace polly {

class Scop;
class ScopInfo;
class DependenceInfoWrapperPass;

class PolyhedralInfo final : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  /// Construct a new PolyhedralInfo pass.
  PolyhedralInfo() : FunctionPass(ID) {}
  ~PolyhedralInfo() {}

  /// Check if a given loop is parallel.
  ///
  /// @param L The loop.
  ///
  /// @return  Returns true, if loop is parallel false otherwise.
  bool isParallel(llvm::Loop *L) const;

  /// Return the SCoP containing the @p L loop.
  ///
  /// @param L The loop.
  ///
  /// @return  Returns the SCoP containing the given loop.
  ///          Returns null if the loop is not contained in any SCoP.
  const Scop *getScopContainingLoop(llvm::Loop *L) const;

  /// Computes the partial schedule for the given @p L loop.
  ///
  /// @param S The SCoP containing the given loop
  /// @param L The loop.
  ///
  /// @return  Returns the partial schedule for the given loop
  __isl_give isl_union_map *getScheduleForLoop(const Scop *S,
                                               llvm::Loop *L) const;

  /// Get the SCoP and dependence analysis information for @p F.
  bool runOnFunction(llvm::Function &F) override;

  /// Release the internal memory.
  void releaseMemory() override {}

  /// Print to @p OS if each dimension of a loop nest is parallel or not.
  void print(llvm::raw_ostream &OS,
             const llvm::Module *M = nullptr) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

private:
  /// Check if a given loop is parallel or vectorizable.
  ///
  /// @param L             The loop.
  /// @param MinDepDistPtr If not nullptr, the minimal dependence distance will
  ///                      be returned at the address of that pointer
  ///
  /// @return  Returns true if loop is parallel or vectorizable, false
  ///          otherwise.
  bool checkParallel(llvm::Loop *L,
                     __isl_give isl_pw_aff **MinDepDistPtr = nullptr) const;

  ScopInfo *SI;
  DependenceInfoWrapperPass *DI;
};

llvm::Pass *createPolyhedralInfoPrinterLegacyPass(llvm::raw_ostream &OS);
} // end namespace polly

namespace llvm {
class PassRegistry;
void initializePolyhedralInfoPass(llvm::PassRegistry &);
void initializePolyhedralInfoPrinterLegacyPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif
