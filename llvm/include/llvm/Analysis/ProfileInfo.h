//===- llvm/Analysis/ProfileInfo.h - Profile Info Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the generic ProfileInfo interface, which is used as the
// common interface used by all clients of profiling information, and
// implemented either by making static guestimations, or by actually reading in
// profiling information gathered by running the program.
//
// Note that to be useful, all profile-based optimizations should preserve
// ProfileInfo, which requires that they notify it when changes to the CFG are
// made. (This is not implemented yet.)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILEINFO_H
#define LLVM_ANALYSIS_PROFILEINFO_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <string>
#include <map>
#include <set>

namespace llvm {
  class Pass;
  class raw_ostream;

  class BasicBlock;
  class Function;
  class MachineBasicBlock;
  class MachineFunction;

  // Helper for dumping edges to dbgs().
  raw_ostream& operator<<(raw_ostream &O, std::pair<const BasicBlock *, const BasicBlock *> E);
  raw_ostream& operator<<(raw_ostream &O, std::pair<const MachineBasicBlock *, const MachineBasicBlock *> E);

  raw_ostream& operator<<(raw_ostream &O, const BasicBlock *BB);
  raw_ostream& operator<<(raw_ostream &O, const MachineBasicBlock *MBB);

  raw_ostream& operator<<(raw_ostream &O, const Function *F);
  raw_ostream& operator<<(raw_ostream &O, const MachineFunction *MF);

  /// ProfileInfo Class - This class holds and maintains profiling
  /// information for some unit of code.
  template<class FType, class BType>
  class ProfileInfoT {
  public:
    // Types for handling profiling information.
    typedef std::pair<const BType*, const BType*> Edge;
    typedef std::pair<Edge, double> EdgeWeight;
    typedef std::map<Edge, double> EdgeWeights;
    typedef std::map<const BType*, double> BlockCounts;
    typedef std::map<const BType*, const BType*> Path;

  protected:
    // EdgeInformation - Count the number of times a transition between two
    // blocks is executed. As a special case, we also hold an edge from the
    // null BasicBlock to the entry block to indicate how many times the
    // function was entered.
    std::map<const FType*, EdgeWeights> EdgeInformation;

    // BlockInformation - Count the number of times a block is executed.
    std::map<const FType*, BlockCounts> BlockInformation;

    // FunctionInformation - Count the number of times a function is executed.
    std::map<const FType*, double> FunctionInformation;

    ProfileInfoT<MachineFunction, MachineBasicBlock> *MachineProfile;
  public:
    static char ID; // Class identification, replacement for typeinfo
    ProfileInfoT();
    ~ProfileInfoT();  // We want to be subclassed

    // MissingValue - The value that is returned for execution counts in case
    // no value is available.
    static const double MissingValue;

    // getFunction() - Returns the Function for an Edge, checking for validity.
    static const FType* getFunction(Edge e) {
      if (e.first)
        return e.first->getParent();
      if (e.second)
        return e.second->getParent();
      llvm_unreachable("Invalid ProfileInfo::Edge");
    }

    // getEdge() - Creates an Edge from two BasicBlocks.
    static Edge getEdge(const BType *Src, const BType *Dest) {
      return std::make_pair(Src, Dest);
    }

    //===------------------------------------------------------------------===//
    /// Profile Information Queries
    ///
    double getExecutionCount(const FType *F);

    double getExecutionCount(const BType *BB);

    void setExecutionCount(const BType *BB, double w);

    void addExecutionCount(const BType *BB, double w);

    double getEdgeWeight(Edge e) const {
      typename std::map<const FType*, EdgeWeights>::const_iterator J =
        EdgeInformation.find(getFunction(e));
      if (J == EdgeInformation.end()) return MissingValue;

      typename EdgeWeights::const_iterator I = J->second.find(e);
      if (I == J->second.end()) return MissingValue;

      return I->second;
    }

    void setEdgeWeight(Edge e, double w) {
      DEBUG_WITH_TYPE("profile-info",
            dbgs() << "Creating Edge " << e
                   << " (weight: " << format("%.20g",w) << ")\n");
      EdgeInformation[getFunction(e)][e] = w;
    }

    void addEdgeWeight(Edge e, double w);

    EdgeWeights &getEdgeWeights (const FType *F) {
      return EdgeInformation[F];
    }

    //===------------------------------------------------------------------===//
    /// Analysis Update Methods
    ///
    void removeBlock(const BType *BB);

    void removeEdge(Edge e);

    void replaceEdge(const Edge &, const Edge &);

    enum GetPathMode {
      GetPathToExit = 1,
      GetPathToValue = 2,
      GetPathToDest = 4,
      GetPathWithNewEdges = 8
    };

    const BType *GetPath(const BType *Src, const BType *Dest,
                              Path &P, unsigned Mode);

    void divertFlow(const Edge &, const Edge &);

    void splitEdge(const BType *FirstBB, const BType *SecondBB,
                   const BType *NewBB, bool MergeIdenticalEdges = false);

    void splitBlock(const BType *Old, const BType* New);

    void splitBlock(const BType *BB, const BType* NewBB,
                    BType *const *Preds, unsigned NumPreds);

    void replaceAllUses(const BType *RmBB, const BType *DestBB);

    void transfer(const FType *Old, const FType *New);

    void repair(const FType *F);

    void dump(FType *F = 0, bool real = true) {
      dbgs() << "**** This is ProfileInfo " << this << " speaking:\n";
      if (!real) {
        typename std::set<const FType*> Functions;

        dbgs() << "Functions: \n";
        if (F) {
          dbgs() << F << "@" << format("%p", F) << ": " << format("%.20g",getExecutionCount(F)) << "\n";
          Functions.insert(F);
        } else {
          for (typename std::map<const FType*, double>::iterator fi = FunctionInformation.begin(),
               fe = FunctionInformation.end(); fi != fe; ++fi) {
            dbgs() << fi->first << "@" << format("%p",fi->first) << ": " << format("%.20g",fi->second) << "\n";
            Functions.insert(fi->first);
          }
        }

        for (typename std::set<const FType*>::iterator FI = Functions.begin(), FE = Functions.end();
             FI != FE; ++FI) {
          const FType *F = *FI;
          typename std::map<const FType*, BlockCounts>::iterator bwi = BlockInformation.find(F);
          dbgs() << "BasicBlocks for Function " << F << ":\n";
          for (typename BlockCounts::const_iterator bi = bwi->second.begin(), be = bwi->second.end(); bi != be; ++bi) {
            dbgs() << bi->first << "@" << format("%p", bi->first) << ": " << format("%.20g",bi->second) << "\n";
          }
        }

        for (typename std::set<const FType*>::iterator FI = Functions.begin(), FE = Functions.end();
             FI != FE; ++FI) {
          typename std::map<const FType*, EdgeWeights>::iterator ei = EdgeInformation.find(*FI);
          dbgs() << "Edges for Function " << ei->first << ":\n";
          for (typename EdgeWeights::iterator ewi = ei->second.begin(), ewe = ei->second.end(); 
               ewi != ewe; ++ewi) {
            dbgs() << ewi->first << ": " << format("%.20g",ewi->second) << "\n";
          }
        }
      } else {
        assert(F && "No function given, this is not supported!");
        dbgs() << "Functions: \n";
        dbgs() << F << "@" << format("%p", F) << ": " << format("%.20g",getExecutionCount(F)) << "\n";

        dbgs() << "BasicBlocks for Function " << F << ":\n";
        for (typename FType::const_iterator BI = F->begin(), BE = F->end();
             BI != BE; ++BI) {
          const BType *BB = &(*BI);
          dbgs() << BB << "@" << format("%p", BB) << ": " << format("%.20g",getExecutionCount(BB)) << "\n";
        }
      }
      dbgs() << "**** ProfileInfo " << this << ", over and out.\n";
    }

    bool CalculateMissingEdge(const BType *BB, Edge &removed, bool assumeEmptyExit = false);

    bool EstimateMissingEdges(const BType *BB);

    ProfileInfoT<MachineFunction, MachineBasicBlock> *MI() {
      if (MachineProfile == 0)
        MachineProfile = new ProfileInfoT<MachineFunction, MachineBasicBlock>();
      return MachineProfile;
    }

    bool hasMI() const {
      return (MachineProfile != 0);
    }
  };

  typedef ProfileInfoT<Function, BasicBlock> ProfileInfo;
  typedef ProfileInfoT<MachineFunction, MachineBasicBlock> MachineProfileInfo;

  /// createProfileLoaderPass - This function returns a Pass that loads the
  /// profiling information for the module from the specified filename, making
  /// it available to the optimizers.
  Pass *createProfileLoaderPass(const std::string &Filename);

} // End llvm namespace

#endif
