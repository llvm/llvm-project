//===- llvm-prof.cpp - Read in and process llvmprof.out data files --------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tools is meant for use with the various LLVM profiling instrumentation
// passes.  It reads in the data file produced by executing an instrumented
// program, and outputs a nice report.
//
//===----------------------------------------------------------------------===//

#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>

using namespace llvm;

namespace {
  cl::opt<std::string>
  BitcodeFile(cl::Positional, cl::desc("<program bitcode file>"),
              cl::Required);

  cl::opt<std::string>
  ProfileDataFile(cl::Positional, cl::desc("<llvmprof.out file>"),
                  cl::Optional, cl::init("llvmprof.out"));

  cl::opt<bool>
  PrintAnnotatedLLVM("annotated-llvm",
                     cl::desc("Print LLVM code with frequency annotations"));
  cl::alias PrintAnnotated2("A", cl::desc("Alias for --annotated-llvm"),
                            cl::aliasopt(PrintAnnotatedLLVM));
  cl::opt<bool>
  PrintAllCode("print-all-code",
               cl::desc("Print annotated code for the entire program"));
}

// PairSecondSort - A sorting predicate to sort by the second element of a pair.
template<class T>
struct PairSecondSortReverse
  : public std::binary_function<std::pair<T, unsigned>,
                                std::pair<T, unsigned>, bool> {
  bool operator()(const std::pair<T, unsigned> &LHS,
                  const std::pair<T, unsigned> &RHS) const {
    return LHS.second > RHS.second;
  }
};

namespace {
  class ProfileAnnotator : public AssemblyAnnotationWriter {
    std::map<const Function  *, unsigned> &FuncFreqs;
    std::map<const BasicBlock*, unsigned> &BlockFreqs;
    std::map<ProfileInfoLoader::Edge, unsigned> &EdgeFreqs;
  public:
    ProfileAnnotator(std::map<const Function  *, unsigned> &FF,
                     std::map<const BasicBlock*, unsigned> &BF,
                     std::map<ProfileInfoLoader::Edge, unsigned> &EF)
      : FuncFreqs(FF), BlockFreqs(BF), EdgeFreqs(EF) {}

    virtual void emitFunctionAnnot(const Function *F, raw_ostream &OS) {
      OS << ";;; %" << F->getName() << " called " << FuncFreqs[F]
         << " times.\n;;;\n";
    }
    virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                          raw_ostream &OS) {
      if (BlockFreqs.empty()) return;
      if (unsigned Count = BlockFreqs[BB])
        OS << "\t;;; Basic block executed " << Count << " times.\n";
      else
        OS << "\t;;; Never executed!\n";
    }

    virtual void emitBasicBlockEndAnnot(const BasicBlock *BB, raw_ostream &OS) {
      if (EdgeFreqs.empty()) return;

      // Figure out how many times each successor executed.
      std::vector<std::pair<const BasicBlock*, unsigned> > SuccCounts;
      const TerminatorInst *TI = BB->getTerminator();

      std::map<ProfileInfoLoader::Edge, unsigned>::iterator I =
        EdgeFreqs.lower_bound(std::make_pair(const_cast<BasicBlock*>(BB), 0U));
      for (; I != EdgeFreqs.end() && I->first.first == BB; ++I)
        if (I->second)
          SuccCounts.push_back(std::make_pair(TI->getSuccessor(I->first.second),
                                              I->second));
      if (!SuccCounts.empty()) {
        OS << "\t;;; Out-edge counts:";
        for (unsigned i = 0, e = SuccCounts.size(); i != e; ++i)
          OS << " [" << SuccCounts[i].second << " -> "
             << SuccCounts[i].first->getName() << "]";
        OS << "\n";
      }
    }
  };
}


int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  try {
    cl::ParseCommandLineOptions(argc, argv, "llvm profile dump decoder\n");
    sys::PrintStackTraceOnErrorSignal();

    // Read in the bitcode file...
    std::string ErrorMessage;
    Module *M = 0;
    if (MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(BitcodeFile,
                                                            &ErrorMessage)) {
      M = ParseBitcodeFile(Buffer, &ErrorMessage);
      delete Buffer;
    }
    if (M == 0) {
      std::cerr << argv[0] << ": " << BitcodeFile << ": " 
        << ErrorMessage << "\n";
      return 1;
    }

    // Read the profiling information
    ProfileInfoLoader PI(argv[0], ProfileDataFile, *M);

    std::map<const Function  *, unsigned> FuncFreqs;
    std::map<const BasicBlock*, unsigned> BlockFreqs;
    std::map<ProfileInfoLoader::Edge, unsigned> EdgeFreqs;

    // Output a report. Eventually, there will be multiple reports selectable on
    // the command line, for now, just keep things simple.

    // Emit the most frequent function table...
    std::vector<std::pair<Function*, unsigned> > FunctionCounts;
    PI.getFunctionCounts(FunctionCounts);
    FuncFreqs.insert(FunctionCounts.begin(), FunctionCounts.end());

    // Sort by the frequency, backwards.
    sort(FunctionCounts.begin(), FunctionCounts.end(),
              PairSecondSortReverse<Function*>());

    uint64_t TotalExecutions = 0;
    for (unsigned i = 0, e = FunctionCounts.size(); i != e; ++i)
      TotalExecutions += FunctionCounts[i].second;

    std::cout << "===" << std::string(73, '-') << "===\n"
              << "LLVM profiling output for execution";
    if (PI.getNumExecutions() != 1) std::cout << "s";
    std::cout << ":\n";

    for (unsigned i = 0, e = PI.getNumExecutions(); i != e; ++i) {
      std::cout << "  ";
      if (e != 1) std::cout << i+1 << ". ";
      std::cout << PI.getExecution(i) << "\n";
    }

    std::cout << "\n===" << std::string(73, '-') << "===\n";
    std::cout << "Function execution frequencies:\n\n";

    // Print out the function frequencies...
    std::cout << " ##   Frequency\n";
    for (unsigned i = 0, e = FunctionCounts.size(); i != e; ++i) {
      if (FunctionCounts[i].second == 0) {
        std::cout << "\n  NOTE: " << e-i << " function" <<
               (e-i-1 ? "s were" : " was") << " never executed!\n";
        break;
      }

      std::cout << std::setw(3) << i+1 << ". " 
        << std::setw(5) << FunctionCounts[i].second << "/"
        << TotalExecutions << " "
        << FunctionCounts[i].first->getName().c_str() << "\n";
    }

    std::set<Function*> FunctionsToPrint;

    // If we have block count information, print out the LLVM module with
    // frequency annotations.
    if (PI.hasAccurateBlockCounts()) {
      std::vector<std::pair<BasicBlock*, unsigned> > Counts;
      PI.getBlockCounts(Counts);

      TotalExecutions = 0;
      for (unsigned i = 0, e = Counts.size(); i != e; ++i)
        TotalExecutions += Counts[i].second;

      // Sort by the frequency, backwards.
      sort(Counts.begin(), Counts.end(),
                PairSecondSortReverse<BasicBlock*>());

      std::cout << "\n===" << std::string(73, '-') << "===\n";
      std::cout << "Top 20 most frequently executed basic blocks:\n\n";

      // Print out the function frequencies...
      std::cout <<" ##      %% \tFrequency\n";
      unsigned BlocksToPrint = Counts.size();
      if (BlocksToPrint > 20) BlocksToPrint = 20;
      for (unsigned i = 0; i != BlocksToPrint; ++i) {
        if (Counts[i].second == 0) break;
        Function *F = Counts[i].first->getParent();
        std::cout << std::setw(3) << i+1 << ". " 
          << std::setw(5) << std::setprecision(2) 
          << Counts[i].second/(double)TotalExecutions*100 << "% "
          << std::setw(5) << Counts[i].second << "/"
          << TotalExecutions << "\t"
          << F->getName().c_str() << "() - "
          << Counts[i].first->getName().c_str() << "\n";
        FunctionsToPrint.insert(F);
      }

      BlockFreqs.insert(Counts.begin(), Counts.end());
    }

    if (PI.hasAccurateEdgeCounts()) {
      std::vector<std::pair<ProfileInfoLoader::Edge, unsigned> > Counts;
      PI.getEdgeCounts(Counts);
      EdgeFreqs.insert(Counts.begin(), Counts.end());
    }

    if (PrintAnnotatedLLVM || PrintAllCode) {
      std::cout << "\n===" << std::string(73, '-') << "===\n";
      std::cout << "Annotated LLVM code for the module:\n\n";

      ProfileAnnotator PA(FuncFreqs, BlockFreqs, EdgeFreqs);

      if (FunctionsToPrint.empty() || PrintAllCode)
        M->print(std::cout, &PA);
      else
        // Print just a subset of the functions.
        for (std::set<Function*>::iterator I = FunctionsToPrint.begin(),
               E = FunctionsToPrint.end(); I != E; ++I)
          (*I)->print(std::cout, &PA);
    }

    return 0;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}
