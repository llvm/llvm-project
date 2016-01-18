//===-- DOTGraphTraitsPass.h - Print/View dotty graphs-----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Templates to create dotty viewer and printer passes for GraphTraits graphs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOTGRAPHTRAITSPASS_H
#define LLVM_ANALYSIS_DOTGRAPHTRAITSPASS_H

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {

/// \brief Default traits class for extracting a graph from an analysis pass.
///
/// This assumes that 'GraphT' is 'AnalysisT *' and so just passes it through.
template <typename AnalysisT, typename GraphT = AnalysisT *>
struct DefaultAnalysisGraphTraits {
  static GraphT getGraph(AnalysisT *A) { return A; }
};

template <
    typename AnalysisT, bool IsSimple, typename GraphT = AnalysisT *,
    typename AnalysisGraphTraitsT = DefaultAnalysisGraphTraits<AnalysisT> >
class DOTGraphTraitsViewer : public FunctionPass {
public:
  DOTGraphTraitsViewer(StringRef GraphName, char &ID)
      : FunctionPass(ID), Name(GraphName) {}

  /// @brief Return true if this function should be processed.
  ///
  /// An implementation of this class my override this function to indicate that
  /// only certain functions should be viewed.
  virtual bool processFunction(Function &F) {
    return true;
  }

  bool runOnFunction(Function &F) override {
    if (!processFunction(F))
      return false;

    GraphT Graph = AnalysisGraphTraitsT::getGraph(&getAnalysis<AnalysisT>());
    std::string GraphName = DOTGraphTraits<GraphT>::getGraphName(Graph);
    std::string Title = GraphName + " for '" + F.getName().str() + "' function";

    ViewGraph(Graph, Name, IsSimple, Title);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<AnalysisT>();
  }

private:
  std::string Name;
};

template <
    typename AnalysisT, bool IsSimple, typename GraphT = AnalysisT *,
    typename AnalysisGraphTraitsT = DefaultAnalysisGraphTraits<AnalysisT> >
class DOTGraphTraitsPrinter : public FunctionPass {
public:
  DOTGraphTraitsPrinter(StringRef GraphName, char &ID)
      : FunctionPass(ID), Name(GraphName) {}

  /// @brief Return true if this function should be processed.
  ///
  /// An implementation of this class my override this function to indicate that
  /// only certain functions should be printed.
  virtual bool processFunction(Function &F) {
    return true;
  }

  bool runOnFunction(Function &F) override {
    if (!processFunction(F))
      return false;

    GraphT Graph = AnalysisGraphTraitsT::getGraph(&getAnalysis<AnalysisT>());
    std::string Filename = Name + "." + F.getName().str() + ".dot";
    std::error_code EC;

    errs() << "Writing '" << Filename << "'...";

    raw_fd_ostream File(Filename, EC, sys::fs::F_Text);
    std::string GraphName = DOTGraphTraits<GraphT>::getGraphName(Graph);
    std::string Title = GraphName + " for '" + F.getName().str() + "' function";

    if (!EC)
      WriteGraph(File, Graph, IsSimple, Title);
    else
      errs() << "  error opening file for writing!";
    errs() << "\n";

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<AnalysisT>();
  }

private:
  std::string Name;
};

template <
    typename AnalysisT, bool IsSimple, typename GraphT = AnalysisT *,
    typename AnalysisGraphTraitsT = DefaultAnalysisGraphTraits<AnalysisT> >
class DOTGraphTraitsModuleViewer : public ModulePass {
public:
  DOTGraphTraitsModuleViewer(StringRef GraphName, char &ID)
      : ModulePass(ID), Name(GraphName) {}

  bool runOnModule(Module &M) override {
    GraphT Graph = AnalysisGraphTraitsT::getGraph(&getAnalysis<AnalysisT>());
    std::string Title = DOTGraphTraits<GraphT>::getGraphName(Graph);

    ViewGraph(Graph, Name, IsSimple, Title);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<AnalysisT>();
  }

private:
  std::string Name;
};

template <
    typename AnalysisT, bool IsSimple, typename GraphT = AnalysisT *,
    typename AnalysisGraphTraitsT = DefaultAnalysisGraphTraits<AnalysisT> >
class DOTGraphTraitsModulePrinter : public ModulePass {
public:
  DOTGraphTraitsModulePrinter(StringRef GraphName, char &ID)
      : ModulePass(ID), Name(GraphName) {}

  bool runOnModule(Module &M) override {
    GraphT Graph = AnalysisGraphTraitsT::getGraph(&getAnalysis<AnalysisT>());
    std::string Filename = Name + ".dot";
    std::error_code EC;

    errs() << "Writing '" << Filename << "'...";

    raw_fd_ostream File(Filename, EC, sys::fs::F_Text);
    std::string Title = DOTGraphTraits<GraphT>::getGraphName(Graph);

    if (!EC)
      WriteGraph(File, Graph, IsSimple, Title);
    else
      errs() << "  error opening file for writing!";
    errs() << "\n";

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<AnalysisT>();
  }

private:
  std::string Name;
};

} // end namespace llvm

#endif
