#define DEBUG_TYPE "bsize-invariance-analysis"

#include "llvm/Transforms/InterprocBSIAnalysisPass.h"

#include <cxxabi.h>

// Print only entrypoints into the callgraph.
#define ENTRYPOINTS_ONLY

using namespace llvm;

inline std::string demangle(const char* name) 
{
  int status = -1; 

  std::unique_ptr<char, void(*)(void*)> res {
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free };
  return (status == 0) ? res.get() : std::string(name);
}

inline bool isEntryPoint(const CallGraph& CG, const CallGraphNode* node) {
  if (node->getNumReferences() > 1) return false;
  if (node->getNumReferences() == 0) return true;
  const auto* root = CG.getExternalCallingNode();
  // If the calling function for node is root, return true.
  for (unsigned i = 0; i < root->size(); i++) {
    if ((*root)[i] == node) return true;
  }
  return false;
}

bool InterproceduralBlockSizeInvarianceAnalysisPass::runOnModule(Module &M) {
  auto &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();

  // Generate topological order of visiting function nodes.
  // Sequence of functions.
  std::vector<Function *> functionList;
  // Set of global entrypoints into the call-graph.
  std::set<Function *> entrypoints;

  // Iterating over the SCCs.
  for (scc_iterator<CallGraph *> I = scc_begin(&CG), IE = scc_end(&CG);
                                                          I != IE; ++I) {
    const std::vector<CallGraphNode *> &SCCCGNs = *I;
    // Iterating over nodes in the SCC.
    for (std::vector<CallGraphNode *>::const_iterator CGNI = SCCCGNs.begin(),
					            CGNIE = SCCCGNs.end(); CGNI != CGNIE; ++CGNI) {
      Function *F = (*CGNI)->getFunction();
      // Insert function only if it is present and is not a declaration.
      if (!F || F->isDeclaration()) { continue; }
      functionList.insert(functionList.end(), F);
      LLVM_DEBUG(errs() << "Inserting function: " << F->getName()
            << " with " << (*CGNI)->getNumReferences() << " refs\n");
      // Add as entrypoint if it has at most single reference.
      if (isEntryPoint(CG, (*CGNI))) { entrypoints.insert(F); }
    }
  }

  // Function to return value map. 
  // Assuming the function takes in block-size independent values, does it
  // return a block-indepedent value?
  std::map<const Function *, BSizeDependenceValue> FunctionReturnValueMap;

  // Is function block-size independent?
  std::map<const Function *, bool> FunctionBSIMap;
 
  // Run analysis on functions.
  for (Function *F : functionList) {
    LLVM_DEBUG(errs() << "-------------- Computing Block-size Invariance ------------------\n");
#ifdef ENTRYPOINTS_ONLY
    if (entrypoints.find(F) != entrypoints.end()) {
#endif
      errs() << "Function: " << demangle(F->getName().data()) << "\n";
#ifdef ENTRYPOINTS_ONLY
    }
#endif
  
    std::set<const Instruction*> dependentAccesses;
    std::set<const Instruction*> syncthreads;
    DominatorTree DT(*F);
    for (int i = 0; i < 3; i++) {
      BlockSizeInvarianceAnalysis BDA(F, &DT, i,
          &FunctionReturnValueMap, &FunctionBSIMap);
      BSizeGPUState st = BDA.BuildInitialState();
      BDA.BuildAnalysisInfo(st);
      auto depSet = BDA.getBlockSizeDependentAccesses();
      auto syncSet = BDA.getSyncThreads();
      dependentAccesses.insert(depSet.begin(), depSet.end());
      syncthreads.insert(syncSet.begin(), syncSet.end());
    }
  
    // If block-size dependent accesses is empty, print block-size invariance.
    if (dependentAccesses.empty() && syncthreads.empty()) {
      FunctionBSIMap[F] = true;
      // Adding method to the set of block-size independent methods!!!!
      BlockSizeIndependentMethods_.insert(F);

#ifdef ENTRYPOINTS_ONLY
      if (entrypoints.find(F) == entrypoints.end()) { continue; }
#endif
      errs() << "Function " << demangle(F->getName().data()) << " is block-size independent!\n";
    } else {
      FunctionBSIMap[F] = false;

#ifdef ENTRYPOINTS_ONLY
      if (entrypoints.find(F) == entrypoints.end()) { continue; }
#endif
      // Print block-size dependent accesses found by the analysis.
      errs() << "  Block-size dependent accesses: #" 
          << dependentAccesses.size() << "\n";
      for (auto it = dependentAccesses.begin(), ite = dependentAccesses.end();
               it != ite; ++it) {
        errs() << "  -- ";
        (*it)->getDebugLoc().print(errs());
        errs() << "\n";
      }
      errs() << "\n";
    }
  }
  return false;
}

char InterproceduralBlockSizeInvarianceAnalysisPass::ID = 0;
static RegisterPass<InterproceduralBlockSizeInvarianceAnalysisPass>
Y("interproc-bsize-invariance-analysis", "Interprocedural analysis to identify block-size independent GPU kernels.");
