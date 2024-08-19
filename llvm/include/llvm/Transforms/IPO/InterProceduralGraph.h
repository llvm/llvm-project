#ifndef INTERPROCEDURALGRAPH_PASS_H
#define INTERPROCEDURALGRAPH_PASS_H

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/CallGraphWrapperPass.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <vector>
#include <set>

namespace llvm {

// Forward declaration of the pass
class InterproceduralGraphPass;

// Node structure to represent graph nodes
struct Node {
    std::string name;
    // Additional properties can be added here
};

// Edge structure to represent graph edges
struct Edge {
    std::string from;
    std::string to;
};

// InterproceduralGraph class to manage graph nodes and edges
struct InterproceduralGraph {
    std::map<std::string, Node> nodes;
    std::vector<Edge> edges;
    std::set<std::string> functionsWithIndirectCalls;

    void addNode(Node node);
    void addEdge(const std::string &from, const std::string &to);
    void addIntraproceduralEdges(Function &F);
    void addInterproceduralEdges(CallGraph &CG);
    void addIndirectCallEdges();
    std::set<std::string> getPossibleIndirectTargets(const std::string &FuncName);
    void outputGraph();
};

// InterproceduralGraphPass class to define the pass
class InterproceduralGraphPass : public PassInfoMixin<InterproceduralGraphPass> {
public:
    // Function to run the pass
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

    // Function to check if the pass is required
    static bool isRequired();
};

} // namespace llvm

#endif // INTERPROCEDURALGRAPH_PASS_H
