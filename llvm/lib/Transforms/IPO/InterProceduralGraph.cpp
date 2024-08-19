#include "llvm/Transforms/IPO/InterProceduralGraph.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <vector>
#include <set>

using namespace llvm;

namespace {

struct Node {
    std::string name;
    // Additional properties can be added here
};

struct Edge {
    std::string from;
    std::string to;
};

struct InterproceduralGraph {
    std::map<std::string, Node> nodes;
    std::vector<Edge> edges;
    std::set<std::string> functionsWithIndirectCalls;

    void addNode(Node node) {
        nodes[node.name] = node;
    }

    void addEdge(const std::string &from, const std::string &to) {
        edges.push_back({from, to});
    }

    void addIntraproceduralEdges(Function &F) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (auto *CI = dyn_cast<CallInst>(&I)) {
                    if (auto *Callee = dyn_cast<Function>(CI->getCalledFunction())) {
                        if (Callee && !Callee->isDeclaration()) {
                            addEdge(F.getName().str(), Callee->getName().str());
                            functionsWithIndirectCalls.insert(Callee->getName().str());
                        }
                    }
                }
            }
        }
    }

    void addInterproceduralEdges(CallGraph &CG) {
        for (auto &nodePair : CG) {
            const Function *caller = nodePair.first;
            CallGraphNode *cgn = nodePair.second.get();

            for (auto it = cgn->begin(); it != cgn->end(); ++it) {
                CallGraphNode::CallRecord callRecord = *it;
                const Function *callee = callRecord.second->getFunction();
                if (callee) {
                    addEdge(caller->getName().str(), callee->getName().str());
                }
            }
        }
    }

    void addIndirectCallEdges() {
        for (const auto &FuncName : functionsWithIndirectCalls) {
            for (const auto &TargetFuncName : getPossibleIndirectTargets(FuncName)) {
                addEdge(FuncName, TargetFuncName);
            }
        }
    }

    std::set<std::string> getPossibleIndirectTargets(const std::string &FuncName) {
        return std::set<std::string>();
    }

    void outputGraph() {
        errs() << "Nodes:\n";
        for (const auto &NodePair : nodes) {
            errs() << "  " << NodePair.first << "\n";
        }

        errs() << "Edges:\n";
        for (const auto &Edge : edges) {
            errs() << "  " << Edge.from << " -> " << Edge.to << "\n";
        }
    }
};

class InterproceduralGraphPass : public PassInfoMixin<InterproceduralGraphPass> {
public:
     PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        InterproceduralGraph IPG;

        CallGraph &CG = AM.getResult<CallGraphAnalysis>(M);

        for (Function &F : M) {
            Node functionNode;
            functionNode.name = F.getName();
            IPG.addNode(functionNode);
            IPG.addIntraproceduralEdges(F);
        }

        IPG.addInterproceduralEdges(CG);
        IPG.addIndirectCallEdges();
        IPG.outputGraph();

        return PreservedAnalyses::none();
    }

    //static bool isRequired() { return true; }
};

} // end of anonymous namespace
