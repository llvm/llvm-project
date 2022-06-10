#include "llvm/Transforms/Utils/ComputationGraphGenerator.h"

#include "llvm/IR/InstIterator.h"

using namespace llvm;

// Defining the Key
AnalysisKey ComputationGraphAnalysis::Key;

// This function will generate a map from instructions to a list of dependency
// paths for that instruction in the Data dependency graph
ComputationGraph ComputationGraphAnalysis::run(Function &F,
                                                        FunctionAnalysisManager &AM) {
    ComputationGraph CG;

//    for (inst_iterator current_instruction = inst_begin(F), last_instruction = inst_end(F);
//         current_instruction != last_instruction; ++current_instruction) {
//      errs() << "Instruction: " << &*current_instruction << "  " << *current_instruction << "\n";
//      assert(!CG.getNode(&(*current_instruction)));
//
//      // Create a node for this instruction.
//      CGNode* new_node = new CGNode(*current_instruction);
//      errs() << "Node: " << &new_node;
//
//      // For every instruction used by this node, add an outgoing edge to the
//      // node.
//      // Iterating through operand list of current_instruction
//      errs() << "Inside for loop:\n";
//      for (Use &U : current_instruction->operands()) {
//        if(CGNode* tgt_node = CG.getNode(static_cast<Instruction *>(U.get()))) {
//          CGEdge new_edge = CGEdge(*tgt_node);
//          new_node.addEdge(new_edge);
//        }
//        errs() << &*U.get() << "  " << *U.get() << "\n";
//      }
//
//      errs() << "Outside for loop:\n";
//      for (DGNode<CGNode, CGEdge>::iterator current_edge =
//               new_node.begin();
//           current_edge != new_node.end(); ++current_edge) {
//        CGNode* tgt_node = &(*current_edge)->getTargetNode();
//        errs() << *tgt_node->getInstruction() << "\n";
//      }
//      CG.addNode(new_node);
//    }

    return CG;
}


PreservedAnalyses
ComputationGraphPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results for function "
     << "'" << F.getName() << "':"
     << "\n";
//  AM.getResult<ComputationGraphAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}