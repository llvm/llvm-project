#include "llvm/Transforms/Utils/DependencyGraphFlattener.h"
#include "llvm/IR/InstIterator.h"

// Procedure to build this Pass was copied from the BranchProbabilityInfo Pass
// You can search for this pass by looking for "branch-prob" in PassRegistry.def

using namespace llvm;

void FlattenedDependencyGraph::print(raw_ostream &OS) const {
  OS << "---- Flattened Dependency Graph ----\n";

  // Printing the DependencyGraph
  // TODO: Improve the display format of this graph
  for (auto map_it = dependency_map.begin(), last_pair = dependency_map.end();
       map_it != last_pair; ++map_it) {
    OS << "The instruction " << *map_it->first << " uses:\n";
    for(std::vector<Value*>::const_iterator value_iterator = map_it->second.begin();
         value_iterator != map_it->second.end(); ++value_iterator) {
      OS << **value_iterator << "\n";
    }
    OS << "\n";
  }
}

// Defining the Key
AnalysisKey DependencyGraphFlattenerAnalysis::Key;

// Defining run function that will generate the Flattened Dependency Graph
FlattenedDependencyGraph DependencyGraphFlattenerAnalysis::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  FlattenedDependencyGraph FDG;

  FDG.LastF = &F; // Store the last function we ran on for printing.

  // Iterating through each instruction
  for (inst_iterator current_instruction = inst_begin(F), last_instruction = inst_end(F);
       current_instruction != last_instruction; ++current_instruction) {
    // Create an empty vector to store direct/indirect dependencies of current_instruction
    std::vector<Value *> dependency_vector;

    // Iterating through operand list of current_instruction
    for (Use &U : current_instruction->operands()) {
      // Add the operand to the local vector
      dependency_vector.push_back(&*U.get());

      // If operand present in FDG.dependency_map, iterate through the corresponding dependency_vector and add each element to the local vector.
      if (FDG.dependency_map.count(U.get())) {
        for (std::vector<Value *>::iterator value_iterator =
                 FDG.dependency_map[U.get()].begin();
             value_iterator != FDG.dependency_map[U.get()].end();
             ++value_iterator) {
          dependency_vector.push_back(*value_iterator);
        }
      }
    }
    // Add instruction along with empty dependency_vector to FDG.dependency_map
    FDG.dependency_map.insert(
        std::make_pair(&*current_instruction, dependency_vector));
  }
  return FDG;
}

PreservedAnalyses
DependencyGraphFlattenerPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<DependencyGraphFlattenerAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

bool FlattenedDependencyGraphWrapperPass::runOnFunction(Function &F) {
  this->FDG.LastF = &F; // Store the last function we ran on for printing.

  // Iterating through each instruction
  for (inst_iterator current_instruction = inst_begin(F), last_instruction = inst_end(F);
       current_instruction != last_instruction; ++current_instruction) {
    // Create an empty vector to store direct/indirect dependencies of current_instruction
    std::vector<Value*> dependency_vector;

    // Iterating through operand list of current_instruction
    for (Use &U : current_instruction->operands()) {
      // Add the operand to the local vector
      dependency_vector.push_back(&*U.get());

      // If operand present in this->FDG.dependency_map, iterate through the corresponding
      // dependency_vector and add each element to the local vector.
      if (this->FDG.dependency_map.count(U.get())) {
        for(std::vector<Value*>::iterator value_iterator = this->FDG.dependency_map[U.get()].begin();
             value_iterator != this->FDG.dependency_map[U.get()].end(); ++value_iterator) {
          dependency_vector.push_back(*value_iterator);
        }
      }
    }
    // Add instruction along with empty dependency_vector to this->FDG.dependency_map
    this->FDG.dependency_map.insert(std::make_pair(&*current_instruction, dependency_vector));
  }

  return false;
}

void FlattenedDependencyGraphWrapperPass::print(raw_ostream &OS, const Module *M) const {
  FDG.print(OS);
}