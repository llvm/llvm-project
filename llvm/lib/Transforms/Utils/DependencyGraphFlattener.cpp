#include "llvm/Transforms/Utils/DependencyGraphFlattener.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/ValueMap.h"

using namespace llvm;

PreservedAnalyses DependencyGraphFlattenerPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  ValueMap<Value*, std::vector<Value*>> dependency_map;

  // Iterating through each instruction
  for (inst_iterator current_instruction = inst_begin(F), last_instruction = inst_end(F);
       current_instruction != last_instruction; ++current_instruction) {
    // Create an empty vector to store direct/indirect dependencies of current_instruction
    std::vector<Value*> dependency_vector;

    // Iterating through operand list of current_instruction
    for (Use &U : current_instruction->operands()) {
      // Add the operand to the local vector
      dependency_vector.push_back(&*U.get());

      // If operand present in dependency_map, iterate through the corresponding 
      // dependency_vector and add each element to the local vector.
      if (dependency_map.count(U.get())) {
        for(std::vector<Value*>::iterator value_iterator = dependency_map[U.get()].begin();
             value_iterator != dependency_map[U.get()].end(); ++value_iterator) {
          dependency_vector.push_back(*value_iterator);
        }
      }

//      Value* v = U.get();
//      errs() << *v << "\n";
    }
    // Add instruction along with empty dependency_vector to dependency_map
    dependency_map.insert(std::make_pair(&*current_instruction, dependency_vector));

//    for(std::vector<Value*>::iterator value_iterator = std::begin(dependency_vector);
//         value_iterator != std::end(dependency_vector); ++value_iterator) {
//      errs() << **value_iterator << "\n";
//    }
  }

  // Printing dependency map for debugging.
//  for (auto map_it = dependency_map.begin(), last_pair = dependency_map.end();
//       map_it != last_pair; ++map_it) {
//    errs() << "The instruction " << *map_it->first << " uses:\n";
//    for(std::vector<Value*>::iterator value_iterator = map_it->second.begin();
//         value_iterator != map_it->second.end(); ++value_iterator) {
//      errs() << **value_iterator << "\n";
//    }
//    errs() << "\n";
//  }

  return PreservedAnalyses::all();
}