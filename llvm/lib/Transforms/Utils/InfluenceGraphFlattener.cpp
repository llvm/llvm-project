//
// Created by tanmay on 5/25/22.
//

#include "llvm/Transforms/Utils/InfluenceGraphFlattener.h"
#include "llvm/IR/InstIterator.h"

// Procedure to build this Pass was copied from the BranchProbabilityInfo Pass
// You can search for this pass by looking for "branch-prob" in PassRegistry.def

using namespace llvm;

void FlattenedInfluenceGraph::print(raw_ostream &OS) const {
  OS << "---- Flattened Influence Graph ----\n";

  // Printing the InfluenceGraph
  // TODO: Improve the display format of this graph
  for (auto map_it = influence_map.begin(), last_pair = influence_map.end();
       map_it != last_pair; ++map_it) {
    OS << "The instruction " << *map_it->first << " is used by:\n";
    for(std::vector<Value*>::const_iterator value_iterator = map_it->second.begin();
         value_iterator != map_it->second.end(); ++value_iterator) {
      OS << **value_iterator << "\n";
    }
    OS << "\n";
  }
}

// Defining the Key
AnalysisKey InfluenceGraphFlattenerAnalysis::Key;

// TODO: Correct this Analysis to make it work for influence instead of
//  dependence by traversing the IR in reverse and accessing the USERS list.
// Defining run function that will generate the Flattened Influence Graph
FlattenedInfluenceGraph InfluenceGraphFlattenerAnalysis::run(Function &F,
                                                               FunctionAnalysisManager &AM) {
  FlattenedInfluenceGraph FIG;

  FIG.LastF = &F; // Store the last function we ran on for printing.

  // Iterating through each instruction
  inst_iterator current_instruction = inst_end(F);
  current_instruction--;
  inst_iterator first_instruction = inst_begin(F);
  for (; current_instruction != first_instruction; --current_instruction) {
    // Create an empty vector to store direct/indirect dependencies of current_instruction
    std::vector<Value*> influence_vector;

    // Iterating through user list of current_instruction
    for (User *U : current_instruction->users()) {
      // Add the user to the local vector
      if (dyn_cast<Instruction>(U)) {
        influence_vector.push_back(U);
      }

      // If User present in FIG.influence_map, iterate through the corresponding
      // influence_vector and add each element to the local vector.
      if (FIG.influence_map.count(U)) {
        for(std::vector<Value*>::iterator value_iterator = FIG.influence_map[U].begin();
             value_iterator != FIG.influence_map[U].end(); ++value_iterator) {
          influence_vector.push_back(*value_iterator);
        }
      }
    }
    // Add instruction along with empty influence_vector to FIG.influence_map
    FIG.influence_map.insert(std::make_pair(&*current_instruction, influence_vector));
  }
  // Create an empty vector to store direct/indirect dependencies of first_instruction
  std::vector<Value*> influence_vector;

  // Iterating through user list of first_instruction
  for (User *U : first_instruction->users()) {
    // Add the user to the local vector
    if (dyn_cast<Instruction>(U)) {
      influence_vector.push_back(U);
    }

    // If User present in FIG.influence_map, iterate through the corresponding
    // influence_vector and add each element to the local vector.
    if (FIG.influence_map.count(U)) {
      for(std::vector<Value*>::iterator value_iterator = FIG.influence_map[U].begin();
           value_iterator != FIG.influence_map[U].end(); ++value_iterator) {
        influence_vector.push_back(*value_iterator);
      }
    }
  }
  // Add instruction along with empty influence_vector to FIG.influence_map
  FIG.influence_map.insert(std::make_pair(&*current_instruction, influence_vector));

  return FIG;
}

PreservedAnalyses
InfluenceGraphFlattenerPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results of FIG for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<InfluenceGraphFlattenerAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

bool FlattenedInfluenceGraphWrapperPass::runOnFunction(Function &F) {
  this->FIG.LastF = &F; // Store the last function we ran on for printing.

  // Iterating through each instruction
  inst_iterator current_instruction = inst_end(F);
  current_instruction--;
  inst_iterator first_instruction = inst_begin(F);
  for (; current_instruction != first_instruction; --current_instruction) {
    // Create an empty vector to store direct/indirect dependencies of current_instruction
    std::vector<Value*> influence_vector;

    // Iterating through user list of current_instruction
    for (User *U : current_instruction->users()) {
      // Add the user to the local vector
      if (dyn_cast<Instruction>(U)) {
        influence_vector.push_back(U);
      }

      // If User present in FIG.influence_map, iterate through the corresponding
      // influence_vector and add each element to the local vector.
      if (FIG.influence_map.count(U)) {
        for(std::vector<Value*>::iterator value_iterator = FIG.influence_map[U].begin();
             value_iterator != FIG.influence_map[U].end(); ++value_iterator) {
          influence_vector.push_back(*value_iterator);
        }
      }
    }
    // Add instruction along with empty influence_vector to FIG.influence_map
    FIG.influence_map.insert(std::make_pair(&*current_instruction, influence_vector));
  }
  // Create an empty vector to store direct/indirect dependencies of first_instruction
  std::vector<Value*> influence_vector;

  // Iterating through user list of first_instruction
  for (User *U : first_instruction->users()) {
    // Add the user to the local vector
    if (dyn_cast<Instruction>(U)) {
      influence_vector.push_back(U);
    }

    // If User present in FIG.influence_map, iterate through the corresponding
    // influence_vector and add each element to the local vector.
    if (FIG.influence_map.count(U)) {
      for(std::vector<Value*>::iterator value_iterator = FIG.influence_map[U].begin();
           value_iterator != FIG.influence_map[U].end(); ++value_iterator) {
        influence_vector.push_back(*value_iterator);
      }
    }
  }
  // Add instruction along with empty influence_vector to FIG.influence_map
  FIG.influence_map.insert(std::make_pair(&*current_instruction, influence_vector));

  return false;
}

void FlattenedInfluenceGraphWrapperPass::print(raw_ostream &OS, const Module *M) const {
  FIG.print(OS);
}