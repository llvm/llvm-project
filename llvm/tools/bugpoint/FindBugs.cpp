//===-- FindBugs.cpp - Run Many Different Optimizations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an interface that allows bugpoint to choose different 
// combinations of optimizations to run on the selected input. Bugpoint will 
// run these optimizations and record the success/failure of each. This way
// we can hopefully spot bugs in the optimizations.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ToolRunner.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <ctime>
#include <iostream>
using namespace llvm;

/// runManyPasses - Take the specified pass list and create different 
/// combinations of passes to compile the program with. Compile the program with
/// each set and mark test to see if it compiled correctly. If the passes 
/// compiled correctly output nothing and rearrange the passes into a new order.
/// If the passes did not compile correctly, output the command required to 
/// recreate the failure. This returns true if a compiler error is found.
///
bool BugDriver::runManyPasses(const std::vector<const PassInfo*> &AllPasses) {
  setPassesToRun(AllPasses);
  std::cout << "Starting bug finding procedure...\n\n";
  
  // Creating a reference output if necessary
  if (initializeExecutionEnvironment()) return false;
  
  std::cout << "\n";
  if (ReferenceOutputFile.empty()) {
    std::cout << "Generating reference output from raw program: \n";
    if (!createReferenceFile(Program))
      return false;
  }
  
  srand(time(NULL));  
  
  unsigned num = 1;
  while(1) {  
    //
    // Step 1: Randomize the order of the optimizer passes.
    //
    std::random_shuffle(PassesToRun.begin(), PassesToRun.end());
    
    //
    // Step 2: Run optimizer passes on the program and check for success.
    //
    std::cout << "Running selected passes on program to test for crash: ";
    for(int i = 0, e = PassesToRun.size(); i != e; i++) {
      std::cout << "-" << PassesToRun[i]->getPassArgument( )<< " ";
    }
    
    std::string Filename;
    if(runPasses(PassesToRun, Filename, false)) {
      std::cout << "\n";
      std::cout << "Optimizer passes caused failure!\n\n";
      debugOptimizerCrash();
      return true;
    } else {
      std::cout << "Combination " << num << " optimized successfully!\n";
    }
    
    //
    // Step 3: Compile the optimized code.
    //
    std::cout << "Running the code generator to test for a crash: ";
    try {
      compileProgram(Program);
      std::cout << '\n';
    } catch (ToolExecutionError &TEE) {
      std::cout << "\n*** compileProgram threw an exception: ";
      std::cout << TEE.what();
      return debugCodeGeneratorCrash();
    }
    
    //
    // Step 4: Run the program and compare its output to the reference 
    // output (created above).
    //
    std::cout << "*** Checking if passes caused miscompliation:\n";
    try {
      if (diffProgram(Filename, "", false)) {
        std::cout << "\n*** diffProgram returned true!\n";
        debugMiscompilation();
        return true;
      } else {
        std::cout << "\n*** diff'd output matches!\n";
      }
    } catch (ToolExecutionError &TEE) {
      std::cerr << TEE.what();
      debugCodeGeneratorCrash();
      return true;
    }
    
    sys::Path(Filename).eraseFromDisk();
    
    std::cout << "\n\n";
    num++;
  } //end while
  
  // Unreachable.
}
