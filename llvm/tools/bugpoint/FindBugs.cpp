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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <ctime>
using namespace llvm;

/// runManyPasses - Take the specified pass list and create different 
/// combinations of passes to compile the program with. Compile the program with
/// each set and mark test to see if it compiled correctly. If the passes 
/// compiled correctly output nothing and rearrange the passes into a new order.
/// If the passes did not compile correctly, output the command required to 
/// recreate the failure. This returns true if a compiler error is found.
///
bool BugDriver::runManyPasses(const std::vector<std::string> &AllPasses,
                              std::string &ErrMsg) {
  setPassesToRun(AllPasses);
  outs() << "Starting bug finding procedure...\n\n";
  
  // Creating a reference output if necessary
  if (initializeExecutionEnvironment()) return false;
  
  outs() << "\n";
  if (ReferenceOutputFile.empty()) {
    outs() << "Generating reference output from raw program: \n";
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
    outs() << "Running selected passes on program to test for crash: ";
    for(int i = 0, e = PassesToRun.size(); i != e; i++) {
      outs() << "-" << PassesToRun[i] << " ";
    }
    
    std::string Filename;
    if(runPasses(Program, PassesToRun, Filename, false)) {
      outs() << "\n";
      outs() << "Optimizer passes caused failure!\n\n";
      debugOptimizerCrash();
      return true;
    } else {
      outs() << "Combination " << num << " optimized successfully!\n";
    }
    
    //
    // Step 3: Compile the optimized code.
    //
    outs() << "Running the code generator to test for a crash: ";
    std::string Error;
    compileProgram(Program, &Error);
    if (!Error.empty()) {
      outs() << "\n*** compileProgram threw an exception: ";
      outs() << Error;
      return debugCodeGeneratorCrash(ErrMsg);
    }
    outs() << '\n';
    
    //
    // Step 4: Run the program and compare its output to the reference 
    // output (created above).
    //
    outs() << "*** Checking if passes caused miscompliation:\n";
    bool Diff = diffProgram(Program, Filename, "", false, &Error);
    if (Error.empty() && Diff) {
      outs() << "\n*** diffProgram returned true!\n";
      debugMiscompilation(&Error);
      if (Error.empty())
        return true;
    }
    if (!Error.empty()) {
      errs() << Error;
      debugCodeGeneratorCrash(ErrMsg);
      return true;
    }
    outs() << "\n*** diff'd output matches!\n";
    
    sys::Path(Filename).eraseFromDisk();
    
    outs() << "\n\n";
    num++;
  } //end while
  
  // Unreachable.
}
