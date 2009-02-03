//===-- LLVMTargetMachine.cpp - Implement the LLVMTargetMachine class -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMTargetMachine class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {
  bool EnableFastISel;
}

static cl::opt<bool> PrintLSR("print-lsr-output", cl::Hidden,
    cl::desc("Print LLVM IR produced by the loop-reduce pass"));
static cl::opt<bool> PrintISelInput("print-isel-input", cl::Hidden,
    cl::desc("Print LLVM IR input to isel pass"));
static cl::opt<bool> PrintEmittedAsm("print-emitted-asm", cl::Hidden,
    cl::desc("Dump emitter generated instructions as assembly"));
static cl::opt<bool> PrintGCInfo("print-gc", cl::Hidden,
    cl::desc("Dump garbage collector data"));

// Hidden options to help debugging
static cl::opt<bool>
EnableSinking("enable-sinking", cl::init(false), cl::Hidden,
              cl::desc("Perform sinking on machine code"));
static cl::opt<bool>
EnableLICM("machine-licm",
           cl::init(false), cl::Hidden,
           cl::desc("Perform loop-invariant code motion on machine code"));

// When this works it will be on by default.
static cl::opt<bool>
DisablePostRAScheduler("disable-post-RA-scheduler",
                       cl::desc("Disable scheduling after register allocation"),
                       cl::init(true));

// Enable or disable FastISel. Both options are needed, because
// FastISel is enabled by default with -fast, and we wish to be
// able to enable or disable fast-isel independently from -fast.
static cl::opt<cl::boolOrDefault>
EnableFastISelOption("fast-isel", cl::Hidden,
  cl::desc("Enable the experimental \"fast\" instruction selector"));

FileModel::Model
LLVMTargetMachine::addPassesToEmitFile(PassManagerBase &PM,
                                       raw_ostream &Out,
                                       CodeGenFileType FileType,
                                       bool Fast) {
  // Add common CodeGen passes.
  if (addCommonCodeGenPasses(PM, Fast))
    return FileModel::Error;

  // Fold redundant debug labels.
  PM.add(createDebugLabelFoldingPass());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  if (addPreEmitPass(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  if (!Fast)
    PM.add(createLoopAlignerPass());

  switch (FileType) {
  default:
    break;
  case TargetMachine::AssemblyFile:
    if (addAssemblyEmitter(PM, Fast, Out))
      return FileModel::Error;
    return FileModel::AsmFile;
  case TargetMachine::ObjectFile:
    if (getMachOWriterInfo())
      return FileModel::MachOFile;
    else if (getELFWriterInfo())
      return FileModel::ElfFile;
  }

  return FileModel::Error;
}

/// addPassesToEmitFileFinish - If the passes to emit the specified file had to
/// be split up (e.g., to add an object writer pass), this method can be used to
/// finish up adding passes to emit the file, if necessary.
bool LLVMTargetMachine::addPassesToEmitFileFinish(PassManagerBase &PM,
                                                  MachineCodeEmitter *MCE,
                                                  bool Fast) {
  if (MCE)
    addSimpleCodeEmitter(PM, Fast, PrintEmittedAsm, *MCE);

  PM.add(createGCInfoDeleter());

  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());

  return false; // success!
}

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to
/// get machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool LLVMTargetMachine::addPassesToEmitMachineCode(PassManagerBase &PM,
                                                   MachineCodeEmitter &MCE,
                                                   bool Fast) {
  // Add common CodeGen passes.
  if (addCommonCodeGenPasses(PM, Fast))
    return true;

  if (addPreEmitPass(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  addCodeEmitter(PM, Fast, PrintEmittedAsm, MCE);

  PM.add(createGCInfoDeleter());

  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());

  return false; // success!
}

/// addCommonCodeGenPasses - Add standard LLVM codegen passes used for
/// both emitting to assembly files or machine code output.
///
bool LLVMTargetMachine::addCommonCodeGenPasses(PassManagerBase &PM, bool Fast) {
  // Standard LLVM-Level Passes.

  // Run loop strength reduction before anything else.
  if (!Fast) {
    PM.add(createLoopStrengthReducePass(getTargetLowering()));
    if (PrintLSR)
      PM.add(createPrintFunctionPass("\n\n*** Code after LSR ***\n", &errs()));
  }

  PM.add(createGCLoweringPass());

  if (!getTargetAsmInfo()->doesSupportExceptionHandling())
    PM.add(createLowerInvokePass(getTargetLowering()));

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  if (!Fast)
    PM.add(createCodeGenPreparePass(getTargetLowering()));

  PM.add(createStackProtectorPass(getTargetLowering()));

  if (PrintISelInput)
    PM.add(createPrintFunctionPass("\n\n"
                                   "*** Final LLVM Code input to ISel ***\n",
                                   &errs()));

  // Standard Lower-Level Passes.

  // Enable FastISel with -fast, but allow that to be overridden.
  if (EnableFastISelOption == cl::BOU_TRUE ||
      (Fast && EnableFastISelOption != cl::BOU_FALSE))
    EnableFastISel = true;

  // Ask the target for an isel.
  if (addInstSelector(PM, Fast))
    return true;

  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  if (EnableLICM)
    PM.add(createMachineLICMPass());

  if (EnableSinking)
    PM.add(createMachineSinkingPass());

  // Run pre-ra passes.
  if (addPreRegAlloc(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Perform register allocation.
  PM.add(createRegisterAllocator());

  // Perform stack slot coloring.
  if (!Fast)
    PM.add(createStackSlotColoringPass());

  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Run post-ra passes.
  if (addPostRegAlloc(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  PM.add(createLowerSubregsPass());

  if (PrintMachineCode)  // Print the subreg lowered code
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Second pass scheduler.
  if (!Fast && !DisablePostRAScheduler) {
    PM.add(createPostRAScheduler());

    if (PrintMachineCode)
      PM.add(createMachineFunctionPrinterPass(cerr));
  }

  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (!Fast)
    PM.add(createBranchFoldingPass(getEnableTailMergeDefault()));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  PM.add(createGCMachineCodeAnalysisPass());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  if (PrintGCInfo)
    PM.add(createGCInfoPrinter(*cerr));

  return false;
}
