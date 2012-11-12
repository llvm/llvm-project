//===-- llvm-dis.cpp - The low-level LLVM disassembler --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-dis [options]      - Read LLVM bitcode from stdin, write asm to stdout
//  llvm-dis [options] x.bc - Read LLVM bitcode from the x.bc file, write asm
//                            to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/DebugInfo.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Assembly/AssemblyAnnotationWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
DontPrint("disable-output", cl::desc("Don't output the .ll file"), cl::Hidden);

static cl::opt<bool>
ShowAnnotations("show-annotations",
                cl::desc("Add informational comments to the .ll file"));

namespace {

static void printDebugLoc(const DebugLoc &DL, formatted_raw_ostream &OS) {
  OS << DL.getLine() << ":" << DL.getCol();
  if (MDNode *N = DL.getInlinedAt(getGlobalContext())) {
    DebugLoc IDL = DebugLoc::getFromDILocation(N);
    if (!IDL.isUnknown()) {
      OS << "@";
      printDebugLoc(IDL,OS);
    }
  }
}
class CommentWriter : public AssemblyAnnotationWriter {
public:
  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &OS) {
    OS << "; [#uses=" << F->getNumUses() << ']';  // Output # uses
    OS << '\n';
  }
  void printInfoComment(const Value &V, formatted_raw_ostream &OS) {
    bool Padded = false;
    if (!V.getType()->isVoidTy()) {
      OS.PadToColumn(50);
      Padded = true;
      OS << "; [#uses=" << V.getNumUses() << " type=" << *V.getType() << "]";  // Output # uses and type
    }
    if (const Instruction *I = dyn_cast<Instruction>(&V)) {
      const DebugLoc &DL = I->getDebugLoc();
      if (!DL.isUnknown()) {
        if (!Padded) {
          OS.PadToColumn(50);
          Padded = true;
          OS << ";";
        }
        OS << " [debug line = ";
        printDebugLoc(DL,OS);
        OS << "]";
      }
      if (const DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(I)) {
        DIVariable Var(DDI->getVariable());
        if (!Padded) {
          OS.PadToColumn(50);
          OS << ";";
        }
        OS << " [debug variable = " << Var.getName() << "]";
      }
      else if (const DbgValueInst *DVI = dyn_cast<DbgValueInst>(I)) {
        DIVariable Var(DVI->getVariable());
        if (!Padded) {
          OS.PadToColumn(50);
          OS << ";";
        }
        OS << " [debug variable = " << Var.getName() << "]";
      }
    }
  }
};

} // end anon namespace

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.


  cl::ParseCommandLineOptions(argc, argv, "llvm .bc -> .ll disassembler\n");

  std::string ErrorMessage;
  std::auto_ptr<Module> M;

  // Use the bitcode streaming interface
  DataStreamer *streamer = getDataFileStreamer(InputFilename, &ErrorMessage);
  if (streamer) {
    std::string DisplayFilename;
    if (InputFilename == "-")
      DisplayFilename = "<stdin>";
    else
      DisplayFilename = InputFilename;
    M.reset(getStreamedBitcodeModule(DisplayFilename, streamer, Context,
                                     &ErrorMessage));
    if(M.get() != 0 && M->MaterializeAllPermanently(&ErrorMessage)) {
      M.reset();
    }
  }

  if (M.get() == 0) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }

  // Just use stdout.  We won't actually print anything on it.
  if (DontPrint)
    OutputFilename = "-";

  if (OutputFilename.empty()) { // Unspecified output, infer it.
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      const std::string &IFN = InputFilename;
      int Len = IFN.length();
      // If the source ends in .bc, strip it off.
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c')
        OutputFilename = std::string(IFN.begin(), IFN.end()-3)+".ll";
      else
        OutputFilename = IFN+".ll";
    }
  }

  std::string ErrorInfo;
  OwningPtr<tool_output_file>
  Out(new tool_output_file(OutputFilename.c_str(), ErrorInfo,
                           raw_fd_ostream::F_Binary));
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    return 1;
  }

  OwningPtr<AssemblyAnnotationWriter> Annotator;
  if (ShowAnnotations)
    Annotator.reset(new CommentWriter());

  // All that llvm-dis does is write the assembly to a file.
  if (!DontPrint)
    M->print(Out->os(), Annotator.get());

  // Declare success.
  Out->keep();

  return 0;
}
