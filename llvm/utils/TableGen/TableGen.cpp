//===- TableGen.cpp - Top-Level TableGen implementation for LLVM ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for LLVM's TableGen.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h" // Declares all backends.

#include "SetTheory.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

enum ActionType {
  PrintRecords,
  GenEmitter,
  GenRegisterInfo,
  GenInstrInfo,
  GenAsmWriter,
  GenAsmMatcher,
  GenDisassembler,
  GenPseudoLowering,
  GenCallingConv,
  GenDAGISel,
  GenDFAPacketizer,
  GenFastISel,
  GenSubtarget,
  GenIntrinsic,
  GenTgtIntrinsic,
  GenEDInfo,
  PrintEnums,
  PrintSets
};

namespace {
  cl::opt<ActionType>
  Action(cl::desc("Action to perform:"),
         cl::values(clEnumValN(PrintRecords, "print-records",
                               "Print all records to stdout (default)"),
                    clEnumValN(GenEmitter, "gen-emitter",
                               "Generate machine code emitter"),
                    clEnumValN(GenRegisterInfo, "gen-register-info",
                               "Generate registers and register classes info"),
                    clEnumValN(GenInstrInfo, "gen-instr-info",
                               "Generate instruction descriptions"),
                    clEnumValN(GenCallingConv, "gen-callingconv",
                               "Generate calling convention descriptions"),
                    clEnumValN(GenAsmWriter, "gen-asm-writer",
                               "Generate assembly writer"),
                    clEnumValN(GenDisassembler, "gen-disassembler",
                               "Generate disassembler"),
                    clEnumValN(GenPseudoLowering, "gen-pseudo-lowering",
                               "Generate pseudo instruction lowering"),
                    clEnumValN(GenAsmMatcher, "gen-asm-matcher",
                               "Generate assembly instruction matcher"),
                    clEnumValN(GenDAGISel, "gen-dag-isel",
                               "Generate a DAG instruction selector"),
                    clEnumValN(GenDFAPacketizer, "gen-dfa-packetizer",
                               "Generate DFA Packetizer for VLIW targets"),
                    clEnumValN(GenFastISel, "gen-fast-isel",
                               "Generate a \"fast\" instruction selector"),
                    clEnumValN(GenSubtarget, "gen-subtarget",
                               "Generate subtarget enumerations"),
                    clEnumValN(GenIntrinsic, "gen-intrinsic",
                               "Generate intrinsic information"),
                    clEnumValN(GenTgtIntrinsic, "gen-tgt-intrinsic",
                               "Generate target intrinsic information"),
                    clEnumValN(GenEDInfo, "gen-enhanced-disassembly-info",
                               "Generate enhanced disassembly info"),
                    clEnumValN(PrintEnums, "print-enums",
                               "Print enum values for a class"),
                    clEnumValN(PrintSets, "print-sets",
                               "Print expanded sets for testing DAG exprs"),
                    clEnumValEnd));

  cl::opt<std::string>
  Class("class", cl::desc("Print Enum list for this class"),
          cl::value_desc("class name"));

bool LLVMTableGenMain(raw_ostream &OS, RecordKeeper &Records) {
  switch (Action) {
  case PrintRecords:
    OS << Records;           // No argument, dump all contents
    break;
  case GenEmitter:
    EmitCodeEmitter(Records, OS);
    break;
  case GenRegisterInfo:
    EmitRegisterInfo(Records, OS);
    break;
  case GenInstrInfo:
    EmitInstrInfo(Records, OS);
    break;
  case GenCallingConv:
    EmitCallingConv(Records, OS);
    break;
  case GenAsmWriter:
    EmitAsmWriter(Records, OS);
    break;
  case GenAsmMatcher:
    EmitAsmMatcher(Records, OS);
    break;
  case GenDisassembler:
    EmitDisassembler(Records, OS);
    break;
  case GenPseudoLowering:
    EmitPseudoLowering(Records, OS);
    break;
  case GenDAGISel:
    EmitDAGISel(Records, OS);
    break;
  case GenDFAPacketizer:
    EmitDFAPacketizer(Records, OS);
    break;
  case GenFastISel:
    EmitFastISel(Records, OS);
    break;
  case GenSubtarget:
    EmitSubtarget(Records, OS);
    break;
  case GenIntrinsic:
    EmitIntrinsics(Records, OS);
    break;
  case GenTgtIntrinsic:
    EmitIntrinsics(Records, OS, true);
    break;
  case GenEDInfo:
    EmitEnhancedDisassemblerInfo(Records, OS);
    break;
  case PrintEnums:
  {
    std::vector<Record*> Recs = Records.getAllDerivedDefinitions(Class);
    for (unsigned i = 0, e = Recs.size(); i != e; ++i)
      OS << Recs[i]->getName() << ", ";
    OS << "\n";
    break;
  }
  case PrintSets:
  {
    SetTheory Sets;
    Sets.addFieldExpander("Set", "Elements");
    std::vector<Record*> Recs = Records.getAllDerivedDefinitions("Set");
    for (unsigned i = 0, e = Recs.size(); i != e; ++i) {
      OS << Recs[i]->getName() << " = [";
      const std::vector<Record*> *Elts = Sets.expand(Recs[i]);
      assert(Elts && "Couldn't expand Set instance");
      for (unsigned ei = 0, ee = Elts->size(); ei != ee; ++ei)
        OS << ' ' << (*Elts)[ei]->getName();
      OS << " ]\n";
    }
    break;
  }
  }

  return false;
}
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &LLVMTableGenMain);
}
