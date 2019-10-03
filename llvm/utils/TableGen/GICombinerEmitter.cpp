//===- GlobalCombinerEmitter.cpp - Generate a combiner --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Generate a combiner implementation for GlobalISel from a declarative
/// syntax
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "CodeGenTarget.h"

using namespace llvm;

#define DEBUG_TYPE "gicombiner-emitter"

cl::OptionCategory
    GICombinerEmitterCat("Options for -gen-global-isel-combiner");
static cl::list<std::string>
    SelectedCombiners("combiners", cl::desc("Emit the specified combiners"),
                      cl::cat(GICombinerEmitterCat), cl::CommaSeparated);
namespace {
class GICombinerEmitter {
  StringRef Name;
  Record *Combiner;
public:
  explicit GICombinerEmitter(RecordKeeper &RK, StringRef Name,
                             Record *Combiner);
  ~GICombinerEmitter() {}

  StringRef getClassName() const {
    return Combiner->getValueAsString("Classname");
  }
  void run(raw_ostream &OS);

};

GICombinerEmitter::GICombinerEmitter(RecordKeeper &RK, StringRef Name,
                                     Record *Combiner)
    : Name(Name), Combiner(Combiner) {}

void GICombinerEmitter::run(raw_ostream &OS) {
  NamedRegionTimer T("Emit", "Time spent emitting the combiner",
                     "Code Generation", "Time spent generating code",
                     TimeRegions);
  OS << "#ifdef " << Name.upper() << "_GENCOMBINERHELPER_DEPS\n"
     << "#endif // ifdef " << Name.upper() << "_GENCOMBINERHELPER_DEPS\n\n";

  OS << "#ifdef " << Name.upper() << "_GENCOMBINERHELPER_H\n"
     << "class " << getClassName() << " {\n"
     << "public:\n"
     << "  bool tryCombineAll(\n"
     << "    GISelChangeObserver &Observer,\n"
     << "    MachineInstr &MI,\n"
     << "    MachineIRBuilder &B) const;\n"
     << "};\n";
  OS << "#endif // ifdef " << Name.upper() << "_GENCOMBINERHELPER_H\n\n";

  OS << "#ifdef " << Name.upper() << "_GENCOMBINERHELPER_CPP\n"
     << "\n"
     << "bool " << getClassName() << "::tryCombineAll(\n"
     << "    GISelChangeObserver &Observer,\n"
     << "    MachineInstr &MI,\n"
     << "    MachineIRBuilder &B) const {\n"
     << "  MachineBasicBlock *MBB = MI.getParent();\n"
     << "  MachineFunction *MF = MBB->getParent();\n"
     << "  MachineRegisterInfo &MRI = MF->getRegInfo();\n"
     << "  (void)MBB; (void)MF; (void)MRI;\n\n";
  OS << "\n  return false;\n"
     << "}\n"
     << "#endif // ifdef " << Name.upper() << "_GENCOMBINERHELPER_CPP\n";
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

namespace llvm {
void EmitGICombiner(RecordKeeper &RK, raw_ostream &OS) {
  CodeGenTarget Target(RK);
  emitSourceFileHeader("Global Combiner", OS);

  if (SelectedCombiners.empty())
    PrintFatalError("No combiners selected with -combiners");
  for (const auto &Combiner : SelectedCombiners) {
    Record *CombinerDef = RK.getDef(Combiner);
    if (!CombinerDef)
      PrintFatalError("Could not find " + Combiner);
    GICombinerEmitter(RK, Combiner, CombinerDef).run(OS);
  }
}

} // namespace llvm
