#ifndef LLVM_LIB_TARGET_GENPAY_GENPAY_H
#define LLVM_LIB_TARGET_GENPAY_GENPAY_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class GenpayTargetMachine;
class PassRegistry;
class Target;

Target &getTheGenpayTarget();
FunctionPass *createGenpayISelDag(GenpayTargetMachine &TM);

// Defined via INITIALIZE_PASS in GenpayAsmPrinter.cpp / GenpayISelDAGToDAG.cpp.
// Declared here (cf. AArch64.h) so the INITIALIZE_PASS macro expansion and
// TargetPassConfig users see a prior declaration inside namespace llvm.
void initializeGenpayAsmPrinterPass(PassRegistry &);
void initializeGenpayDAGToDAGISelLegacyPass(PassRegistry &);

} // namespace llvm

#endif
