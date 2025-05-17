/* --- PE.h --- */

/* ------------------------------------------
Author: undefined
Date: 4/7/2025
------------------------------------------ */

#ifndef PE_H
#define PE_H
#include "llvm/Target/TargetMachine.h"
namespace llvm{
    class FunctionPass;
    class PETargetMachine;
    FunctionPass *createPEISelDag(PETargetMachine &TM,CodeGenOptLevel OptLevel);
    void initializePEDAGToDAGISelLegacyPass(PassRegistry &);
}


#endif // PE_H
