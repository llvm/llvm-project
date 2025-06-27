/* --- PE.h --- */

/* ------------------------------------------
Author: undefined
Date: 4/7/2025
------------------------------------------ */

#ifndef PE_H
#define PE_H
#include "llvm/Target/TargetMachine.h"
namespace llvm {

// 对齐公式，x代表index，align代表对齐倍数（4 / 8 / 16）
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define ROUND_UP(x, align) (DIV_ROUND_UP(x, align) * (align))

class FunctionPass;
class PETargetMachine;
class PassRegistry;

FunctionPass *createPEISelDag(PETargetMachine &TM, CodeGenOptLevel OptLevel);

void initializePEDAGToDAGISelLegacyPass(PassRegistry &);
} // namespace llvm

#endif // PE_H
