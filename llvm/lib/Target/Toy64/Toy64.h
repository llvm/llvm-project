//===-- Toy64.h - Top-level interface for Toy64 backend --------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64_H
#define LLVM_LIB_TARGET_TOY64_TOY64_H

namespace llvm {

class FunctionPass;
class PassRegistry;
class Toy64TargetMachine;

FunctionPass *
createToy64ISelDagLegacyPass(
    Toy64TargetMachine &TM);

void initializeToy64DAGToDAGISelLegacyPass(
    PassRegistry &);

} // namespace llvm

#endif