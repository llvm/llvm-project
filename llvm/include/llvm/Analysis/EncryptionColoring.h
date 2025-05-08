//===- EncryptionColoring.cpp - Color values as encrypted or not -----------==//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// The definition of the encryption coloring algorithm. In fully-homomorphic
// encryption programs, values may be encrypted or not, requiring different
// treatment. This analysis pass computes which values are encrypted of not from
// the DDG. The rule is simply that operations over only plaintext operands
// produce a plaintext while operations over one or more ciphertexts produce
// a ciphertext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ENCRYPTION_COLORING_H
#define LLVM_ANALYSIS_ENCRYPTION_COLORING_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/EncryptionColor.h"
#include "llvm/ADT/MapVector.h"

namespace llvm {
class Function;

class EncryptionColoringAnalysis;

class EncryptionColoring {
    friend EncryptionColoringAnalysis;
private:
    Function* Parent;
    bool IsApplicable;
    MapVector<const Value*, EncryptionColor> Colors;

public:
    EncryptionColoring(Function &F);

    std::optional<EncryptionColor> getColor(const Value* val);
};

class EncryptionColoringAnalysis : public AnalysisInfoMixin<EncryptionColoringAnalysis> {
    friend AnalysisInfoMixin<EncryptionColoringAnalysis>;
    static AnalysisKey Key;

private:
    void colorInstruction(const Instruction &I, EncryptionColoring &Coloring);

public:
    using Result = EncryptionColoring;

    EncryptionColoring run(Function &F, FunctionAnalysisManager &M);
};


}

#endif
