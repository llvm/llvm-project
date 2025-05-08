//===- EncryptionColoring.cpp - Color values as encrypted or not -----------==//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// The implementation of the encryption coloring algorithm. In fully-homomorphic
// encryption programs, values may be encrypted or not, requiring different
// treatment. This analysis pass computes which values are encrypted of not from
// the DDG. The rule is simply that operations over only plaintext operands
// produce a plaintext while operations over one or more ciphertexts produce
// a ciphertext.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/EncryptionColoring.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

AnalysisKey EncryptionColoringAnalysis::Key;

EncryptionColoring EncryptionColoringAnalysis::run(Function &F,
                                                   FunctionAnalysisManager &M) {
    EncryptionColoring Coloring(F);

    if (!F.hasFnAttribute(Attribute::FheCircuit)) {
        return Coloring;
    } else {
        Coloring.IsApplicable = true;
    }

    // First, capture the color for the function arguments.
    for (const auto &A : F.args()) {
        if (!A.getType()->isPointerTy()) {
            report_fatal_error("An fhe_circuit function had a non-pointer argument");
        }

        if (A.hasEncryptedAttr()) {
            Coloring.Colors.insert(std::pair(&A, EncryptedPtrEncryptedPlainIndex));
        } else {
            Coloring.Colors.insert(std::pair(&A, PlainPtr));
        }
    }

    // Next, color each identifier for each statement.
    for (const auto &inst : instructions(F)) {
        colorInstruction(inst, Coloring);
    }

    return Coloring;
}

std::optional<EncryptionColor> EncryptionColoring::getColor(const Value* value) {
    auto val = Colors.find(value);

    return val == Colors.end() ? std::optional<EncryptionColor>() : std::optional<EncryptionColor>(val->second);
}

EncryptionColor mapPtrToRegColoring(EncryptionColor c) {
    switch (c) {
        case EncryptionColor::EncryptedPtrEncryptedIndex:
        case EncryptionColor::EncryptedPtrEncryptedPlainIndex:
            return EncryptionColor::Encrypted;
        case EncryptionColor::PlainPtr:
            return EncryptionColor::Plaintext;
        default:
            llvm_unreachable("Unexpected color");

    }
}

void EncryptionColoringAnalysis::colorInstruction(const Instruction &Inst, EncryptionColoring &Coloring) {
    if (Inst.isTerminator()) {
        return;
    }

    switch (Inst.getOpcode()) {
        case Instruction::Load: {
            auto ptrIdent = Inst.getOperand(0);
            auto ptrColor = Coloring.Colors.find(ptrIdent)->second;
            auto regColor = mapPtrToRegColoring(ptrColor);

            Coloring.Colors.insert(std::pair(&Inst, regColor));
            break;
        }
        case Instruction::Alloca: {
            Coloring.Colors.insert(std::pair(&Inst, EncryptionColor::Plaintext));
            break;
        }
        case Instruction::Ret: {
            break;
        }
        case Instruction::Store: {
            break;
        }
        default: {
            auto kind = EncryptionColor::Plaintext;

            for (auto &op : Inst.operands()) {
                if (Coloring.Colors.find(op)->second == EncryptionColor::Encrypted) {
                    kind = EncryptionColor::Encrypted;
                }
            }

            Coloring.Colors.insert(std::pair(&Inst, kind));
        }
            
    }
}

EncryptionColoring::EncryptionColoring(Function& F): Parent(&F), IsApplicable(false), Colors() {
}
