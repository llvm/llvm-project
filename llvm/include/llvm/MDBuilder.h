//===---- llvm/MDBuilder.h - Builder for LLVM metadata ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MDBuilder class, which is used as a convenient way to
// create LLVM metadata with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MDBUILDER_H
#define LLVM_MDBUILDER_H

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/APInt.h"

namespace llvm {

  class MDBuilder {
    LLVMContext &Context;

  public:
    MDBuilder(LLVMContext &context) : Context(context) {}

    /// \brief Return the given string as metadata.
    MDString *createString(StringRef Str) {
      return MDString::get(Context, Str);
    }

    //===------------------------------------------------------------------===//
    // FPMath metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata with the given settings.  The special value 0.0
    /// for the Accuracy parameter indicates the default (maximal precision)
    /// setting.
    MDNode *createFPMath(float Accuracy) {
      if (Accuracy == 0.0)
        return 0;
      assert(Accuracy > 0.0 && "Invalid fpmath accuracy!");
      Value *Op = ConstantFP::get(Type::getFloatTy(Context), Accuracy);
      return MDNode::get(Context, Op);
    }

    //===------------------------------------------------------------------===//
    // Prof metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata containing two branch weights.
    MDNode *createBranchWeights(uint32_t TrueWeight, uint32_t FalseWeight) {
      uint32_t Weights[] = { TrueWeight, FalseWeight };
      return createBranchWeights(Weights);
    }

    /// \brief Return metadata containing a number of branch weights.
    MDNode *createBranchWeights(ArrayRef<uint32_t> Weights) {
      assert(Weights.size() >= 2 && "Need at least two branch weights!");

      SmallVector<Value *, 4> Vals(Weights.size()+1);
      Vals[0] = createString("branch_weights");

      Type *Int32Ty = Type::getInt32Ty(Context);
      for (unsigned i = 0, e = Weights.size(); i != e; ++i)
        Vals[i+1] = ConstantInt::get(Int32Ty, Weights[i]);

      return MDNode::get(Context, Vals);
    }

    //===------------------------------------------------------------------===//
    // Range metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata describing the range [Lo, Hi).
    MDNode *createRange(const APInt &Lo, const APInt &Hi) {
      assert(Lo.getBitWidth() == Hi.getBitWidth() && "Mismatched bitwidths!");
      // If the range is everything then it is useless.
      if (Hi == Lo)
        return 0;

      // Return the range [Lo, Hi).
      Type *Ty = IntegerType::get(Context, Lo.getBitWidth());
      Value *Range[2] = { ConstantInt::get(Ty, Lo), ConstantInt::get(Ty, Hi) };
      return MDNode::get(Context, Range);
    }


    //===------------------------------------------------------------------===//
    // TBAA metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata appropriate for a TBAA root node.  Each returned
    /// node is distinct from all other metadata and will never be identified
    /// (uniqued) with anything else.
    MDNode *createAnonymousTBAARoot() {
      // To ensure uniqueness the root node is self-referential.
      MDNode *Dummy = MDNode::getTemporary(Context, ArrayRef<Value*>());
      MDNode *Root = MDNode::get(Context, Dummy);
      // At this point we have
      //   !0 = metadata !{}            <- dummy
      //   !1 = metadata !{metadata !0} <- root
      // Replace the dummy operand with the root node itself and delete the dummy.
      Root->replaceOperandWith(0, Root);
      MDNode::deleteTemporary(Dummy);
      // We now have
      //   !1 = metadata !{metadata !1} <- self-referential root
      return Root;
    }

    /// \brief Return metadata appropriate for a TBAA root node with the given
    /// name.  This may be identified (uniqued) with other roots with the same
    /// name.
    MDNode *createTBAARoot(StringRef Name) {
      return MDNode::get(Context, createString(Name));
    }

    /// \brief Return metadata for a non-root TBAA node with the given name,
    /// parent in the TBAA tree, and value for 'pointsToConstantMemory'.
    MDNode *createTBAANode(StringRef Name, MDNode *Parent,
                           bool isConstant = false) {
      if (isConstant) {
        Constant *Flags = ConstantInt::get(Type::getInt64Ty(Context), 1);
        Value *Ops[3] = { createString(Name), Parent, Flags };
        return MDNode::get(Context, Ops);
      } else {
        Value *Ops[2] = { createString(Name), Parent };
        return MDNode::get(Context, Ops);
      }
    }

    struct TBAAStructField {
      uint64_t Offset;
      uint64_t Size;
      MDNode *TBAA;
      TBAAStructField(uint64_t Offset, uint64_t Size, MDNode *TBAA) :
        Offset(Offset), Size(Size), TBAA(TBAA) {}
    };

    /// \brief Return metadata for a tbaa.struct node with the given
    /// struct field descriptions.
    MDNode *createTBAAStructNode(ArrayRef<TBAAStructField> Fields) {
      SmallVector<Value *, 4> Vals(Fields.size() * 3);
      Type *Int64 = IntegerType::get(Context, 64);
      for (unsigned i = 0, e = Fields.size(); i != e; ++i) {
        Vals[i * 3 + 0] = ConstantInt::get(Int64, Fields[i].Offset);
        Vals[i * 3 + 1] = ConstantInt::get(Int64, Fields[i].Size);
        Vals[i * 3 + 2] = Fields[i].TBAA;
      }
      return MDNode::get(Context, Vals);
    }

  };

} // end namespace llvm

#endif
