//===-- CoroInstr.h - Coroutine Intrinsics Instruction Wrappers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file defines classes that make it really easy to deal with intrinsic
// functions with the isa/dyncast family of functions.  In particular, this
// allows you to do things like:
//
//     if (auto *SF = dyn_cast<CoroSubFnInst>(Inst))
//        ... SF->getFrame() ... SF->getAlloc() ...
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
// The helpful comment above is borrowed from llvm/IntrinsicInst.h, we keep
// coroutine intrinsic wrappers here since they are only used by the passes in
// the Coroutine library.
//===----------------------------------------------------------------------===//

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"

namespace llvm {

/// This class represents the llvm.coro.subfn.addr instruction.
class LLVM_LIBRARY_VISIBILITY CoroSubFnInst : public IntrinsicInst {
  enum { FrameArg, IndexArg };

public:
  enum ResumeKind {
    ResumeIndex,
    DestroyIndex,
    IndexLast,
    IndexFirst = ResumeIndex
  };

  Value *getFrame() const { return getArgOperand(FrameArg); }
  ResumeKind getIndex() const {
    int64_t Index = getRawIndex()->getValue().getSExtValue();
    assert(Index >= IndexFirst && Index < IndexLast &&
           "unexpected CoroSubFnInst index argument");
    return static_cast<ResumeKind>(Index);
  }

  ConstantInt *getRawIndex() const {
    return cast<ConstantInt>(getArgOperand(IndexArg));
  }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_subfn_addr;
  }
  static inline bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents the llvm.coro.begin instruction.
class LLVM_LIBRARY_VISIBILITY CoroBeginInst : public IntrinsicInst {
  enum { MemArg, AlignArg, PromiseArg, InfoArg };

public:
  Constant *getRawInfo() const {
    return cast<Constant>(getArgOperand(InfoArg)->stripPointerCasts());
  }

  void setInfo(Constant *C) { setArgOperand(InfoArg, C); }

  // Info argument of coro.begin is
  //   fresh out of the frontend: null ;
  //   outlined                 : {Init, Return, Susp1, Susp2, ...} ;
  //   postsplit                : [resume, destroy, cleanup] ;
  //
  // If parts of the coroutine were outlined to protect against undesirable
  // code motion, these functions will be stored in a struct literal referred to
  // by the Info parameter. Note: this is only needed before coroutine is split.
  //
  // After coroutine is split, resume functions are stored in an array
  // referred to by this parameter.

  struct Info {
    ConstantStruct *OutlinedParts = nullptr;
    ConstantArray *Resumers = nullptr;

    bool hasOutlinedParts() const { return OutlinedParts != nullptr; }
    bool isPostSplit() const { return Resumers != nullptr; }
  };
  Info getInfo() const {
    Info Result;
    auto *GV = dyn_cast<GlobalVariable>(getRawInfo());
    if (!GV)
      return Result;

    assert(GV->isConstant() && GV->hasDefinitiveInitializer());
    Constant *Initializer = GV->getInitializer();
    if ((Result.OutlinedParts = dyn_cast<ConstantStruct>(Initializer)))
      return Result;

    Result.Resumers = cast<ConstantArray>(Initializer);
    return Result;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::coro_begin;
  }
  static inline bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

} // End namespace llvm.
