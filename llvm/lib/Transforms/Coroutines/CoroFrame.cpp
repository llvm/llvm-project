//===- CoroFrame.cpp - Builds and manipulates coroutine frame -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file contains classes used to discover if for a particular value
// there from sue to definition that crosses a suspend block.
//
// Using the information discovered we form a Coroutine Frame structure to
// contain those values. All uses of those values are replaced with appropriate
// GEP + load from the coroutine frame. At the point of the definition we spill
// the value into the coroutine frame.
//
// TODO: pack values tightly using liveness info.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

// TODO: Implement in future patches.
struct SpillInfo {};

// Build a struct that will keep state for an active coroutine.
//   struct f.frame {
//     ResumeFnTy ResumeFnAddr;
//     ResumeFnTy DestroyFnAddr;
//     int ResumeIndex;
//     ... promise (if present) ...
//     ... spills ...
//   };
static StructType *buildFrameType(Function &F, coro::Shape &Shape,
                                  SpillInfo &Spills) {
  LLVMContext &C = F.getContext();
  SmallString<32> Name(F.getName());
  Name.append(".Frame");
  StructType *FrameTy = StructType::create(C, Name);
  auto *FramePtrTy = FrameTy->getPointerTo();
  auto *FnTy = FunctionType::get(Type::getVoidTy(C), FramePtrTy,
                                 /*IsVarArgs=*/false);
  auto *FnPtrTy = FnTy->getPointerTo();

  if (Shape.CoroSuspends.size() > UINT32_MAX)
    report_fatal_error("Cannot handle coroutine with this many suspend points");

  SmallVector<Type *, 8> Types{FnPtrTy, FnPtrTy, Type::getInt32Ty(C)};
  // TODO: Populate from Spills.
  FrameTy->setBody(Types);

  return FrameTy;
}

// Replace all alloca and SSA values that are accessed across suspend points
// with GetElementPointer from coroutine frame + loads and stores. Create an
// AllocaSpillBB that will become the new entry block for the resume parts of
// the coroutine:
//
//    %hdl = coro.begin(...)
//    whatever
//
// becomes:
//
//    %hdl = coro.begin(...)
//    %FramePtr = bitcast i8* hdl to %f.frame*
//    br label %AllocaSpillBB
//
//  AllocaSpillBB:
//    ; geps corresponding to allocas that were moved to coroutine frame
//    br label PostSpill
//
//  PostSpill:
//    whatever
//
//
static Instruction *insertSpills(SpillInfo &Spills, coro::Shape &Shape) {
  auto *CB = Shape.CoroBegin;
  IRBuilder<> Builder(CB->getNextNode());
  PointerType *FramePtrTy = Shape.FrameTy->getPointerTo();
  auto *FramePtr =
      cast<Instruction>(Builder.CreateBitCast(CB, FramePtrTy, "FramePtr"));

  // TODO: Insert Spills.

  auto *FramePtrBB = FramePtr->getParent();
  Shape.AllocaSpillBlock =
      FramePtrBB->splitBasicBlock(FramePtr->getNextNode(), "AllocaSpillBB");
  Shape.AllocaSpillBlock->splitBasicBlock(&Shape.AllocaSpillBlock->front(),
                                          "PostSpill");
  // TODO: Insert geps for alloca moved to coroutine frame.

  return FramePtr;
}

void coro::buildCoroutineFrame(Function &F, Shape &Shape) {
  SpillInfo Spills;
  // TODO: Compute Spills (incoming in later patches).

  Shape.FrameTy = buildFrameType(F, Shape, Spills);
  Shape.FramePtr = insertSpills(Spills, Shape);
}
