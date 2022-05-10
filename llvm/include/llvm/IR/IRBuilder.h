//===- llvm/IRBuilder.h - Builder for LLVM Instructions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_IRBUILDER_H
#define LLVM_IR_IRBUILDER_H

#include "llvm-c/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantFolder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <utility>

namespace llvm {

class APInt;
class Use;

/// This provides the default implementation of the IRBuilder
/// 'InsertHelper' method that is called whenever an instruction is created by
/// IRBuilder and needs to be inserted.
///
/// By default, this inserts the instruction at the insertion point.
class IRBuilderDefaultInserter {
public:
  virtual ~IRBuilderDefaultInserter();

  virtual void InsertHelper(Instruction *I, const Twine &Name,
                            BasicBlock *BB,
                            BasicBlock::iterator InsertPt) const {
    if (BB) BB->getInstList().insert(InsertPt, I);
    I->setName(Name);
  }
};

/// Provides an 'InsertHelper' that calls a user-provided callback after
/// performing the default insertion.
class IRBuilderCallbackInserter : public IRBuilderDefaultInserter {
  std::function<void(Instruction *)> Callback;

public:
  ~IRBuilderCallbackInserter() override;

  IRBuilderCallbackInserter(std::function<void(Instruction *)> Callback)
      : Callback(std::move(Callback)) {}

  void InsertHelper(Instruction *I, const Twine &Name,
                    BasicBlock *BB,
                    BasicBlock::iterator InsertPt) const override {
    IRBuilderDefaultInserter::InsertHelper(I, Name, BB, InsertPt);
    Callback(I);
  }
};

/// Common base class shared among various IRBuilders.
class IRBuilderBase {
  /// Pairs of (metadata kind, MDNode *) that should be added to all newly
  /// created instructions, like !dbg metadata.
  SmallVector<std::pair<unsigned, MDNode *>, 2> MetadataToCopy;

  /// Add or update the an entry (Kind, MD) to MetadataToCopy, if \p MD is not
  /// null. If \p MD is null, remove the entry with \p Kind.
  void AddOrRemoveMetadataToCopy(unsigned Kind, MDNode *MD) {
    if (!MD) {
      erase_if(MetadataToCopy, [Kind](const std::pair<unsigned, MDNode *> &KV) {
        return KV.first == Kind;
      });
      return;
    }

    for (auto &KV : MetadataToCopy)
      if (KV.first == Kind) {
        KV.second = MD;
        return;
      }

    MetadataToCopy.emplace_back(Kind, MD);
  }

protected:
  BasicBlock *BB;
  BasicBlock::iterator InsertPt;
  LLVMContext &Context;
  const IRBuilderFolder &Folder;
  const IRBuilderDefaultInserter &Inserter;

  MDNode *DefaultFPMathTag;
  FastMathFlags FMF;

  bool IsFPConstrained = false;
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  RoundingMode DefaultConstrainedRounding = RoundingMode::Dynamic;

  ArrayRef<OperandBundleDef> DefaultOperandBundles;

public:
  IRBuilderBase(LLVMContext &context, const IRBuilderFolder &Folder,
                const IRBuilderDefaultInserter &Inserter, MDNode *FPMathTag,
                ArrayRef<OperandBundleDef> OpBundles)
      : Context(context), Folder(Folder), Inserter(Inserter),
        DefaultFPMathTag(FPMathTag), DefaultOperandBundles(OpBundles) {
    ClearInsertionPoint();
  }

  /// Insert and return the specified instruction.
  template<typename InstTy>
  InstTy *Insert(InstTy *I, const Twine &Name = "") const {
    Inserter.InsertHelper(I, Name, BB, InsertPt);
    AddMetadataToInst(I);
    return I;
  }

  /// No-op overload to handle constants.
  Constant *Insert(Constant *C, const Twine& = "") const {
    return C;
  }

  Value *Insert(Value *V, const Twine &Name = "") const {
    if (Instruction *I = dyn_cast<Instruction>(V))
      return Insert(I, Name);
    assert(isa<Constant>(V));
    return V;
  }

  //===--------------------------------------------------------------------===//
  // Builder configuration methods
  //===--------------------------------------------------------------------===//

  /// Clear the insertion point: created instructions will not be
  /// inserted into a block.
  void ClearInsertionPoint() {
    BB = nullptr;
    InsertPt = BasicBlock::iterator();
  }

  BasicBlock *GetInsertBlock() const { return BB; }
  BasicBlock::iterator GetInsertPoint() const { return InsertPt; }
  LLVMContext &getContext() const { return Context; }

  /// This specifies that created instructions should be appended to the
  /// end of the specified block.
  void SetInsertPoint(BasicBlock *TheBB) {
    BB = TheBB;
    InsertPt = BB->end();
  }

  /// This specifies that created instructions should be inserted before
  /// the specified instruction.
  void SetInsertPoint(Instruction *I) {
    BB = I->getParent();
    InsertPt = I->getIterator();
    assert(InsertPt != BB->end() && "Can't read debug loc from end()");
    SetCurrentDebugLocation(I->getDebugLoc());
  }

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  void SetInsertPoint(BasicBlock *TheBB, BasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
    if (IP != TheBB->end())
      SetCurrentDebugLocation(IP->getDebugLoc());
  }

  /// Set location information used by debugging information.
  void SetCurrentDebugLocation(DebugLoc L) {
    AddOrRemoveMetadataToCopy(LLVMContext::MD_dbg, L.getAsMDNode());
  }

  /// Collect metadata with IDs \p MetadataKinds from \p Src which should be
  /// added to all created instructions. Entries present in MedataDataToCopy but
  /// not on \p Src will be dropped from MetadataToCopy.
  void CollectMetadataToCopy(Instruction *Src,
                             ArrayRef<unsigned> MetadataKinds) {
    for (unsigned K : MetadataKinds)
      AddOrRemoveMetadataToCopy(K, Src->getMetadata(K));
  }

  /// Get location information used by debugging information.
  DebugLoc getCurrentDebugLocation() const;

  /// If this builder has a current debug location, set it on the
  /// specified instruction.
  void SetInstDebugLocation(Instruction *I) const;

  /// Add all entries in MetadataToCopy to \p I.
  void AddMetadataToInst(Instruction *I) const {
    for (auto &KV : MetadataToCopy)
      I->setMetadata(KV.first, KV.second);
  }

  /// Get the return type of the current function that we're emitting
  /// into.
  Type *getCurrentFunctionReturnType() const;

  /// InsertPoint - A saved insertion point.
  class InsertPoint {
    BasicBlock *Block = nullptr;
    BasicBlock::iterator Point;

  public:
    /// Creates a new insertion point which doesn't point to anything.
    InsertPoint() = default;

    /// Creates a new insertion point at the given location.
    InsertPoint(BasicBlock *InsertBlock, BasicBlock::iterator InsertPoint)
        : Block(InsertBlock), Point(InsertPoint) {}

    /// Returns true if this insert point is set.
    bool isSet() const { return (Block != nullptr); }

    BasicBlock *getBlock() const { return Block; }
    BasicBlock::iterator getPoint() const { return Point; }
  };

  /// Returns the current insert point.
  InsertPoint saveIP() const {
    return InsertPoint(GetInsertBlock(), GetInsertPoint());
  }

  /// Returns the current insert point, clearing it in the process.
  InsertPoint saveAndClearIP() {
    InsertPoint IP(GetInsertBlock(), GetInsertPoint());
    ClearInsertionPoint();
    return IP;
  }

  /// Sets the current insert point to a previously-saved location.
  void restoreIP(InsertPoint IP) {
    if (IP.isSet())
      SetInsertPoint(IP.getBlock(), IP.getPoint());
    else
      ClearInsertionPoint();
  }

  /// Get the floating point math metadata being used.
  MDNode *getDefaultFPMathTag() const { return DefaultFPMathTag; }

  /// Get the flags to be applied to created floating point ops
  FastMathFlags getFastMathFlags() const { return FMF; }

  FastMathFlags &getFastMathFlags() { return FMF; }

  /// Clear the fast-math flags.
  void clearFastMathFlags() { FMF.clear(); }

  /// Set the floating point math metadata to be used.
  void setDefaultFPMathTag(MDNode *FPMathTag) { DefaultFPMathTag = FPMathTag; }

  /// Set the fast-math flags to be used with generated fp-math operators
  void setFastMathFlags(FastMathFlags NewFMF) { FMF = NewFMF; }

  /// Enable/Disable use of constrained floating point math. When
  /// enabled the CreateF<op>() calls instead create constrained
  /// floating point intrinsic calls. Fast math flags are unaffected
  /// by this setting.
  void setIsFPConstrained(bool IsCon) { IsFPConstrained = IsCon; }

  /// Query for the use of constrained floating point math
  bool getIsFPConstrained() { return IsFPConstrained; }

  /// Set the exception handling to be used with constrained floating point
  void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    Optional<StringRef> ExceptStr = convertExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr.hasValue() && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }

  /// Set the rounding mode handling to be used with constrained floating point
  void setDefaultConstrainedRounding(RoundingMode NewRounding) {
#ifndef NDEBUG
    Optional<StringRef> RoundingStr = convertRoundingModeToStr(NewRounding);
    assert(RoundingStr.hasValue() && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }

  /// Get the exception handling used with constrained floating point
  fp::ExceptionBehavior getDefaultConstrainedExcept() {
    return DefaultConstrainedExcept;
  }

  /// Get the rounding mode handling used with constrained floating point
  RoundingMode getDefaultConstrainedRounding() {
    return DefaultConstrainedRounding;
  }

  void setConstrainedFPFunctionAttr() {
    assert(BB && "Must have a basic block to set any function attributes!");

    Function *F = BB->getParent();
    if (!F->hasFnAttribute(Attribute::StrictFP)) {
      F->addFnAttr(Attribute::StrictFP);
    }
  }

  void setConstrainedFPCallAttr(CallBase *I) {
    I->addFnAttr(Attribute::StrictFP);
  }

  void setDefaultOperandBundles(ArrayRef<OperandBundleDef> OpBundles) {
    DefaultOperandBundles = OpBundles;
  }

  //===--------------------------------------------------------------------===//
  // RAII helpers.
  //===--------------------------------------------------------------------===//

  // RAII object that stores the current insertion point and restores it
  // when the object is destroyed. This includes the debug location.
  class InsertPointGuard {
    IRBuilderBase &Builder;
    AssertingVH<BasicBlock> Block;
    BasicBlock::iterator Point;
    DebugLoc DbgLoc;

  public:
    InsertPointGuard(IRBuilderBase &B)
        : Builder(B), Block(B.GetInsertBlock()), Point(B.GetInsertPoint()),
          DbgLoc(B.getCurrentDebugLocation()) {}

    InsertPointGuard(const InsertPointGuard &) = delete;
    InsertPointGuard &operator=(const InsertPointGuard &) = delete;

    ~InsertPointGuard() {
      Builder.restoreIP(InsertPoint(Block, Point));
      Builder.SetCurrentDebugLocation(DbgLoc);
    }
  };

  // RAII object that stores the current fast math settings and restores
  // them when the object is destroyed.
  class FastMathFlagGuard {
    IRBuilderBase &Builder;
    FastMathFlags FMF;
    MDNode *FPMathTag;
    bool IsFPConstrained;
    fp::ExceptionBehavior DefaultConstrainedExcept;
    RoundingMode DefaultConstrainedRounding;

  public:
    FastMathFlagGuard(IRBuilderBase &B)
        : Builder(B), FMF(B.FMF), FPMathTag(B.DefaultFPMathTag),
          IsFPConstrained(B.IsFPConstrained),
          DefaultConstrainedExcept(B.DefaultConstrainedExcept),
          DefaultConstrainedRounding(B.DefaultConstrainedRounding) {}

    FastMathFlagGuard(const FastMathFlagGuard &) = delete;
    FastMathFlagGuard &operator=(const FastMathFlagGuard &) = delete;

    ~FastMathFlagGuard() {
      Builder.FMF = FMF;
      Builder.DefaultFPMathTag = FPMathTag;
      Builder.IsFPConstrained = IsFPConstrained;
      Builder.DefaultConstrainedExcept = DefaultConstrainedExcept;
      Builder.DefaultConstrainedRounding = DefaultConstrainedRounding;
    }
  };

  // RAII object that stores the current default operand bundles and restores
  // them when the object is destroyed.
  class OperandBundlesGuard {
    IRBuilderBase &Builder;
    ArrayRef<OperandBundleDef> DefaultOperandBundles;

  public:
    OperandBundlesGuard(IRBuilderBase &B)
        : Builder(B), DefaultOperandBundles(B.DefaultOperandBundles) {}

    OperandBundlesGuard(const OperandBundlesGuard &) = delete;
    OperandBundlesGuard &operator=(const OperandBundlesGuard &) = delete;

    ~OperandBundlesGuard() {
      Builder.DefaultOperandBundles = DefaultOperandBundles;
    }
  };


  //===--------------------------------------------------------------------===//
  // Miscellaneous creation methods.
  //===--------------------------------------------------------------------===//

  /// Make a new global variable with initializer type i8*
  ///
  /// Make a new global variable with an initializer that has array of i8 type
  /// filled in with the null terminated string value specified.  The new global
  /// variable will be marked mergable with any others of the same contents.  If
  /// Name is specified, it is the name of the global variable created.
  ///
  /// If no module is given via \p M, it is take from the insertion point basic
  /// block.
  GlobalVariable *CreateGlobalString(StringRef Str, const Twine &Name = "",
                                     unsigned AddressSpace = 0,
                                     Module *M = nullptr);

  /// Get a constant value representing either true or false.
  ConstantInt *getInt1(bool V) {
    return ConstantInt::get(getInt1Ty(), V);
  }

  /// Get the constant value for i1 true.
  ConstantInt *getTrue() {
    return ConstantInt::getTrue(Context);
  }

  /// Get the constant value for i1 false.
  ConstantInt *getFalse() {
    return ConstantInt::getFalse(Context);
  }

  /// Get a constant 8-bit value.
  ConstantInt *getInt8(uint8_t C) {
    return ConstantInt::get(getInt8Ty(), C);
  }

  /// Get a constant 16-bit value.
  ConstantInt *getInt16(uint16_t C) {
    return ConstantInt::get(getInt16Ty(), C);
  }

  /// Get a constant 32-bit value.
  ConstantInt *getInt32(uint32_t C) {
    return ConstantInt::get(getInt32Ty(), C);
  }

  /// Get a constant 64-bit value.
  ConstantInt *getInt64(uint64_t C) {
    return ConstantInt::get(getInt64Ty(), C);
  }

  /// Get a constant N-bit value, zero extended or truncated from
  /// a 64-bit value.
  ConstantInt *getIntN(unsigned N, uint64_t C) {
    return ConstantInt::get(getIntNTy(N), C);
  }

  /// Get a constant integer value.
  ConstantInt *getInt(const APInt &AI) {
    return ConstantInt::get(Context, AI);
  }

  //===--------------------------------------------------------------------===//
  // Type creation methods
  //===--------------------------------------------------------------------===//

  /// Fetch the type representing a single bit
  IntegerType *getInt1Ty() {
    return Type::getInt1Ty(Context);
  }

  /// Fetch the type representing an 8-bit integer.
  IntegerType *getInt8Ty() {
    return Type::getInt8Ty(Context);
  }

  /// Fetch the type representing a 16-bit integer.
  IntegerType *getInt16Ty() {
    return Type::getInt16Ty(Context);
  }

  /// Fetch the type representing a 32-bit integer.
  IntegerType *getInt32Ty() {
    return Type::getInt32Ty(Context);
  }

  /// Fetch the type representing a 64-bit integer.
  IntegerType *getInt64Ty() {
    return Type::getInt64Ty(Context);
  }

  /// Fetch the type representing a 128-bit integer.
  IntegerType *getInt128Ty() { return Type::getInt128Ty(Context); }

  /// Fetch the type representing an N-bit integer.
  IntegerType *getIntNTy(unsigned N) {
    return Type::getIntNTy(Context, N);
  }

  /// Fetch the type representing a 16-bit floating point value.
  Type *getHalfTy() {
    return Type::getHalfTy(Context);
  }

  /// Fetch the type representing a 16-bit brain floating point value.
  Type *getBFloatTy() {
    return Type::getBFloatTy(Context);
  }

  /// Fetch the type representing a 32-bit floating point value.
  Type *getFloatTy() {
    return Type::getFloatTy(Context);
  }

  /// Fetch the type representing a 64-bit floating point value.
  Type *getDoubleTy() {
    return Type::getDoubleTy(Context);
  }

  /// Fetch the type representing void.
  Type *getVoidTy() {
    return Type::getVoidTy(Context);
  }

  /// Fetch the type representing a pointer.
  PointerType *getPtrTy(unsigned AddrSpace = 0) {
    return PointerType::get(Context, AddrSpace);
  }

  /// Fetch the type representing a pointer to an 8-bit integer value.
  PointerType *getInt8PtrTy(unsigned AddrSpace = 0) {
    return Type::getInt8PtrTy(Context, AddrSpace);
  }

  /// Fetch the type representing a pointer to an integer value.
  IntegerType *getIntPtrTy(const DataLayout &DL, unsigned AddrSpace = 0) {
    return DL.getIntPtrType(Context, AddrSpace);
  }

  //===--------------------------------------------------------------------===//
  // Intrinsic creation methods
  //===--------------------------------------------------------------------===//

  /// Create and insert a memset to the specified pointer and the
  /// specified value.
  ///
  /// If the pointer isn't an i8*, it will be converted. If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst *CreateMemSet(Value *Ptr, Value *Val, uint64_t Size,
                         MaybeAlign Align, bool isVolatile = false,
                         MDNode *TBAATag = nullptr, MDNode *ScopeTag = nullptr,
                         MDNode *NoAliasTag = nullptr) {
    return CreateMemSet(Ptr, Val, getInt64(Size), Align, isVolatile,
                        TBAATag, ScopeTag, NoAliasTag);
  }

  CallInst *CreateMemSet(Value *Ptr, Value *Val, Value *Size, MaybeAlign Align,
                         bool isVolatile = false, MDNode *TBAATag = nullptr,
                         MDNode *ScopeTag = nullptr,
                         MDNode *NoAliasTag = nullptr);

  /// Create and insert an element unordered-atomic memset of the region of
  /// memory starting at the given pointer to the given value.
  ///
  /// If the pointer isn't an i8*, it will be converted. If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst *CreateElementUnorderedAtomicMemSet(Value *Ptr, Value *Val,
                                               uint64_t Size, Align Alignment,
                                               uint32_t ElementSize,
                                               MDNode *TBAATag = nullptr,
                                               MDNode *ScopeTag = nullptr,
                                               MDNode *NoAliasTag = nullptr) {
    return CreateElementUnorderedAtomicMemSet(Ptr, Val, getInt64(Size),
                                              Align(Alignment), ElementSize,
                                              TBAATag, ScopeTag, NoAliasTag);
  }

  CallInst *CreateElementUnorderedAtomicMemSet(Value *Ptr, Value *Val,
                                               Value *Size, Align Alignment,
                                               uint32_t ElementSize,
                                               MDNode *TBAATag = nullptr,
                                               MDNode *ScopeTag = nullptr,
                                               MDNode *NoAliasTag = nullptr);

  /// Create and insert a memcpy between the specified pointers.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst *CreateMemCpy(Value *Dst, MaybeAlign DstAlign, Value *Src,
                         MaybeAlign SrcAlign, uint64_t Size,
                         bool isVolatile = false, MDNode *TBAATag = nullptr,
                         MDNode *TBAAStructTag = nullptr,
                         MDNode *ScopeTag = nullptr,
                         MDNode *NoAliasTag = nullptr) {
    return CreateMemCpy(Dst, DstAlign, Src, SrcAlign, getInt64(Size),
                        isVolatile, TBAATag, TBAAStructTag, ScopeTag,
                        NoAliasTag);
  }

  CallInst *CreateMemTransferInst(
      Intrinsic::ID IntrID, Value *Dst, MaybeAlign DstAlign, Value *Src,
      MaybeAlign SrcAlign, Value *Size, bool isVolatile = false,
      MDNode *TBAATag = nullptr, MDNode *TBAAStructTag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr);

  CallInst *CreateMemCpy(Value *Dst, MaybeAlign DstAlign, Value *Src,
                         MaybeAlign SrcAlign, Value *Size,
                         bool isVolatile = false, MDNode *TBAATag = nullptr,
                         MDNode *TBAAStructTag = nullptr,
                         MDNode *ScopeTag = nullptr,
                         MDNode *NoAliasTag = nullptr) {
    return CreateMemTransferInst(Intrinsic::memcpy, Dst, DstAlign, Src,
                                 SrcAlign, Size, isVolatile, TBAATag,
                                 TBAAStructTag, ScopeTag, NoAliasTag);
  }

  CallInst *
  CreateMemCpyInline(Value *Dst, MaybeAlign DstAlign, Value *Src,
                     MaybeAlign SrcAlign, Value *Size, bool IsVolatile = false,
                     MDNode *TBAATag = nullptr, MDNode *TBAAStructTag = nullptr,
                     MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr);

  /// Create and insert an element unordered-atomic memcpy between the
  /// specified pointers.
  ///
  /// DstAlign/SrcAlign are the alignments of the Dst/Src pointers, respectively.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst *CreateElementUnorderedAtomicMemCpy(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, Value *Size,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr);

  CallInst *CreateMemMove(Value *Dst, MaybeAlign DstAlign, Value *Src,
                          MaybeAlign SrcAlign, uint64_t Size,
                          bool isVolatile = false, MDNode *TBAATag = nullptr,
                          MDNode *ScopeTag = nullptr,
                          MDNode *NoAliasTag = nullptr) {
    return CreateMemMove(Dst, DstAlign, Src, SrcAlign, getInt64(Size),
                         isVolatile, TBAATag, ScopeTag, NoAliasTag);
  }

  CallInst *CreateMemMove(Value *Dst, MaybeAlign DstAlign, Value *Src,
                          MaybeAlign SrcAlign, Value *Size,
                          bool isVolatile = false, MDNode *TBAATag = nullptr,
                          MDNode *ScopeTag = nullptr,
                          MDNode *NoAliasTag = nullptr);

  /// \brief Create and insert an element unordered-atomic memmove between the
  /// specified pointers.
  ///
  /// DstAlign/SrcAlign are the alignments of the Dst/Src pointers,
  /// respectively.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst *CreateElementUnorderedAtomicMemMove(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, Value *Size,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr);

  /// Create a sequential vector fadd reduction intrinsic of the source vector.
  /// The first parameter is a scalar accumulator value. An unordered reduction
  /// can be created by adding the reassoc fast-math flag to the resulting
  /// sequential reduction.
  CallInst *CreateFAddReduce(Value *Acc, Value *Src);

  /// Create a sequential vector fmul reduction intrinsic of the source vector.
  /// The first parameter is a scalar accumulator value. An unordered reduction
  /// can be created by adding the reassoc fast-math flag to the resulting
  /// sequential reduction.
  CallInst *CreateFMulReduce(Value *Acc, Value *Src);

  /// Create a vector int add reduction intrinsic of the source vector.
  CallInst *CreateAddReduce(Value *Src);

  /// Create a vector int mul reduction intrinsic of the source vector.
  CallInst *CreateMulReduce(Value *Src);

  /// Create a vector int AND reduction intrinsic of the source vector.
  CallInst *CreateAndReduce(Value *Src);

  /// Create a vector int OR reduction intrinsic of the source vector.
  CallInst *CreateOrReduce(Value *Src);

  /// Create a vector int XOR reduction intrinsic of the source vector.
  CallInst *CreateXorReduce(Value *Src);

  /// Create a vector integer max reduction intrinsic of the source
  /// vector.
  CallInst *CreateIntMaxReduce(Value *Src, bool IsSigned = false);

  /// Create a vector integer min reduction intrinsic of the source
  /// vector.
  CallInst *CreateIntMinReduce(Value *Src, bool IsSigned = false);

  /// Create a vector float max reduction intrinsic of the source
  /// vector.
  CallInst *CreateFPMaxReduce(Value *Src);

  /// Create a vector float min reduction intrinsic of the source
  /// vector.
  CallInst *CreateFPMinReduce(Value *Src);

  /// Create a lifetime.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst *CreateLifetimeStart(Value *Ptr, ConstantInt *Size = nullptr);

  /// Create a lifetime.end intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst *CreateLifetimeEnd(Value *Ptr, ConstantInt *Size = nullptr);

  /// Create a call to invariant.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst *CreateInvariantStart(Value *Ptr, ConstantInt *Size = nullptr);

  /// Create a call to Masked Load intrinsic
  CallInst *CreateMaskedLoad(Type *Ty, Value *Ptr, Align Alignment, Value *Mask,
                             Value *PassThru = nullptr, const Twine &Name = "");

  /// Create a call to Masked Store intrinsic
  CallInst *CreateMaskedStore(Value *Val, Value *Ptr, Align Alignment,
                              Value *Mask);

  /// Create a call to Masked Gather intrinsic
  CallInst *CreateMaskedGather(Type *Ty, Value *Ptrs, Align Alignment,
                               Value *Mask = nullptr, Value *PassThru = nullptr,
                               const Twine &Name = "");

  /// Create a call to Masked Scatter intrinsic
  CallInst *CreateMaskedScatter(Value *Val, Value *Ptrs, Align Alignment,
                                Value *Mask = nullptr);

  /// Create an assume intrinsic call that allows the optimizer to
  /// assume that the provided condition will be true.
  ///
  /// The optional argument \p OpBundles specifies operand bundles that are
  /// added to the call instruction.
  CallInst *CreateAssumption(Value *Cond,
                             ArrayRef<OperandBundleDef> OpBundles = llvm::None);

  /// Create a llvm.experimental.noalias.scope.decl intrinsic call.
  Instruction *CreateNoAliasScopeDeclaration(Value *Scope);
  Instruction *CreateNoAliasScopeDeclaration(MDNode *ScopeTag) {
    return CreateNoAliasScopeDeclaration(
        MetadataAsValue::get(Context, ScopeTag));
  }

  /// Create a call to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  CallInst *CreateGCStatepointCall(uint64_t ID, uint32_t NumPatchBytes,
                                   FunctionCallee ActualCallee,
                                   ArrayRef<Value *> CallArgs,
                                   Optional<ArrayRef<Value *>> DeoptArgs,
                                   ArrayRef<Value *> GCArgs,
                                   const Twine &Name = "");

  /// Create a call to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  CallInst *CreateGCStatepointCall(uint64_t ID, uint32_t NumPatchBytes,
                                   FunctionCallee ActualCallee, uint32_t Flags,
                                   ArrayRef<Value *> CallArgs,
                                   Optional<ArrayRef<Use>> TransitionArgs,
                                   Optional<ArrayRef<Use>> DeoptArgs,
                                   ArrayRef<Value *> GCArgs,
                                   const Twine &Name = "");

  /// Conveninence function for the common case when CallArgs are filled
  /// in using makeArrayRef(CS.arg_begin(), CS.arg_end()); Use needs to be
  /// .get()'ed to get the Value pointer.
  CallInst *CreateGCStatepointCall(uint64_t ID, uint32_t NumPatchBytes,
                                   FunctionCallee ActualCallee,
                                   ArrayRef<Use> CallArgs,
                                   Optional<ArrayRef<Value *>> DeoptArgs,
                                   ArrayRef<Value *> GCArgs,
                                   const Twine &Name = "");

  /// Create an invoke to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  InvokeInst *
  CreateGCStatepointInvoke(uint64_t ID, uint32_t NumPatchBytes,
                           FunctionCallee ActualInvokee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, ArrayRef<Value *> InvokeArgs,
                           Optional<ArrayRef<Value *>> DeoptArgs,
                           ArrayRef<Value *> GCArgs, const Twine &Name = "");

  /// Create an invoke to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  InvokeInst *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, uint32_t Flags,
      ArrayRef<Value *> InvokeArgs, Optional<ArrayRef<Use>> TransitionArgs,
      Optional<ArrayRef<Use>> DeoptArgs, ArrayRef<Value *> GCArgs,
      const Twine &Name = "");

  // Convenience function for the common case when CallArgs are filled in using
  // makeArrayRef(CS.arg_begin(), CS.arg_end()); Use needs to be .get()'ed to
  // get the Value *.
  InvokeInst *
  CreateGCStatepointInvoke(uint64_t ID, uint32_t NumPatchBytes,
                           FunctionCallee ActualInvokee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, ArrayRef<Use> InvokeArgs,
                           Optional<ArrayRef<Value *>> DeoptArgs,
                           ArrayRef<Value *> GCArgs, const Twine &Name = "");

  /// Create a call to the experimental.gc.result intrinsic to extract
  /// the result from a call wrapped in a statepoint.
  CallInst *CreateGCResult(Instruction *Statepoint,
                           Type *ResultType,
                           const Twine &Name = "");

  /// Create a call to the experimental.gc.relocate intrinsics to
  /// project the relocated value of one pointer from the statepoint.
  CallInst *CreateGCRelocate(Instruction *Statepoint,
                             int BaseOffset,
                             int DerivedOffset,
                             Type *ResultType,
                             const Twine &Name = "");

  /// Create a call to the experimental.gc.pointer.base intrinsic to get the
  /// base pointer for the specified derived pointer.
  CallInst *CreateGCGetPointerBase(Value *DerivedPtr, const Twine &Name = "");

  /// Create a call to the experimental.gc.get.pointer.offset intrinsic to get
  /// the offset of the specified derived pointer from its base.
  CallInst *CreateGCGetPointerOffset(Value *DerivedPtr, const Twine &Name = "");

  /// Create a call to llvm.vscale, multiplied by \p Scaling. The type of VScale
  /// will be the same type as that of \p Scaling.
  Value *CreateVScale(Constant *Scaling, const Twine &Name = "");

  /// Creates a vector of type \p DstType with the linear sequence <0, 1, ...>
  Value *CreateStepVector(Type *DstType, const Twine &Name = "");

  /// Create a call to intrinsic \p ID with 1 operand which is mangled on its
  /// type.
  CallInst *CreateUnaryIntrinsic(Intrinsic::ID ID, Value *V,
                                 Instruction *FMFSource = nullptr,
                                 const Twine &Name = "");

  /// Create a call to intrinsic \p ID with 2 operands which is mangled on the
  /// first type.
  CallInst *CreateBinaryIntrinsic(Intrinsic::ID ID, Value *LHS, Value *RHS,
                                  Instruction *FMFSource = nullptr,
                                  const Twine &Name = "");

  /// Create a call to intrinsic \p ID with \p args, mangled using \p Types. If
  /// \p FMFSource is provided, copy fast-math-flags from that instruction to
  /// the intrinsic.
  CallInst *CreateIntrinsic(Intrinsic::ID ID, ArrayRef<Type *> Types,
                            ArrayRef<Value *> Args,
                            Instruction *FMFSource = nullptr,
                            const Twine &Name = "");

  /// Create call to the minnum intrinsic.
  CallInst *CreateMinNum(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateBinaryIntrinsic(Intrinsic::minnum, LHS, RHS, nullptr, Name);
  }

  /// Create call to the maxnum intrinsic.
  CallInst *CreateMaxNum(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateBinaryIntrinsic(Intrinsic::maxnum, LHS, RHS, nullptr, Name);
  }

  /// Create call to the minimum intrinsic.
  CallInst *CreateMinimum(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateBinaryIntrinsic(Intrinsic::minimum, LHS, RHS, nullptr, Name);
  }

  /// Create call to the maximum intrinsic.
  CallInst *CreateMaximum(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateBinaryIntrinsic(Intrinsic::maximum, LHS, RHS, nullptr, Name);
  }

  /// Create a call to the arithmetic_fence intrinsic.
  CallInst *CreateArithmeticFence(Value *Val, Type *DstType,
                                  const Twine &Name = "") {
    return CreateIntrinsic(Intrinsic::arithmetic_fence, DstType, Val, nullptr,
                           Name);
  }

  /// Create a call to the experimental.vector.extract intrinsic.
  CallInst *CreateExtractVector(Type *DstType, Value *SrcVec, Value *Idx,
                                const Twine &Name = "") {
    return CreateIntrinsic(Intrinsic::experimental_vector_extract,
                           {DstType, SrcVec->getType()}, {SrcVec, Idx}, nullptr,
                           Name);
  }

  /// Create a call to the experimental.vector.insert intrinsic.
  CallInst *CreateInsertVector(Type *DstType, Value *SrcVec, Value *SubVec,
                               Value *Idx, const Twine &Name = "") {
    return CreateIntrinsic(Intrinsic::experimental_vector_insert,
                           {DstType, SubVec->getType()}, {SrcVec, SubVec, Idx},
                           nullptr, Name);
  }

private:
  /// Create a call to a masked intrinsic with given Id.
  CallInst *CreateMaskedIntrinsic(Intrinsic::ID Id, ArrayRef<Value *> Ops,
                                  ArrayRef<Type *> OverloadedTypes,
                                  const Twine &Name = "");

  Value *getCastedInt8PtrValue(Value *Ptr);

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Terminators
  //===--------------------------------------------------------------------===//

private:
  /// Helper to add branch weight and unpredictable metadata onto an
  /// instruction.
  /// \returns The annotated instruction.
  template <typename InstTy>
  InstTy *addBranchMetadata(InstTy *I, MDNode *Weights, MDNode *Unpredictable) {
    if (Weights)
      I->setMetadata(LLVMContext::MD_prof, Weights);
    if (Unpredictable)
      I->setMetadata(LLVMContext::MD_unpredictable, Unpredictable);
    return I;
  }

public:
  /// Create a 'ret void' instruction.
  ReturnInst *CreateRetVoid() {
    return Insert(ReturnInst::Create(Context));
  }

  /// Create a 'ret <val>' instruction.
  ReturnInst *CreateRet(Value *V) {
    return Insert(ReturnInst::Create(Context, V));
  }

  /// Create a sequence of N insertvalue instructions,
  /// with one Value from the retVals array each, that build a aggregate
  /// return value one value at a time, and a ret instruction to return
  /// the resulting aggregate value.
  ///
  /// This is a convenience function for code that uses aggregate return values
  /// as a vehicle for having multiple return values.
  ReturnInst *CreateAggregateRet(Value *const *retVals, unsigned N) {
    Value *V = UndefValue::get(getCurrentFunctionReturnType());
    for (unsigned i = 0; i != N; ++i)
      V = CreateInsertValue(V, retVals[i], i, "mrv");
    return Insert(ReturnInst::Create(Context, V));
  }

  /// Create an unconditional 'br label X' instruction.
  BranchInst *CreateBr(BasicBlock *Dest) {
    return Insert(BranchInst::Create(Dest));
  }

  /// Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  BranchInst *CreateCondBr(Value *Cond, BasicBlock *True, BasicBlock *False,
                           MDNode *BranchWeights = nullptr,
                           MDNode *Unpredictable = nullptr) {
    return Insert(addBranchMetadata(BranchInst::Create(True, False, Cond),
                                    BranchWeights, Unpredictable));
  }

  /// Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction. Copy branch meta data if available.
  BranchInst *CreateCondBr(Value *Cond, BasicBlock *True, BasicBlock *False,
                           Instruction *MDSrc) {
    BranchInst *Br = BranchInst::Create(True, False, Cond);
    if (MDSrc) {
      unsigned WL[4] = {LLVMContext::MD_prof, LLVMContext::MD_unpredictable,
                        LLVMContext::MD_make_implicit, LLVMContext::MD_dbg};
      Br->copyMetadata(*MDSrc, makeArrayRef(&WL[0], 4));
    }
    return Insert(Br);
  }

  /// Create a switch instruction with the specified value, default dest,
  /// and with a hint for the number of cases that will be added (for efficient
  /// allocation).
  SwitchInst *CreateSwitch(Value *V, BasicBlock *Dest, unsigned NumCases = 10,
                           MDNode *BranchWeights = nullptr,
                           MDNode *Unpredictable = nullptr) {
    return Insert(addBranchMetadata(SwitchInst::Create(V, Dest, NumCases),
                                    BranchWeights, Unpredictable));
  }

  /// Create an indirect branch instruction with the specified address
  /// operand, with an optional hint for the number of destinations that will be
  /// added (for efficient allocation).
  IndirectBrInst *CreateIndirectBr(Value *Addr, unsigned NumDests = 10) {
    return Insert(IndirectBrInst::Create(Addr, NumDests));
  }

  /// Create an invoke instruction.
  InvokeInst *CreateInvoke(FunctionType *Ty, Value *Callee,
                           BasicBlock *NormalDest, BasicBlock *UnwindDest,
                           ArrayRef<Value *> Args,
                           ArrayRef<OperandBundleDef> OpBundles,
                           const Twine &Name = "") {
    InvokeInst *II =
        InvokeInst::Create(Ty, Callee, NormalDest, UnwindDest, Args, OpBundles);
    if (IsFPConstrained)
      setConstrainedFPCallAttr(II);
    return Insert(II, Name);
  }
  InvokeInst *CreateInvoke(FunctionType *Ty, Value *Callee,
                           BasicBlock *NormalDest, BasicBlock *UnwindDest,
                           ArrayRef<Value *> Args = None,
                           const Twine &Name = "") {
    InvokeInst *II =
        InvokeInst::Create(Ty, Callee, NormalDest, UnwindDest, Args);
    if (IsFPConstrained)
      setConstrainedFPCallAttr(II);
    return Insert(II, Name);
  }

  InvokeInst *CreateInvoke(FunctionCallee Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest, ArrayRef<Value *> Args,
                           ArrayRef<OperandBundleDef> OpBundles,
                           const Twine &Name = "") {
    return CreateInvoke(Callee.getFunctionType(), Callee.getCallee(),
                        NormalDest, UnwindDest, Args, OpBundles, Name);
  }

  InvokeInst *CreateInvoke(FunctionCallee Callee, BasicBlock *NormalDest,
                           BasicBlock *UnwindDest,
                           ArrayRef<Value *> Args = None,
                           const Twine &Name = "") {
    return CreateInvoke(Callee.getFunctionType(), Callee.getCallee(),
                        NormalDest, UnwindDest, Args, Name);
  }

  /// \brief Create a callbr instruction.
  CallBrInst *CreateCallBr(FunctionType *Ty, Value *Callee,
                           BasicBlock *DefaultDest,
                           ArrayRef<BasicBlock *> IndirectDests,
                           ArrayRef<Value *> Args = None,
                           const Twine &Name = "") {
    return Insert(CallBrInst::Create(Ty, Callee, DefaultDest, IndirectDests,
                                     Args), Name);
  }
  CallBrInst *CreateCallBr(FunctionType *Ty, Value *Callee,
                           BasicBlock *DefaultDest,
                           ArrayRef<BasicBlock *> IndirectDests,
                           ArrayRef<Value *> Args,
                           ArrayRef<OperandBundleDef> OpBundles,
                           const Twine &Name = "") {
    return Insert(
        CallBrInst::Create(Ty, Callee, DefaultDest, IndirectDests, Args,
                           OpBundles), Name);
  }

  CallBrInst *CreateCallBr(FunctionCallee Callee, BasicBlock *DefaultDest,
                           ArrayRef<BasicBlock *> IndirectDests,
                           ArrayRef<Value *> Args = None,
                           const Twine &Name = "") {
    return CreateCallBr(Callee.getFunctionType(), Callee.getCallee(),
                        DefaultDest, IndirectDests, Args, Name);
  }
  CallBrInst *CreateCallBr(FunctionCallee Callee, BasicBlock *DefaultDest,
                           ArrayRef<BasicBlock *> IndirectDests,
                           ArrayRef<Value *> Args,
                           ArrayRef<OperandBundleDef> OpBundles,
                           const Twine &Name = "") {
    return CreateCallBr(Callee.getFunctionType(), Callee.getCallee(),
                        DefaultDest, IndirectDests, Args, Name);
  }

  ResumeInst *CreateResume(Value *Exn) {
    return Insert(ResumeInst::Create(Exn));
  }

  CleanupReturnInst *CreateCleanupRet(CleanupPadInst *CleanupPad,
                                      BasicBlock *UnwindBB = nullptr) {
    return Insert(CleanupReturnInst::Create(CleanupPad, UnwindBB));
  }

  CatchSwitchInst *CreateCatchSwitch(Value *ParentPad, BasicBlock *UnwindBB,
                                     unsigned NumHandlers,
                                     const Twine &Name = "") {
    return Insert(CatchSwitchInst::Create(ParentPad, UnwindBB, NumHandlers),
                  Name);
  }

  CatchPadInst *CreateCatchPad(Value *ParentPad, ArrayRef<Value *> Args,
                               const Twine &Name = "") {
    return Insert(CatchPadInst::Create(ParentPad, Args), Name);
  }

  CleanupPadInst *CreateCleanupPad(Value *ParentPad,
                                   ArrayRef<Value *> Args = None,
                                   const Twine &Name = "") {
    return Insert(CleanupPadInst::Create(ParentPad, Args), Name);
  }

  CatchReturnInst *CreateCatchRet(CatchPadInst *CatchPad, BasicBlock *BB) {
    return Insert(CatchReturnInst::Create(CatchPad, BB));
  }

  UnreachableInst *CreateUnreachable() {
    return Insert(new UnreachableInst(Context));
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Binary Operators
  //===--------------------------------------------------------------------===//
private:
  BinaryOperator *CreateInsertNUWNSWBinOp(BinaryOperator::BinaryOps Opc,
                                          Value *LHS, Value *RHS,
                                          const Twine &Name,
                                          bool HasNUW, bool HasNSW) {
    BinaryOperator *BO = Insert(BinaryOperator::Create(Opc, LHS, RHS), Name);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Instruction *setFPAttrs(Instruction *I, MDNode *FPMD,
                          FastMathFlags FMF) const {
    if (!FPMD)
      FPMD = DefaultFPMathTag;
    if (FPMD)
      I->setMetadata(LLVMContext::MD_fpmath, FPMD);
    I->setFastMathFlags(FMF);
    return I;
  }

  Value *foldConstant(Instruction::BinaryOps Opc, Value *L,
                      Value *R, const Twine &Name) const {
    auto *LC = dyn_cast<Constant>(L);
    auto *RC = dyn_cast<Constant>(R);
    return (LC && RC) ? Insert(Folder.CreateBinOp(Opc, LC, RC), Name) : nullptr;
  }

  Value *getConstrainedFPRounding(Optional<RoundingMode> Rounding) {
    RoundingMode UseRounding = DefaultConstrainedRounding;

    if (Rounding.hasValue())
      UseRounding = Rounding.getValue();

    Optional<StringRef> RoundingStr = convertRoundingModeToStr(UseRounding);
    assert(RoundingStr.hasValue() && "Garbage strict rounding mode!");
    auto *RoundingMDS = MDString::get(Context, RoundingStr.getValue());

    return MetadataAsValue::get(Context, RoundingMDS);
  }

  Value *getConstrainedFPExcept(Optional<fp::ExceptionBehavior> Except) {
    fp::ExceptionBehavior UseExcept = DefaultConstrainedExcept;

    if (Except.hasValue())
      UseExcept = Except.getValue();

    Optional<StringRef> ExceptStr = convertExceptionBehaviorToStr(UseExcept);
    assert(ExceptStr.hasValue() && "Garbage strict exception behavior!");
    auto *ExceptMDS = MDString::get(Context, ExceptStr.getValue());

    return MetadataAsValue::get(Context, ExceptMDS);
  }

  Value *getConstrainedFPPredicate(CmpInst::Predicate Predicate) {
    assert(CmpInst::isFPPredicate(Predicate) &&
           Predicate != CmpInst::FCMP_FALSE &&
           Predicate != CmpInst::FCMP_TRUE &&
           "Invalid constrained FP comparison predicate!");

    StringRef PredicateStr = CmpInst::getPredicateName(Predicate);
    auto *PredicateMDS = MDString::get(Context, PredicateStr);

    return MetadataAsValue::get(Context, PredicateMDS);
  }

public:
  Value *CreateAdd(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (auto *V = Folder.FoldAdd(LHS, RHS, HasNUW, HasNSW))
      return V;
    return CreateInsertNUWNSWBinOp(Instruction::Add, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }

  Value *CreateNSWAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateAdd(LHS, RHS, Name, false, true);
  }

  Value *CreateNUWAdd(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateAdd(LHS, RHS, Name, true, false);
  }

  Value *CreateSub(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateSub(LC, RC, HasNUW, HasNSW), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Sub, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }

  Value *CreateNSWSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateSub(LHS, RHS, Name, false, true);
  }

  Value *CreateNUWSub(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateSub(LHS, RHS, Name, true, false);
  }

  Value *CreateMul(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateMul(LC, RC, HasNUW, HasNSW), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Mul, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }

  Value *CreateNSWMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateMul(LHS, RHS, Name, false, true);
  }

  Value *CreateNUWMul(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateMul(LHS, RHS, Name, true, false);
  }

  Value *CreateUDiv(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateUDiv(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateUDiv(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactUDiv(LHS, RHS), Name);
  }

  Value *CreateExactUDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateUDiv(LHS, RHS, Name, true);
  }

  Value *CreateSDiv(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateSDiv(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateSDiv(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactSDiv(LHS, RHS), Name);
  }

  Value *CreateExactSDiv(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateSDiv(LHS, RHS, Name, true);
  }

  Value *CreateURem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Value *V = foldConstant(Instruction::URem, LHS, RHS, Name)) return V;
    return Insert(BinaryOperator::CreateURem(LHS, RHS), Name);
  }

  Value *CreateSRem(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Value *V = foldConstant(Instruction::SRem, LHS, RHS, Name)) return V;
    return Insert(BinaryOperator::CreateSRem(LHS, RHS), Name);
  }

  Value *CreateShl(Value *LHS, Value *RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateShl(LC, RC, HasNUW, HasNSW), Name);
    return CreateInsertNUWNSWBinOp(Instruction::Shl, LHS, RHS, Name,
                                   HasNUW, HasNSW);
  }

  Value *CreateShl(Value *LHS, const APInt &RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    return CreateShl(LHS, ConstantInt::get(LHS->getType(), RHS), Name,
                     HasNUW, HasNSW);
  }

  Value *CreateShl(Value *LHS, uint64_t RHS, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    return CreateShl(LHS, ConstantInt::get(LHS->getType(), RHS), Name,
                     HasNUW, HasNSW);
  }

  Value *CreateLShr(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateLShr(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateLShr(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactLShr(LHS, RHS), Name);
  }

  Value *CreateLShr(Value *LHS, const APInt &RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateLShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }

  Value *CreateLShr(Value *LHS, uint64_t RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateLShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }

  Value *CreateAShr(Value *LHS, Value *RHS, const Twine &Name = "",
                    bool isExact = false) {
    if (auto *LC = dyn_cast<Constant>(LHS))
      if (auto *RC = dyn_cast<Constant>(RHS))
        return Insert(Folder.CreateAShr(LC, RC, isExact), Name);
    if (!isExact)
      return Insert(BinaryOperator::CreateAShr(LHS, RHS), Name);
    return Insert(BinaryOperator::CreateExactAShr(LHS, RHS), Name);
  }

  Value *CreateAShr(Value *LHS, const APInt &RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateAShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }

  Value *CreateAShr(Value *LHS, uint64_t RHS, const Twine &Name = "",
                    bool isExact = false) {
    return CreateAShr(LHS, ConstantInt::get(LHS->getType(), RHS), Name,isExact);
  }

  Value *CreateAnd(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (auto *V = Folder.FoldAnd(LHS, RHS))
      return V;
    return Insert(BinaryOperator::CreateAnd(LHS, RHS), Name);
  }

  Value *CreateAnd(Value *LHS, const APInt &RHS, const Twine &Name = "") {
    return CreateAnd(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateAnd(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    return CreateAnd(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateAnd(ArrayRef<Value*> Ops) {
    assert(!Ops.empty());
    Value *Accum = Ops[0];
    for (unsigned i = 1; i < Ops.size(); i++)
      Accum = CreateAnd(Accum, Ops[i]);
    return Accum;
  }

  Value *CreateOr(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (auto *V = Folder.FoldOr(LHS, RHS))
      return V;
    return Insert(BinaryOperator::CreateOr(LHS, RHS), Name);
  }

  Value *CreateOr(Value *LHS, const APInt &RHS, const Twine &Name = "") {
    return CreateOr(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateOr(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    return CreateOr(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateOr(ArrayRef<Value*> Ops) {
    assert(!Ops.empty());
    Value *Accum = Ops[0];
    for (unsigned i = 1; i < Ops.size(); i++)
      Accum = CreateOr(Accum, Ops[i]);
    return Accum;
  }

  Value *CreateXor(Value *LHS, Value *RHS, const Twine &Name = "") {
    if (Value *V = foldConstant(Instruction::Xor, LHS, RHS, Name)) return V;
    return Insert(BinaryOperator::CreateXor(LHS, RHS), Name);
  }

  Value *CreateXor(Value *LHS, const APInt &RHS, const Twine &Name = "") {
    return CreateXor(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateXor(Value *LHS, uint64_t RHS, const Twine &Name = "") {
    return CreateXor(LHS, ConstantInt::get(LHS->getType(), RHS), Name);
  }

  Value *CreateFAdd(Value *L, Value *R, const Twine &Name = "",
                    MDNode *FPMD = nullptr) {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fadd,
                                      L, R, nullptr, Name, FPMD);

    if (Value *V = foldConstant(Instruction::FAdd, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFAdd(L, R), FPMD, FMF);
    return Insert(I, Name);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value *CreateFAddFMF(Value *L, Value *R, Instruction *FMFSource,
                       const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fadd,
                                      L, R, FMFSource, Name);

    if (Value *V = foldConstant(Instruction::FAdd, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFAdd(L, R), nullptr,
                                FMFSource->getFastMathFlags());
    return Insert(I, Name);
  }

  Value *CreateFSub(Value *L, Value *R, const Twine &Name = "",
                    MDNode *FPMD = nullptr) {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fsub,
                                      L, R, nullptr, Name, FPMD);

    if (Value *V = foldConstant(Instruction::FSub, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFSub(L, R), FPMD, FMF);
    return Insert(I, Name);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value *CreateFSubFMF(Value *L, Value *R, Instruction *FMFSource,
                       const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fsub,
                                      L, R, FMFSource, Name);

    if (Value *V = foldConstant(Instruction::FSub, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFSub(L, R), nullptr,
                                FMFSource->getFastMathFlags());
    return Insert(I, Name);
  }

  Value *CreateFMul(Value *L, Value *R, const Twine &Name = "",
                    MDNode *FPMD = nullptr) {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fmul,
                                      L, R, nullptr, Name, FPMD);

    if (Value *V = foldConstant(Instruction::FMul, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFMul(L, R), FPMD, FMF);
    return Insert(I, Name);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value *CreateFMulFMF(Value *L, Value *R, Instruction *FMFSource,
                       const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fmul,
                                      L, R, FMFSource, Name);

    if (Value *V = foldConstant(Instruction::FMul, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFMul(L, R), nullptr,
                                FMFSource->getFastMathFlags());
    return Insert(I, Name);
  }

  Value *CreateFDiv(Value *L, Value *R, const Twine &Name = "",
                    MDNode *FPMD = nullptr) {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fdiv,
                                      L, R, nullptr, Name, FPMD);

    if (Value *V = foldConstant(Instruction::FDiv, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFDiv(L, R), FPMD, FMF);
    return Insert(I, Name);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value *CreateFDivFMF(Value *L, Value *R, Instruction *FMFSource,
                       const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fdiv,
                                      L, R, FMFSource, Name);

    if (Value *V = foldConstant(Instruction::FDiv, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFDiv(L, R), nullptr,
                                FMFSource->getFastMathFlags());
    return Insert(I, Name);
  }

  Value *CreateFRem(Value *L, Value *R, const Twine &Name = "",
                    MDNode *FPMD = nullptr) {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_frem,
                                      L, R, nullptr, Name, FPMD);

    if (Value *V = foldConstant(Instruction::FRem, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFRem(L, R), FPMD, FMF);
    return Insert(I, Name);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value *CreateFRemFMF(Value *L, Value *R, Instruction *FMFSource,
                       const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_frem,
                                      L, R, FMFSource, Name);

    if (Value *V = foldConstant(Instruction::FRem, L, R, Name)) return V;
    Instruction *I = setFPAttrs(BinaryOperator::CreateFRem(L, R), nullptr,
                                FMFSource->getFastMathFlags());
    return Insert(I, Name);
  }

  Value *CreateBinOp(Instruction::BinaryOps Opc,
                     Value *LHS, Value *RHS, const Twine &Name = "",
                     MDNode *FPMathTag = nullptr) {
    if (Value *V = foldConstant(Opc, LHS, RHS, Name)) return V;
    Instruction *BinOp = BinaryOperator::Create(Opc, LHS, RHS);
    if (isa<FPMathOperator>(BinOp))
      setFPAttrs(BinOp, FPMathTag, FMF);
    return Insert(BinOp, Name);
  }

  Value *CreateLogicalAnd(Value *Cond1, Value *Cond2, const Twine &Name = "") {
    assert(Cond2->getType()->isIntOrIntVectorTy(1));
    return CreateSelect(Cond1, Cond2,
                        ConstantInt::getNullValue(Cond2->getType()), Name);
  }

  Value *CreateLogicalOr(Value *Cond1, Value *Cond2, const Twine &Name = "") {
    assert(Cond2->getType()->isIntOrIntVectorTy(1));
    return CreateSelect(Cond1, ConstantInt::getAllOnesValue(Cond2->getType()),
                        Cond2, Name);
  }

  // NOTE: this is sequential, non-commutative, ordered reduction!
  Value *CreateLogicalOr(ArrayRef<Value *> Ops) {
    assert(!Ops.empty());
    Value *Accum = Ops[0];
    for (unsigned i = 1; i < Ops.size(); i++)
      Accum = CreateLogicalOr(Accum, Ops[i]);
    return Accum;
  }

  CallInst *CreateConstrainedFPBinOp(
      Intrinsic::ID ID, Value *L, Value *R, Instruction *FMFSource = nullptr,
      const Twine &Name = "", MDNode *FPMathTag = nullptr,
      Optional<RoundingMode> Rounding = None,
      Optional<fp::ExceptionBehavior> Except = None);

  Value *CreateNeg(Value *V, const Twine &Name = "",
                   bool HasNUW = false, bool HasNSW = false) {
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateNeg(VC, HasNUW, HasNSW), Name);
    BinaryOperator *BO = Insert(BinaryOperator::CreateNeg(V), Name);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Value *CreateNSWNeg(Value *V, const Twine &Name = "") {
    return CreateNeg(V, Name, false, true);
  }

  Value *CreateNUWNeg(Value *V, const Twine &Name = "") {
    return CreateNeg(V, Name, true, false);
  }

  Value *CreateFNeg(Value *V, const Twine &Name = "",
                    MDNode *FPMathTag = nullptr) {
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateFNeg(VC), Name);
    return Insert(setFPAttrs(UnaryOperator::CreateFNeg(V), FPMathTag, FMF),
                  Name);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value *CreateFNegFMF(Value *V, Instruction *FMFSource,
                       const Twine &Name = "") {
   if (auto *VC = dyn_cast<Constant>(V))
     return Insert(Folder.CreateFNeg(VC), Name);
   return Insert(setFPAttrs(UnaryOperator::CreateFNeg(V), nullptr,
                            FMFSource->getFastMathFlags()),
                 Name);
  }

  Value *CreateNot(Value *V, const Twine &Name = "") {
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateNot(VC), Name);
    return Insert(BinaryOperator::CreateNot(V), Name);
  }

  Value *CreateUnOp(Instruction::UnaryOps Opc,
                    Value *V, const Twine &Name = "",
                    MDNode *FPMathTag = nullptr) {
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateUnOp(Opc, VC), Name);
    Instruction *UnOp = UnaryOperator::Create(Opc, V);
    if (isa<FPMathOperator>(UnOp))
      setFPAttrs(UnOp, FPMathTag, FMF);
    return Insert(UnOp, Name);
  }

  /// Create either a UnaryOperator or BinaryOperator depending on \p Opc.
  /// Correct number of operands must be passed accordingly.
  Value *CreateNAryOp(unsigned Opc, ArrayRef<Value *> Ops,
                      const Twine &Name = "", MDNode *FPMathTag = nullptr);

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//

  AllocaInst *CreateAlloca(Type *Ty, unsigned AddrSpace,
                           Value *ArraySize = nullptr, const Twine &Name = "") {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    Align AllocaAlign = DL.getPrefTypeAlign(Ty);
    return Insert(new AllocaInst(Ty, AddrSpace, ArraySize, AllocaAlign), Name);
  }

  AllocaInst *CreateAlloca(Type *Ty, Value *ArraySize = nullptr,
                           const Twine &Name = "") {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    Align AllocaAlign = DL.getPrefTypeAlign(Ty);
    unsigned AddrSpace = DL.getAllocaAddrSpace();
    return Insert(new AllocaInst(Ty, AddrSpace, ArraySize, AllocaAlign), Name);
  }

  /// Provided to resolve 'CreateLoad(Ty, Ptr, "...")' correctly, instead of
  /// converting the string to 'bool' for the isVolatile parameter.
  LoadInst *CreateLoad(Type *Ty, Value *Ptr, const char *Name) {
    return CreateAlignedLoad(Ty, Ptr, MaybeAlign(), Name);
  }

  LoadInst *CreateLoad(Type *Ty, Value *Ptr, const Twine &Name = "") {
    return CreateAlignedLoad(Ty, Ptr, MaybeAlign(), Name);
  }

  LoadInst *CreateLoad(Type *Ty, Value *Ptr, bool isVolatile,
                       const Twine &Name = "") {
    return CreateAlignedLoad(Ty, Ptr, MaybeAlign(), isVolatile, Name);
  }

  StoreInst *CreateStore(Value *Val, Value *Ptr, bool isVolatile = false) {
    return CreateAlignedStore(Val, Ptr, MaybeAlign(), isVolatile);
  }

  LoadInst *CreateAlignedLoad(Type *Ty, Value *Ptr, MaybeAlign Align,
                              const char *Name) {
    return CreateAlignedLoad(Ty, Ptr, Align, /*isVolatile*/false, Name);
  }

  LoadInst *CreateAlignedLoad(Type *Ty, Value *Ptr, MaybeAlign Align,
                              const Twine &Name = "") {
    return CreateAlignedLoad(Ty, Ptr, Align, /*isVolatile*/false, Name);
  }

  LoadInst *CreateAlignedLoad(Type *Ty, Value *Ptr, MaybeAlign Align,
                              bool isVolatile, const Twine &Name = "") {
    if (!Align) {
      const DataLayout &DL = BB->getModule()->getDataLayout();
      Align = DL.getABITypeAlign(Ty);
    }
    return Insert(new LoadInst(Ty, Ptr, Twine(), isVolatile, *Align), Name);
  }

  StoreInst *CreateAlignedStore(Value *Val, Value *Ptr, MaybeAlign Align,
                                bool isVolatile = false) {
    if (!Align) {
      const DataLayout &DL = BB->getModule()->getDataLayout();
      Align = DL.getABITypeAlign(Val->getType());
    }
    return Insert(new StoreInst(Val, Ptr, isVolatile, *Align));
  }
  FenceInst *CreateFence(AtomicOrdering Ordering,
                         SyncScope::ID SSID = SyncScope::System,
                         const Twine &Name = "") {
    return Insert(new FenceInst(Context, Ordering, SSID), Name);
  }

  AtomicCmpXchgInst *
  CreateAtomicCmpXchg(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
                      AtomicOrdering SuccessOrdering,
                      AtomicOrdering FailureOrdering,
                      SyncScope::ID SSID = SyncScope::System) {
    if (!Align) {
      const DataLayout &DL = BB->getModule()->getDataLayout();
      Align = llvm::Align(DL.getTypeStoreSize(New->getType()));
    }

    return Insert(new AtomicCmpXchgInst(Ptr, Cmp, New, *Align, SuccessOrdering,
                                        FailureOrdering, SSID));
  }

  AtomicRMWInst *CreateAtomicRMW(AtomicRMWInst::BinOp Op, Value *Ptr,
                                 Value *Val, MaybeAlign Align,
                                 AtomicOrdering Ordering,
                                 SyncScope::ID SSID = SyncScope::System) {
    if (!Align) {
      const DataLayout &DL = BB->getModule()->getDataLayout();
      Align = llvm::Align(DL.getTypeStoreSize(Val->getType()));
    }

    return Insert(new AtomicRMWInst(Op, Ptr, Val, *Align, Ordering, SSID));
  }

  Value *CreateGEP(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                   const Twine &Name = "") {
    if (auto *V = Folder.FoldGEP(Ty, Ptr, IdxList, /*IsInBounds=*/false))
      return V;
    return Insert(GetElementPtrInst::Create(Ty, Ptr, IdxList), Name);
  }

  Value *CreateInBoundsGEP(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                           const Twine &Name = "") {
    if (auto *V = Folder.FoldGEP(Ty, Ptr, IdxList, /*IsInBounds=*/true))
      return V;
    return Insert(GetElementPtrInst::CreateInBounds(Ty, Ptr, IdxList), Name);
  }

  Value *CreateGEP(Type *Ty, Value *Ptr, Value *Idx, const Twine &Name = "") {
    if (auto *V = Folder.FoldGEP(Ty, Ptr, {Idx}, /*IsInBounds=*/false))
      return V;
    return Insert(GetElementPtrInst::Create(Ty, Ptr, Idx), Name);
  }

  Value *CreateInBoundsGEP(Type *Ty, Value *Ptr, Value *Idx,
                           const Twine &Name = "") {
    if (auto *V = Folder.FoldGEP(Ty, Ptr, {Idx}, /*IsInBounds=*/true))
      return V;
    return Insert(GetElementPtrInst::CreateInBounds(Ty, Ptr, Idx), Name);
  }

  Value *CreateConstGEP1_32(Type *Ty, Value *Ptr, unsigned Idx0,
                            const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt32Ty(Context), Idx0);

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idx, /*IsInBounds=*/false))
      return V;

    return Insert(GetElementPtrInst::Create(Ty, Ptr, Idx), Name);
  }

  Value *CreateConstInBoundsGEP1_32(Type *Ty, Value *Ptr, unsigned Idx0,
                                    const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt32Ty(Context), Idx0);

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idx, /*IsInBounds=*/true))
      return V;

    return Insert(GetElementPtrInst::CreateInBounds(Ty, Ptr, Idx), Name);
  }

  Value *CreateConstGEP2_32(Type *Ty, Value *Ptr, unsigned Idx0, unsigned Idx1,
                            const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt32Ty(Context), Idx0),
      ConstantInt::get(Type::getInt32Ty(Context), Idx1)
    };

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idxs, /*IsInBounds=*/false))
      return V;

    return Insert(GetElementPtrInst::Create(Ty, Ptr, Idxs), Name);
  }

  Value *CreateConstInBoundsGEP2_32(Type *Ty, Value *Ptr, unsigned Idx0,
                                    unsigned Idx1, const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt32Ty(Context), Idx0),
      ConstantInt::get(Type::getInt32Ty(Context), Idx1)
    };

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idxs, /*IsInBounds=*/true))
      return V;

    return Insert(GetElementPtrInst::CreateInBounds(Ty, Ptr, Idxs), Name);
  }

  Value *CreateConstGEP1_64(Type *Ty, Value *Ptr, uint64_t Idx0,
                            const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt64Ty(Context), Idx0);

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idx, /*IsInBounds=*/false))
      return V;

    return Insert(GetElementPtrInst::Create(Ty, Ptr, Idx), Name);
  }

  Value *CreateConstInBoundsGEP1_64(Type *Ty, Value *Ptr, uint64_t Idx0,
                                    const Twine &Name = "") {
    Value *Idx = ConstantInt::get(Type::getInt64Ty(Context), Idx0);

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idx, /*IsInBounds=*/true))
      return V;

    return Insert(GetElementPtrInst::CreateInBounds(Ty, Ptr, Idx), Name);
  }

  Value *CreateConstGEP2_64(Type *Ty, Value *Ptr, uint64_t Idx0, uint64_t Idx1,
                            const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt64Ty(Context), Idx0),
      ConstantInt::get(Type::getInt64Ty(Context), Idx1)
    };

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idxs, /*IsInBounds=*/false))
      return V;

    return Insert(GetElementPtrInst::Create(Ty, Ptr, Idxs), Name);
  }

  Value *CreateConstInBoundsGEP2_64(Type *Ty, Value *Ptr, uint64_t Idx0,
                                    uint64_t Idx1, const Twine &Name = "") {
    Value *Idxs[] = {
      ConstantInt::get(Type::getInt64Ty(Context), Idx0),
      ConstantInt::get(Type::getInt64Ty(Context), Idx1)
    };

    if (auto *V = Folder.FoldGEP(Ty, Ptr, Idxs, /*IsInBounds=*/true))
      return V;

    return Insert(GetElementPtrInst::CreateInBounds(Ty, Ptr, Idxs), Name);
  }

  Value *CreateStructGEP(Type *Ty, Value *Ptr, unsigned Idx,
                         const Twine &Name = "") {
    return CreateConstInBoundsGEP2_32(Ty, Ptr, 0, Idx, Name);
  }

  /// Same as CreateGlobalString, but return a pointer with "i8*" type
  /// instead of a pointer to array of i8.
  ///
  /// If no module is given via \p M, it is take from the insertion point basic
  /// block.
  Constant *CreateGlobalStringPtr(StringRef Str, const Twine &Name = "",
                                  unsigned AddressSpace = 0,
                                  Module *M = nullptr) {
    GlobalVariable *GV = CreateGlobalString(Str, Name, AddressSpace, M);
    Constant *Zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Constant *Indices[] = {Zero, Zero};
    return ConstantExpr::getInBoundsGetElementPtr(GV->getValueType(), GV,
                                                  Indices);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Value *CreateTrunc(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::Trunc, V, DestTy, Name);
  }

  Value *CreateZExt(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::ZExt, V, DestTy, Name);
  }

  Value *CreateSExt(Value *V, Type *DestTy, const Twine &Name = "") {
    return CreateCast(Instruction::SExt, V, DestTy, Name);
  }

  /// Create a ZExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  Value *CreateZExtOrTrunc(Value *V, Type *DestTy,
                           const Twine &Name = "") {
    assert(V->getType()->isIntOrIntVectorTy() &&
           DestTy->isIntOrIntVectorTy() &&
           "Can only zero extend/truncate integers!");
    Type *VTy = V->getType();
    if (VTy->getScalarSizeInBits() < DestTy->getScalarSizeInBits())
      return CreateZExt(V, DestTy, Name);
    if (VTy->getScalarSizeInBits() > DestTy->getScalarSizeInBits())
      return CreateTrunc(V, DestTy, Name);
    return V;
  }

  /// Create a SExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  Value *CreateSExtOrTrunc(Value *V, Type *DestTy,
                           const Twine &Name = "") {
    assert(V->getType()->isIntOrIntVectorTy() &&
           DestTy->isIntOrIntVectorTy() &&
           "Can only sign extend/truncate integers!");
    Type *VTy = V->getType();
    if (VTy->getScalarSizeInBits() < DestTy->getScalarSizeInBits())
      return CreateSExt(V, DestTy, Name);
    if (VTy->getScalarSizeInBits() > DestTy->getScalarSizeInBits())
      return CreateTrunc(V, DestTy, Name);
    return V;
  }

  Value *CreateFPToUI(Value *V, Type *DestTy, const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPCast(Intrinsic::experimental_constrained_fptoui,
                                     V, DestTy, nullptr, Name);
    return CreateCast(Instruction::FPToUI, V, DestTy, Name);
  }

  Value *CreateFPToSI(Value *V, Type *DestTy, const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPCast(Intrinsic::experimental_constrained_fptosi,
                                     V, DestTy, nullptr, Name);
    return CreateCast(Instruction::FPToSI, V, DestTy, Name);
  }

  Value *CreateUIToFP(Value *V, Type *DestTy, const Twine &Name = ""){
    if (IsFPConstrained)
      return CreateConstrainedFPCast(Intrinsic::experimental_constrained_uitofp,
                                     V, DestTy, nullptr, Name);
    return CreateCast(Instruction::UIToFP, V, DestTy, Name);
  }

  Value *CreateSIToFP(Value *V, Type *DestTy, const Twine &Name = ""){
    if (IsFPConstrained)
      return CreateConstrainedFPCast(Intrinsic::experimental_constrained_sitofp,
                                     V, DestTy, nullptr, Name);
    return CreateCast(Instruction::SIToFP, V, DestTy, Name);
  }

  Value *CreateFPTrunc(Value *V, Type *DestTy,
                       const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPCast(
          Intrinsic::experimental_constrained_fptrunc, V, DestTy, nullptr,
          Name);
    return CreateCast(Instruction::FPTrunc, V, DestTy, Name);
  }

  Value *CreateFPExt(Value *V, Type *DestTy, const Twine &Name = "") {
    if (IsFPConstrained)
      return CreateConstrainedFPCast(Intrinsic::experimental_constrained_fpext,
                                     V, DestTy, nullptr, Name);
    return CreateCast(Instruction::FPExt, V, DestTy, Name);
  }

  Value *CreatePtrToInt(Value *V, Type *DestTy,
                        const Twine &Name = "") {
    return CreateCast(Instruction::PtrToInt, V, DestTy, Name);
  }

  Value *CreateIntToPtr(Value *V, Type *DestTy,
                        const Twine &Name = "") {
    return CreateCast(Instruction::IntToPtr, V, DestTy, Name);
  }

  Value *CreateBitCast(Value *V, Type *DestTy,
                       const Twine &Name = "") {
    return CreateCast(Instruction::BitCast, V, DestTy, Name);
  }

  Value *CreateAddrSpaceCast(Value *V, Type *DestTy,
                             const Twine &Name = "") {
    return CreateCast(Instruction::AddrSpaceCast, V, DestTy, Name);
  }

  Value *CreateZExtOrBitCast(Value *V, Type *DestTy,
                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateZExtOrBitCast(VC, DestTy), Name);
    return Insert(CastInst::CreateZExtOrBitCast(V, DestTy), Name);
  }

  Value *CreateSExtOrBitCast(Value *V, Type *DestTy,
                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateSExtOrBitCast(VC, DestTy), Name);
    return Insert(CastInst::CreateSExtOrBitCast(V, DestTy), Name);
  }

  Value *CreateTruncOrBitCast(Value *V, Type *DestTy,
                              const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateTruncOrBitCast(VC, DestTy), Name);
    return Insert(CastInst::CreateTruncOrBitCast(V, DestTy), Name);
  }

  Value *CreateCast(Instruction::CastOps Op, Value *V, Type *DestTy,
                    const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateCast(Op, VC, DestTy), Name);
    return Insert(CastInst::Create(Op, V, DestTy), Name);
  }

  Value *CreatePointerCast(Value *V, Type *DestTy,
                           const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreatePointerCast(VC, DestTy), Name);
    return Insert(CastInst::CreatePointerCast(V, DestTy), Name);
  }

  Value *CreatePointerBitCastOrAddrSpaceCast(Value *V, Type *DestTy,
                                             const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;

    if (auto *VC = dyn_cast<Constant>(V)) {
      return Insert(Folder.CreatePointerBitCastOrAddrSpaceCast(VC, DestTy),
                    Name);
    }

    return Insert(CastInst::CreatePointerBitCastOrAddrSpaceCast(V, DestTy),
                  Name);
  }

  Value *CreateIntCast(Value *V, Type *DestTy, bool isSigned,
                       const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateIntCast(VC, DestTy, isSigned), Name);
    return Insert(CastInst::CreateIntegerCast(V, DestTy, isSigned), Name);
  }

  Value *CreateBitOrPointerCast(Value *V, Type *DestTy,
                                const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (V->getType()->isPtrOrPtrVectorTy() && DestTy->isIntOrIntVectorTy())
      return CreatePtrToInt(V, DestTy, Name);
    if (V->getType()->isIntOrIntVectorTy() && DestTy->isPtrOrPtrVectorTy())
      return CreateIntToPtr(V, DestTy, Name);

    return CreateBitCast(V, DestTy, Name);
  }

  Value *CreateFPCast(Value *V, Type *DestTy, const Twine &Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (auto *VC = dyn_cast<Constant>(V))
      return Insert(Folder.CreateFPCast(VC, DestTy), Name);
    return Insert(CastInst::CreateFPCast(V, DestTy), Name);
  }

  CallInst *CreateConstrainedFPCast(
      Intrinsic::ID ID, Value *V, Type *DestTy,
      Instruction *FMFSource = nullptr, const Twine &Name = "",
      MDNode *FPMathTag = nullptr,
      Optional<RoundingMode> Rounding = None,
      Optional<fp::ExceptionBehavior> Except = None);

  // Provided to resolve 'CreateIntCast(Ptr, Ptr, "...")', giving a
  // compile time error, instead of converting the string to bool for the
  // isSigned parameter.
  Value *CreateIntCast(Value *, Type *, const char *) = delete;

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//

  Value *CreateICmpEQ(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_EQ, LHS, RHS, Name);
  }

  Value *CreateICmpNE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_NE, LHS, RHS, Name);
  }

  Value *CreateICmpUGT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGT, LHS, RHS, Name);
  }

  Value *CreateICmpUGE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGE, LHS, RHS, Name);
  }

  Value *CreateICmpULT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULT, LHS, RHS, Name);
  }

  Value *CreateICmpULE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULE, LHS, RHS, Name);
  }

  Value *CreateICmpSGT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGT, LHS, RHS, Name);
  }

  Value *CreateICmpSGE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGE, LHS, RHS, Name);
  }

  Value *CreateICmpSLT(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLT, LHS, RHS, Name);
  }

  Value *CreateICmpSLE(Value *LHS, Value *RHS, const Twine &Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLE, LHS, RHS, Name);
  }

  Value *CreateFCmpOEQ(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_OEQ, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpOGT(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_OGT, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpOGE(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_OGE, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpOLT(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_OLT, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpOLE(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_OLE, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpONE(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_ONE, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpORD(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_ORD, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpUNO(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_UNO, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpUEQ(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_UEQ, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpUGT(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_UGT, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpUGE(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_UGE, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpULT(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_ULT, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpULE(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_ULE, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateFCmpUNE(Value *LHS, Value *RHS, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    return CreateFCmp(FCmpInst::FCMP_UNE, LHS, RHS, Name, FPMathTag);
  }

  Value *CreateICmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                    const Twine &Name = "") {
    if (auto *V = Folder.FoldICmp(P, LHS, RHS))
      return V;
    return Insert(new ICmpInst(P, LHS, RHS), Name);
  }

  // Create a quiet floating-point comparison (i.e. one that raises an FP
  // exception only in the case where an input is a signaling NaN).
  // Note that this differs from CreateFCmpS only if IsFPConstrained is true.
  Value *CreateFCmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                    const Twine &Name = "", MDNode *FPMathTag = nullptr) {
    return CreateFCmpHelper(P, LHS, RHS, Name, FPMathTag, false);
  }

  Value *CreateCmp(CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                   const Twine &Name = "", MDNode *FPMathTag = nullptr) {
    return CmpInst::isFPPredicate(Pred)
               ? CreateFCmp(Pred, LHS, RHS, Name, FPMathTag)
               : CreateICmp(Pred, LHS, RHS, Name);
  }

  // Create a signaling floating-point comparison (i.e. one that raises an FP
  // exception whenever an input is any NaN, signaling or quiet).
  // Note that this differs from CreateFCmp only if IsFPConstrained is true.
  Value *CreateFCmpS(CmpInst::Predicate P, Value *LHS, Value *RHS,
                     const Twine &Name = "", MDNode *FPMathTag = nullptr) {
    return CreateFCmpHelper(P, LHS, RHS, Name, FPMathTag, true);
  }

private:
  // Helper routine to create either a signaling or a quiet FP comparison.
  Value *CreateFCmpHelper(CmpInst::Predicate P, Value *LHS, Value *RHS,
                          const Twine &Name, MDNode *FPMathTag,
                          bool IsSignaling);

public:
  CallInst *CreateConstrainedFPCmp(
      Intrinsic::ID ID, CmpInst::Predicate P, Value *L, Value *R,
      const Twine &Name = "", Optional<fp::ExceptionBehavior> Except = None);

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//

  PHINode *CreatePHI(Type *Ty, unsigned NumReservedValues,
                     const Twine &Name = "") {
    PHINode *Phi = PHINode::Create(Ty, NumReservedValues);
    if (isa<FPMathOperator>(Phi))
      setFPAttrs(Phi, nullptr /* MDNode* */, FMF);
    return Insert(Phi, Name);
  }

  CallInst *CreateCall(FunctionType *FTy, Value *Callee,
                       ArrayRef<Value *> Args = None, const Twine &Name = "",
                       MDNode *FPMathTag = nullptr) {
    CallInst *CI = CallInst::Create(FTy, Callee, Args, DefaultOperandBundles);
    if (IsFPConstrained)
      setConstrainedFPCallAttr(CI);
    if (isa<FPMathOperator>(CI))
      setFPAttrs(CI, FPMathTag, FMF);
    return Insert(CI, Name);
  }

  CallInst *CreateCall(FunctionType *FTy, Value *Callee, ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = "", MDNode *FPMathTag = nullptr) {
    CallInst *CI = CallInst::Create(FTy, Callee, Args, OpBundles);
    if (IsFPConstrained)
      setConstrainedFPCallAttr(CI);
    if (isa<FPMathOperator>(CI))
      setFPAttrs(CI, FPMathTag, FMF);
    return Insert(CI, Name);
  }

  CallInst *CreateCall(FunctionCallee Callee, ArrayRef<Value *> Args = None,
                       const Twine &Name = "", MDNode *FPMathTag = nullptr) {
    return CreateCall(Callee.getFunctionType(), Callee.getCallee(), Args, Name,
                      FPMathTag);
  }

  CallInst *CreateCall(FunctionCallee Callee, ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = "", MDNode *FPMathTag = nullptr) {
    return CreateCall(Callee.getFunctionType(), Callee.getCallee(), Args,
                      OpBundles, Name, FPMathTag);
  }

  CallInst *CreateConstrainedFPCall(
      Function *Callee, ArrayRef<Value *> Args, const Twine &Name = "",
      Optional<RoundingMode> Rounding = None,
      Optional<fp::ExceptionBehavior> Except = None);

  Value *CreateSelect(Value *C, Value *True, Value *False,
                      const Twine &Name = "", Instruction *MDFrom = nullptr);

  VAArgInst *CreateVAArg(Value *List, Type *Ty, const Twine &Name = "") {
    return Insert(new VAArgInst(List, Ty), Name);
  }

  Value *CreateExtractElement(Value *Vec, Value *Idx,
                              const Twine &Name = "") {
    if (auto *VC = dyn_cast<Constant>(Vec))
      if (auto *IC = dyn_cast<Constant>(Idx))
        return Insert(Folder.CreateExtractElement(VC, IC), Name);
    return Insert(ExtractElementInst::Create(Vec, Idx), Name);
  }

  Value *CreateExtractElement(Value *Vec, uint64_t Idx,
                              const Twine &Name = "") {
    return CreateExtractElement(Vec, getInt64(Idx), Name);
  }

  Value *CreateInsertElement(Type *VecTy, Value *NewElt, Value *Idx,
                             const Twine &Name = "") {
    return CreateInsertElement(PoisonValue::get(VecTy), NewElt, Idx, Name);
  }

  Value *CreateInsertElement(Type *VecTy, Value *NewElt, uint64_t Idx,
                             const Twine &Name = "") {
    return CreateInsertElement(PoisonValue::get(VecTy), NewElt, Idx, Name);
  }

  Value *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                             const Twine &Name = "") {
    if (auto *VC = dyn_cast<Constant>(Vec))
      if (auto *NC = dyn_cast<Constant>(NewElt))
        if (auto *IC = dyn_cast<Constant>(Idx))
          return Insert(Folder.CreateInsertElement(VC, NC, IC), Name);
    return Insert(InsertElementInst::Create(Vec, NewElt, Idx), Name);
  }

  Value *CreateInsertElement(Value *Vec, Value *NewElt, uint64_t Idx,
                             const Twine &Name = "") {
    return CreateInsertElement(Vec, NewElt, getInt64(Idx), Name);
  }

  Value *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                             const Twine &Name = "") {
    SmallVector<int, 16> IntMask;
    ShuffleVectorInst::getShuffleMask(cast<Constant>(Mask), IntMask);
    return CreateShuffleVector(V1, V2, IntMask, Name);
  }

  /// See class ShuffleVectorInst for a description of the mask representation.
  Value *CreateShuffleVector(Value *V1, Value *V2, ArrayRef<int> Mask,
                             const Twine &Name = "") {
    if (auto *V1C = dyn_cast<Constant>(V1))
      if (auto *V2C = dyn_cast<Constant>(V2))
        return Insert(Folder.CreateShuffleVector(V1C, V2C, Mask), Name);
    return Insert(new ShuffleVectorInst(V1, V2, Mask), Name);
  }

  /// Create a unary shuffle. The second vector operand of the IR instruction
  /// is poison.
  Value *CreateShuffleVector(Value *V, ArrayRef<int> Mask,
                             const Twine &Name = "") {
    return CreateShuffleVector(V, PoisonValue::get(V->getType()), Mask, Name);
  }

  Value *CreateExtractValue(Value *Agg,
                            ArrayRef<unsigned> Idxs,
                            const Twine &Name = "") {
    if (auto *AggC = dyn_cast<Constant>(Agg))
      return Insert(Folder.CreateExtractValue(AggC, Idxs), Name);
    return Insert(ExtractValueInst::Create(Agg, Idxs), Name);
  }

  Value *CreateInsertValue(Value *Agg, Value *Val,
                           ArrayRef<unsigned> Idxs,
                           const Twine &Name = "") {
    if (auto *AggC = dyn_cast<Constant>(Agg))
      if (auto *ValC = dyn_cast<Constant>(Val))
        return Insert(Folder.CreateInsertValue(AggC, ValC, Idxs), Name);
    return Insert(InsertValueInst::Create(Agg, Val, Idxs), Name);
  }

  LandingPadInst *CreateLandingPad(Type *Ty, unsigned NumClauses,
                                   const Twine &Name = "") {
    return Insert(LandingPadInst::Create(Ty, NumClauses), Name);
  }

  Value *CreateFreeze(Value *V, const Twine &Name = "") {
    return Insert(new FreezeInst(V), Name);
  }

  //===--------------------------------------------------------------------===//
  // Utility creation methods
  //===--------------------------------------------------------------------===//

  /// Return an i1 value testing if \p Arg is null.
  Value *CreateIsNull(Value *Arg, const Twine &Name = "") {
    return CreateICmpEQ(Arg, Constant::getNullValue(Arg->getType()),
                        Name);
  }

  /// Return an i1 value testing if \p Arg is not null.
  Value *CreateIsNotNull(Value *Arg, const Twine &Name = "") {
    return CreateICmpNE(Arg, Constant::getNullValue(Arg->getType()),
                        Name);
  }

  /// Return the i64 difference between two pointer values, dividing out
  /// the size of the pointed-to objects.
  ///
  /// This is intended to implement C-style pointer subtraction. As such, the
  /// pointers must be appropriately aligned for their element types and
  /// pointing into the same object.
  Value *CreatePtrDiff(Type *ElemTy, Value *LHS, Value *RHS,
                       const Twine &Name = "");

  /// Create a launder.invariant.group intrinsic call. If Ptr type is
  /// different from pointer to i8, it's casted to pointer to i8 in the same
  /// address space before call and casted back to Ptr type after call.
  Value *CreateLaunderInvariantGroup(Value *Ptr);

  /// \brief Create a strip.invariant.group intrinsic call. If Ptr type is
  /// different from pointer to i8, it's casted to pointer to i8 in the same
  /// address space before call and casted back to Ptr type after call.
  Value *CreateStripInvariantGroup(Value *Ptr);

  /// Return a vector value that contains the vector V reversed
  Value *CreateVectorReverse(Value *V, const Twine &Name = "");

  /// Return a vector splice intrinsic if using scalable vectors, otherwise
  /// return a shufflevector. If the immediate is positive, a vector is
  /// extracted from concat(V1, V2), starting at Imm. If the immediate
  /// is negative, we extract -Imm elements from V1 and the remaining
  /// elements from V2. Imm is a signed integer in the range
  /// -VL <= Imm < VL (where VL is the runtime vector length of the
  /// source/result vector)
  Value *CreateVectorSplice(Value *V1, Value *V2, int64_t Imm,
                            const Twine &Name = "");

  /// Return a vector value that contains \arg V broadcasted to \p
  /// NumElts elements.
  Value *CreateVectorSplat(unsigned NumElts, Value *V, const Twine &Name = "");

  /// Return a vector value that contains \arg V broadcasted to \p
  /// EC elements.
  Value *CreateVectorSplat(ElementCount EC, Value *V, const Twine &Name = "");

  /// Return a value that has been extracted from a larger integer type.
  Value *CreateExtractInteger(const DataLayout &DL, Value *From,
                              IntegerType *ExtractedTy, uint64_t Offset,
                              const Twine &Name);

  Value *CreatePreserveArrayAccessIndex(Type *ElTy, Value *Base,
                                        unsigned Dimension, unsigned LastIndex,
                                        MDNode *DbgInfo);

  Value *CreatePreserveUnionAccessIndex(Value *Base, unsigned FieldIndex,
                                        MDNode *DbgInfo);

  Value *CreatePreserveStructAccessIndex(Type *ElTy, Value *Base,
                                         unsigned Index, unsigned FieldIndex,
                                         MDNode *DbgInfo);

private:
  /// Helper function that creates an assume intrinsic call that
  /// represents an alignment assumption on the provided pointer \p PtrValue
  /// with offset \p OffsetValue and alignment value \p AlignValue.
  CallInst *CreateAlignmentAssumptionHelper(const DataLayout &DL,
                                            Value *PtrValue, Value *AlignValue,
                                            Value *OffsetValue);

public:
  /// Create an assume intrinsic call that represents an alignment
  /// assumption on the provided pointer.
  ///
  /// An optional offset can be provided, and if it is provided, the offset
  /// must be subtracted from the provided pointer to get the pointer with the
  /// specified alignment.
  CallInst *CreateAlignmentAssumption(const DataLayout &DL, Value *PtrValue,
                                      unsigned Alignment,
                                      Value *OffsetValue = nullptr);

  /// Create an assume intrinsic call that represents an alignment
  /// assumption on the provided pointer.
  ///
  /// An optional offset can be provided, and if it is provided, the offset
  /// must be subtracted from the provided pointer to get the pointer with the
  /// specified alignment.
  ///
  /// This overload handles the condition where the Alignment is dependent
  /// on an existing value rather than a static value.
  CallInst *CreateAlignmentAssumption(const DataLayout &DL, Value *PtrValue,
                                      Value *Alignment,
                                      Value *OffsetValue = nullptr);
};

/// This provides a uniform API for creating instructions and inserting
/// them into a basic block: either at the end of a BasicBlock, or at a specific
/// iterator location in a block.
///
/// Note that the builder does not expose the full generality of LLVM
/// instructions.  For access to extra instruction properties, use the mutators
/// (e.g. setVolatile) on the instructions after they have been
/// created. Convenience state exists to specify fast-math flags and fp-math
/// tags.
///
/// The first template argument specifies a class to use for creating constants.
/// This defaults to creating minimally folded constants.  The second template
/// argument allows clients to specify custom insertion hooks that are called on
/// every newly created insertion.
template <typename FolderTy = ConstantFolder,
          typename InserterTy = IRBuilderDefaultInserter>
class IRBuilder : public IRBuilderBase {
private:
  FolderTy Folder;
  InserterTy Inserter;

public:
  IRBuilder(LLVMContext &C, FolderTy Folder, InserterTy Inserter = InserterTy(),
            MDNode *FPMathTag = nullptr,
            ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(C, this->Folder, this->Inserter, FPMathTag, OpBundles),
        Folder(Folder), Inserter(Inserter) {}

  explicit IRBuilder(LLVMContext &C, MDNode *FPMathTag = nullptr,
                     ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(C, this->Folder, this->Inserter, FPMathTag, OpBundles) {}

  explicit IRBuilder(BasicBlock *TheBB, FolderTy Folder,
                     MDNode *FPMathTag = nullptr,
                     ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(TheBB->getContext(), this->Folder, this->Inserter,
                      FPMathTag, OpBundles), Folder(Folder) {
    SetInsertPoint(TheBB);
  }

  explicit IRBuilder(BasicBlock *TheBB, MDNode *FPMathTag = nullptr,
                     ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(TheBB->getContext(), this->Folder, this->Inserter,
                      FPMathTag, OpBundles) {
    SetInsertPoint(TheBB);
  }

  explicit IRBuilder(Instruction *IP, MDNode *FPMathTag = nullptr,
                     ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(IP->getContext(), this->Folder, this->Inserter,
                      FPMathTag, OpBundles) {
    SetInsertPoint(IP);
  }

  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP, FolderTy Folder,
            MDNode *FPMathTag = nullptr,
            ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(TheBB->getContext(), this->Folder, this->Inserter,
                      FPMathTag, OpBundles), Folder(Folder) {
    SetInsertPoint(TheBB, IP);
  }

  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP,
            MDNode *FPMathTag = nullptr,
            ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilderBase(TheBB->getContext(), this->Folder, this->Inserter,
                      FPMathTag, OpBundles) {
    SetInsertPoint(TheBB, IP);
  }

  /// Avoid copying the full IRBuilder. Prefer using InsertPointGuard
  /// or FastMathFlagGuard instead.
  IRBuilder(const IRBuilder &) = delete;

  InserterTy &getInserter() { return Inserter; }
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(IRBuilder<>, LLVMBuilderRef)

} // end namespace llvm

#endif // LLVM_IR_IRBUILDER_H
