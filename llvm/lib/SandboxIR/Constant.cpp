//===- Constant.cpp - The Constant classes of Sandbox IR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/Support/Compiler.h"

namespace llvm::sandboxir {

#ifndef NDEBUG
void Constant::dumpOS(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
#endif // NDEBUG

ConstantInt *ConstantInt::getTrue(Context &Ctx) {
  auto *LLVMC = llvm::ConstantInt::getTrue(Ctx.LLVMCtx);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::getFalse(Context &Ctx) {
  auto *LLVMC = llvm::ConstantInt::getFalse(Ctx.LLVMCtx);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::getBool(Context &Ctx, bool V) {
  auto *LLVMC = llvm::ConstantInt::getBool(Ctx.LLVMCtx, V);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
Constant *ConstantInt::getTrue(Type *Ty) {
  auto *LLVMC = llvm::ConstantInt::getTrue(Ty->LLVMTy);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
Constant *ConstantInt::getFalse(Type *Ty) {
  auto *LLVMC = llvm::ConstantInt::getFalse(Ty->LLVMTy);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
Constant *ConstantInt::getBool(Type *Ty, bool V) {
  auto *LLVMC = llvm::ConstantInt::getBool(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
ConstantInt *ConstantInt::get(Type *Ty, uint64_t V, bool IsSigned) {
  auto *LLVMC = llvm::ConstantInt::get(Ty->LLVMTy, V, IsSigned);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::get(IntegerType *Ty, uint64_t V, bool IsSigned) {
  auto *LLVMC = llvm::ConstantInt::get(Ty->LLVMTy, V, IsSigned);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::getSigned(IntegerType *Ty, int64_t V) {
  auto *LLVMC =
      llvm::ConstantInt::getSigned(cast<llvm::IntegerType>(Ty->LLVMTy), V);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantInt::getSigned(Type *Ty, int64_t V) {
  auto *LLVMC = llvm::ConstantInt::getSigned(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
ConstantInt *ConstantInt::get(Context &Ctx, const APInt &V) {
  auto *LLVMC = llvm::ConstantInt::get(Ctx.LLVMCtx, V);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::get(IntegerType *Ty, StringRef Str, uint8_t Radix) {
  auto *LLVMC =
      llvm::ConstantInt::get(cast<llvm::IntegerType>(Ty->LLVMTy), Str, Radix);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantInt::get(Type *Ty, const APInt &V) {
  auto *LLVMC = llvm::ConstantInt::get(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
IntegerType *ConstantInt::getIntegerType() const {
  auto *LLVMTy = cast<llvm::ConstantInt>(Val)->getIntegerType();
  return cast<IntegerType>(Ctx.getType(LLVMTy));
}

bool ConstantInt::isValueValidForType(Type *Ty, uint64_t V) {
  return llvm::ConstantInt::isValueValidForType(Ty->LLVMTy, V);
}
bool ConstantInt::isValueValidForType(Type *Ty, int64_t V) {
  return llvm::ConstantInt::isValueValidForType(Ty->LLVMTy, V);
}

Constant *ConstantFP::get(Type *Ty, double V) {
  auto *LLVMC = llvm::ConstantFP::get(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}

Constant *ConstantFP::get(Type *Ty, const APFloat &V) {
  auto *LLVMC = llvm::ConstantFP::get(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}

Constant *ConstantFP::get(Type *Ty, StringRef Str) {
  auto *LLVMC = llvm::ConstantFP::get(Ty->LLVMTy, Str);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}

ConstantFP *ConstantFP::get(const APFloat &V, Context &Ctx) {
  auto *LLVMC = llvm::ConstantFP::get(Ctx.LLVMCtx, V);
  return cast<ConstantFP>(Ctx.getOrCreateConstant(LLVMC));
}

Constant *ConstantFP::getNaN(Type *Ty, bool Negative, uint64_t Payload) {
  auto *LLVMC = llvm::ConstantFP::getNaN(Ty->LLVMTy, Negative, Payload);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getQNaN(Type *Ty, bool Negative, APInt *Payload) {
  auto *LLVMC = llvm::ConstantFP::getQNaN(Ty->LLVMTy, Negative, Payload);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getSNaN(Type *Ty, bool Negative, APInt *Payload) {
  auto *LLVMC = llvm::ConstantFP::getSNaN(Ty->LLVMTy, Negative, Payload);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getZero(Type *Ty, bool Negative) {
  auto *LLVMC = llvm::ConstantFP::getZero(Ty->LLVMTy, Negative);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getNegativeZero(Type *Ty) {
  auto *LLVMC = llvm::ConstantFP::getNegativeZero(Ty->LLVMTy);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getInfinity(Type *Ty, bool Negative) {
  auto *LLVMC = llvm::ConstantFP::getInfinity(Ty->LLVMTy, Negative);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
bool ConstantFP::isValueValidForType(Type *Ty, const APFloat &V) {
  return llvm::ConstantFP::isValueValidForType(Ty->LLVMTy, V);
}

Constant *ConstantArray::get(ArrayType *T, ArrayRef<Constant *> V) {
  auto &Ctx = T->getContext();
  SmallVector<llvm::Constant *> LLVMValues;
  LLVMValues.reserve(V.size());
  for (auto *Elm : V)
    LLVMValues.push_back(cast<llvm::Constant>(Elm->Val));
  auto *LLVMC =
      llvm::ConstantArray::get(cast<llvm::ArrayType>(T->LLVMTy), LLVMValues);
  return cast<ConstantArray>(Ctx.getOrCreateConstant(LLVMC));
}

ArrayType *ConstantArray::getType() const {
  return cast<ArrayType>(
      Ctx.getType(cast<llvm::ConstantArray>(Val)->getType()));
}

Constant *ConstantStruct::get(StructType *T, ArrayRef<Constant *> V) {
  auto &Ctx = T->getContext();
  SmallVector<llvm::Constant *> LLVMValues;
  LLVMValues.reserve(V.size());
  for (auto *Elm : V)
    LLVMValues.push_back(cast<llvm::Constant>(Elm->Val));
  auto *LLVMC =
      llvm::ConstantStruct::get(cast<llvm::StructType>(T->LLVMTy), LLVMValues);
  return cast<ConstantStruct>(Ctx.getOrCreateConstant(LLVMC));
}

StructType *ConstantStruct::getTypeForElements(Context &Ctx,
                                               ArrayRef<Constant *> V,
                                               bool Packed) {
  unsigned VecSize = V.size();
  SmallVector<Type *, 16> EltTypes;
  EltTypes.reserve(VecSize);
  for (Constant *Elm : V)
    EltTypes.push_back(Elm->getType());
  return StructType::get(Ctx, EltTypes, Packed);
}

Constant *ConstantVector::get(ArrayRef<Constant *> V) {
  assert(!V.empty() && "Expected non-empty V!");
  auto &Ctx = V[0]->getContext();
  SmallVector<llvm::Constant *, 8> LLVMV;
  LLVMV.reserve(V.size());
  for (auto *Elm : V)
    LLVMV.push_back(cast<llvm::Constant>(Elm->Val));
  return Ctx.getOrCreateConstant(llvm::ConstantVector::get(LLVMV));
}

Constant *ConstantVector::getSplat(ElementCount EC, Constant *Elt) {
  auto *LLVMElt = cast<llvm::Constant>(Elt->Val);
  auto &Ctx = Elt->getContext();
  return Ctx.getOrCreateConstant(llvm::ConstantVector::getSplat(EC, LLVMElt));
}

Constant *ConstantVector::getSplatValue(bool AllowPoison) const {
  auto *LLVMSplatValue = cast_or_null<llvm::Constant>(
      cast<llvm::ConstantVector>(Val)->getSplatValue(AllowPoison));
  return LLVMSplatValue ? Ctx.getOrCreateConstant(LLVMSplatValue) : nullptr;
}

ConstantAggregateZero *ConstantAggregateZero::get(Type *Ty) {
  auto *LLVMC = llvm::ConstantAggregateZero::get(Ty->LLVMTy);
  return cast<ConstantAggregateZero>(
      Ty->getContext().getOrCreateConstant(LLVMC));
}

Constant *ConstantAggregateZero::getSequentialElement() const {
  return cast<Constant>(Ctx.getValue(
      cast<llvm::ConstantAggregateZero>(Val)->getSequentialElement()));
}
Constant *ConstantAggregateZero::getStructElement(unsigned Elt) const {
  return cast<Constant>(Ctx.getValue(
      cast<llvm::ConstantAggregateZero>(Val)->getStructElement(Elt)));
}
Constant *ConstantAggregateZero::getElementValue(Constant *C) const {
  return cast<Constant>(
      Ctx.getValue(cast<llvm::ConstantAggregateZero>(Val)->getElementValue(
          cast<llvm::Constant>(C->Val))));
}
Constant *ConstantAggregateZero::getElementValue(unsigned Idx) const {
  return cast<Constant>(Ctx.getValue(
      cast<llvm::ConstantAggregateZero>(Val)->getElementValue(Idx)));
}

ConstantPointerNull *ConstantPointerNull::get(PointerType *Ty) {
  auto *LLVMC =
      llvm::ConstantPointerNull::get(cast<llvm::PointerType>(Ty->LLVMTy));
  return cast<ConstantPointerNull>(Ty->getContext().getOrCreateConstant(LLVMC));
}

PointerType *ConstantPointerNull::getType() const {
  return cast<PointerType>(
      Ctx.getType(cast<llvm::ConstantPointerNull>(Val)->getType()));
}

UndefValue *UndefValue::get(Type *T) {
  auto *LLVMC = llvm::UndefValue::get(T->LLVMTy);
  return cast<UndefValue>(T->getContext().getOrCreateConstant(LLVMC));
}

UndefValue *UndefValue::getSequentialElement() const {
  return cast<UndefValue>(Ctx.getOrCreateConstant(
      cast<llvm::UndefValue>(Val)->getSequentialElement()));
}

UndefValue *UndefValue::getStructElement(unsigned Elt) const {
  return cast<UndefValue>(Ctx.getOrCreateConstant(
      cast<llvm::UndefValue>(Val)->getStructElement(Elt)));
}

UndefValue *UndefValue::getElementValue(Constant *C) const {
  return cast<UndefValue>(
      Ctx.getOrCreateConstant(cast<llvm::UndefValue>(Val)->getElementValue(
          cast<llvm::Constant>(C->Val))));
}

UndefValue *UndefValue::getElementValue(unsigned Idx) const {
  return cast<UndefValue>(Ctx.getOrCreateConstant(
      cast<llvm::UndefValue>(Val)->getElementValue(Idx)));
}

PoisonValue *PoisonValue::get(Type *T) {
  auto *LLVMC = llvm::PoisonValue::get(T->LLVMTy);
  return cast<PoisonValue>(T->getContext().getOrCreateConstant(LLVMC));
}

PoisonValue *PoisonValue::getSequentialElement() const {
  return cast<PoisonValue>(Ctx.getOrCreateConstant(
      cast<llvm::PoisonValue>(Val)->getSequentialElement()));
}

PoisonValue *PoisonValue::getStructElement(unsigned Elt) const {
  return cast<PoisonValue>(Ctx.getOrCreateConstant(
      cast<llvm::PoisonValue>(Val)->getStructElement(Elt)));
}

PoisonValue *PoisonValue::getElementValue(Constant *C) const {
  return cast<PoisonValue>(
      Ctx.getOrCreateConstant(cast<llvm::PoisonValue>(Val)->getElementValue(
          cast<llvm::Constant>(C->Val))));
}

PoisonValue *PoisonValue::getElementValue(unsigned Idx) const {
  return cast<PoisonValue>(Ctx.getOrCreateConstant(
      cast<llvm::PoisonValue>(Val)->getElementValue(Idx)));
}

void GlobalVariable::setAlignment(MaybeAlign Align) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalVariable::getAlign,
                                       &GlobalVariable::setAlignment>>(this);
  cast<llvm::GlobalVariable>(Val)->setAlignment(Align);
}

void GlobalObject::setSection(StringRef S) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalObject::getSection, &GlobalObject::setSection>>(
          this);
  cast<llvm::GlobalObject>(Val)->setSection(S);
}

template <typename GlobalT, typename LLVMGlobalT, typename ParentT,
          typename LLVMParentT>
GlobalT &GlobalWithNodeAPI<GlobalT, LLVMGlobalT, ParentT, LLVMParentT>::
    LLVMGVToGV::operator()(LLVMGlobalT &LLVMGV) const {
  return cast<GlobalT>(*Ctx.getValue(&LLVMGV));
}

// Explicit instantiations.
template class LLVM_EXPORT_TEMPLATE GlobalWithNodeAPI<
    GlobalIFunc, llvm::GlobalIFunc, GlobalObject, llvm::GlobalObject>;
template class LLVM_EXPORT_TEMPLATE GlobalWithNodeAPI<
    Function, llvm::Function, GlobalObject, llvm::GlobalObject>;
template class LLVM_EXPORT_TEMPLATE GlobalWithNodeAPI<
    GlobalVariable, llvm::GlobalVariable, GlobalObject, llvm::GlobalObject>;
template class LLVM_EXPORT_TEMPLATE GlobalWithNodeAPI<
    GlobalAlias, llvm::GlobalAlias, GlobalValue, llvm::GlobalValue>;

void GlobalIFunc::setResolver(Constant *Resolver) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalIFunc::getResolver, &GlobalIFunc::setResolver>>(
          this);
  cast<llvm::GlobalIFunc>(Val)->setResolver(
      cast<llvm::Constant>(Resolver->Val));
}

Constant *GlobalIFunc::getResolver() const {
  return Ctx.getOrCreateConstant(cast<llvm::GlobalIFunc>(Val)->getResolver());
}

Function *GlobalIFunc::getResolverFunction() {
  return cast<Function>(Ctx.getOrCreateConstant(
      cast<llvm::GlobalIFunc>(Val)->getResolverFunction()));
}

GlobalVariable &
GlobalVariable::LLVMGVToGV::operator()(llvm::GlobalVariable &LLVMGV) const {
  return cast<GlobalVariable>(*Ctx.getValue(&LLVMGV));
}

Constant *GlobalVariable::getInitializer() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::GlobalVariable>(Val)->getInitializer());
}

void GlobalVariable::setInitializer(Constant *InitVal) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalVariable::getInitializer,
                                       &GlobalVariable::setInitializer>>(this);
  cast<llvm::GlobalVariable>(Val)->setInitializer(
      cast<llvm::Constant>(InitVal->Val));
}

void GlobalVariable::setConstant(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalVariable::isConstant,
                                       &GlobalVariable::setConstant>>(this);
  cast<llvm::GlobalVariable>(Val)->setConstant(V);
}

void GlobalVariable::setExternallyInitialized(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalVariable::isExternallyInitialized,
                        &GlobalVariable::setExternallyInitialized>>(this);
  cast<llvm::GlobalVariable>(Val)->setExternallyInitialized(V);
}

void GlobalAlias::setAliasee(Constant *Aliasee) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalAlias::getAliasee, &GlobalAlias::setAliasee>>(
          this);
  cast<llvm::GlobalAlias>(Val)->setAliasee(cast<llvm::Constant>(Aliasee->Val));
}

Constant *GlobalAlias::getAliasee() const {
  return cast<Constant>(
      Ctx.getOrCreateConstant(cast<llvm::GlobalAlias>(Val)->getAliasee()));
}

const GlobalObject *GlobalAlias::getAliaseeObject() const {
  return cast<GlobalObject>(Ctx.getOrCreateConstant(
      cast<llvm::GlobalAlias>(Val)->getAliaseeObject()));
}

void GlobalValue::setUnnamedAddr(UnnamedAddr V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalValue::getUnnamedAddr,
                                       &GlobalValue::setUnnamedAddr>>(this);
  cast<llvm::GlobalValue>(Val)->setUnnamedAddr(V);
}

void GlobalValue::setVisibility(VisibilityTypes V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalValue::getVisibility,
                                       &GlobalValue::setVisibility>>(this);
  cast<llvm::GlobalValue>(Val)->setVisibility(V);
}

NoCFIValue *NoCFIValue::get(GlobalValue *GV) {
  auto *LLVMC = llvm::NoCFIValue::get(cast<llvm::GlobalValue>(GV->Val));
  return cast<NoCFIValue>(GV->getContext().getOrCreateConstant(LLVMC));
}

GlobalValue *NoCFIValue::getGlobalValue() const {
  auto *LLVMC = cast<llvm::NoCFIValue>(Val)->getGlobalValue();
  return cast<GlobalValue>(Ctx.getOrCreateConstant(LLVMC));
}

PointerType *NoCFIValue::getType() const {
  return cast<PointerType>(Ctx.getType(cast<llvm::NoCFIValue>(Val)->getType()));
}

ConstantPtrAuth *ConstantPtrAuth::get(Constant *Ptr, ConstantInt *Key,
                                      ConstantInt *Disc, Constant *AddrDisc) {
  auto *LLVMC = llvm::ConstantPtrAuth::get(
      cast<llvm::Constant>(Ptr->Val), cast<llvm::ConstantInt>(Key->Val),
      cast<llvm::ConstantInt>(Disc->Val), cast<llvm::Constant>(AddrDisc->Val));
  return cast<ConstantPtrAuth>(Ptr->getContext().getOrCreateConstant(LLVMC));
}

Constant *ConstantPtrAuth::getPointer() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::ConstantPtrAuth>(Val)->getPointer());
}

ConstantInt *ConstantPtrAuth::getKey() const {
  return cast<ConstantInt>(
      Ctx.getOrCreateConstant(cast<llvm::ConstantPtrAuth>(Val)->getKey()));
}

ConstantInt *ConstantPtrAuth::getDiscriminator() const {
  return cast<ConstantInt>(Ctx.getOrCreateConstant(
      cast<llvm::ConstantPtrAuth>(Val)->getDiscriminator()));
}

Constant *ConstantPtrAuth::getAddrDiscriminator() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::ConstantPtrAuth>(Val)->getAddrDiscriminator());
}

ConstantPtrAuth *ConstantPtrAuth::getWithSameSchema(Constant *Pointer) const {
  auto *LLVMC = cast<llvm::ConstantPtrAuth>(Val)->getWithSameSchema(
      cast<llvm::Constant>(Pointer->Val));
  return cast<ConstantPtrAuth>(Ctx.getOrCreateConstant(LLVMC));
}

BlockAddress *BlockAddress::get(Function *F, BasicBlock *BB) {
  auto *LLVMC = llvm::BlockAddress::get(cast<llvm::Function>(F->Val),
                                        cast<llvm::BasicBlock>(BB->Val));
  return cast<BlockAddress>(F->getContext().getOrCreateConstant(LLVMC));
}

BlockAddress *BlockAddress::get(BasicBlock *BB) {
  auto *LLVMC = llvm::BlockAddress::get(cast<llvm::BasicBlock>(BB->Val));
  return cast<BlockAddress>(BB->getContext().getOrCreateConstant(LLVMC));
}

BlockAddress *BlockAddress::lookup(const BasicBlock *BB) {
  auto *LLVMC = llvm::BlockAddress::lookup(cast<llvm::BasicBlock>(BB->Val));
  return cast_or_null<BlockAddress>(BB->getContext().getValue(LLVMC));
}

Function *BlockAddress::getFunction() const {
  return cast<Function>(
      Ctx.getValue(cast<llvm::BlockAddress>(Val)->getFunction()));
}

BasicBlock *BlockAddress::getBasicBlock() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::BlockAddress>(Val)->getBasicBlock()));
}

DSOLocalEquivalent *DSOLocalEquivalent::get(GlobalValue *GV) {
  auto *LLVMC = llvm::DSOLocalEquivalent::get(cast<llvm::GlobalValue>(GV->Val));
  return cast<DSOLocalEquivalent>(GV->getContext().getValue(LLVMC));
}

GlobalValue *DSOLocalEquivalent::getGlobalValue() const {
  return cast<GlobalValue>(
      Ctx.getValue(cast<llvm::DSOLocalEquivalent>(Val)->getGlobalValue()));
}

} // namespace llvm::sandboxir
