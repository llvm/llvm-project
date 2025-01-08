//===- Function.cpp - Implement the Global object classes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Function class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/SymbolTableListTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ModRef.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

using namespace llvm;
using ProfileCount = Function::ProfileCount;

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file...
template class llvm::SymbolTableListTraits<BasicBlock>;

static cl::opt<int> NonGlobalValueMaxNameSize(
    "non-global-value-max-name-size", cl::Hidden, cl::init(1024),
    cl::desc("Maximum size for the name of non-global values."));

extern cl::opt<bool> UseNewDbgInfoFormat;

void Function::renumberBlocks() {
  validateBlockNumbers();

  NextBlockNum = 0;
  for (auto &BB : *this)
    BB.Number = NextBlockNum++;
  BlockNumEpoch++;
}

void Function::validateBlockNumbers() const {
#ifndef NDEBUG
  BitVector Numbers(NextBlockNum);
  for (const auto &BB : *this) {
    unsigned Num = BB.getNumber();
    assert(Num < NextBlockNum && "out of range block number");
    assert(!Numbers[Num] && "duplicate block numbers");
    Numbers.set(Num);
  }
#endif
}

void Function::convertToNewDbgValues() {
  IsNewDbgInfoFormat = true;
  for (auto &BB : *this) {
    BB.convertToNewDbgValues();
  }
}

void Function::convertFromNewDbgValues() {
  IsNewDbgInfoFormat = false;
  for (auto &BB : *this) {
    BB.convertFromNewDbgValues();
  }
}

void Function::setIsNewDbgInfoFormat(bool NewFlag) {
  if (NewFlag && !IsNewDbgInfoFormat)
    convertToNewDbgValues();
  else if (!NewFlag && IsNewDbgInfoFormat)
    convertFromNewDbgValues();
}
void Function::setNewDbgInfoFormatFlag(bool NewFlag) {
  for (auto &BB : *this) {
    BB.setNewDbgInfoFormatFlag(NewFlag);
  }
  IsNewDbgInfoFormat = NewFlag;
}

//===----------------------------------------------------------------------===//
// Argument Implementation
//===----------------------------------------------------------------------===//

Argument::Argument(Type *Ty, const Twine &Name, Function *Par, unsigned ArgNo)
    : Value(Ty, Value::ArgumentVal), Parent(Par), ArgNo(ArgNo) {
  setName(Name);
}

void Argument::setParent(Function *parent) {
  Parent = parent;
}

bool Argument::hasNonNullAttr(bool AllowUndefOrPoison) const {
  if (!getType()->isPointerTy()) return false;
  if (getParent()->hasParamAttribute(getArgNo(), Attribute::NonNull) &&
      (AllowUndefOrPoison ||
       getParent()->hasParamAttribute(getArgNo(), Attribute::NoUndef)))
    return true;
  else if (getDereferenceableBytes() > 0 &&
           !NullPointerIsDefined(getParent(),
                                 getType()->getPointerAddressSpace()))
    return true;
  return false;
}

bool Argument::hasByValAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::ByVal);
}

bool Argument::hasByRefAttr() const {
  if (!getType()->isPointerTy())
    return false;
  return hasAttribute(Attribute::ByRef);
}

bool Argument::hasSwiftSelfAttr() const {
  return getParent()->hasParamAttribute(getArgNo(), Attribute::SwiftSelf);
}

bool Argument::hasSwiftErrorAttr() const {
  return getParent()->hasParamAttribute(getArgNo(), Attribute::SwiftError);
}

bool Argument::hasInAllocaAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::InAlloca);
}

bool Argument::hasPreallocatedAttr() const {
  if (!getType()->isPointerTy())
    return false;
  return hasAttribute(Attribute::Preallocated);
}

bool Argument::hasPassPointeeByValueCopyAttr() const {
  if (!getType()->isPointerTy()) return false;
  AttributeList Attrs = getParent()->getAttributes();
  return Attrs.hasParamAttr(getArgNo(), Attribute::ByVal) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::InAlloca) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::Preallocated);
}

bool Argument::hasPointeeInMemoryValueAttr() const {
  if (!getType()->isPointerTy())
    return false;
  AttributeList Attrs = getParent()->getAttributes();
  return Attrs.hasParamAttr(getArgNo(), Attribute::ByVal) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::StructRet) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::InAlloca) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::Preallocated) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::ByRef);
}

/// For a byval, sret, inalloca, or preallocated parameter, get the in-memory
/// parameter type.
static Type *getMemoryParamAllocType(AttributeSet ParamAttrs) {
  // FIXME: All the type carrying attributes are mutually exclusive, so there
  // should be a single query to get the stored type that handles any of them.
  if (Type *ByValTy = ParamAttrs.getByValType())
    return ByValTy;
  if (Type *ByRefTy = ParamAttrs.getByRefType())
    return ByRefTy;
  if (Type *PreAllocTy = ParamAttrs.getPreallocatedType())
    return PreAllocTy;
  if (Type *InAllocaTy = ParamAttrs.getInAllocaType())
    return InAllocaTy;
  if (Type *SRetTy = ParamAttrs.getStructRetType())
    return SRetTy;

  return nullptr;
}

uint64_t Argument::getPassPointeeByValueCopySize(const DataLayout &DL) const {
  AttributeSet ParamAttrs =
      getParent()->getAttributes().getParamAttrs(getArgNo());
  if (Type *MemTy = getMemoryParamAllocType(ParamAttrs))
    return DL.getTypeAllocSize(MemTy);
  return 0;
}

Type *Argument::getPointeeInMemoryValueType() const {
  AttributeSet ParamAttrs =
      getParent()->getAttributes().getParamAttrs(getArgNo());
  return getMemoryParamAllocType(ParamAttrs);
}

MaybeAlign Argument::getParamAlign() const {
  assert(getType()->isPointerTy() && "Only pointers have alignments");
  return getParent()->getParamAlign(getArgNo());
}

MaybeAlign Argument::getParamStackAlign() const {
  return getParent()->getParamStackAlign(getArgNo());
}

Type *Argument::getParamByValType() const {
  assert(getType()->isPointerTy() && "Only pointers have byval types");
  return getParent()->getParamByValType(getArgNo());
}

Type *Argument::getParamStructRetType() const {
  assert(getType()->isPointerTy() && "Only pointers have sret types");
  return getParent()->getParamStructRetType(getArgNo());
}

Type *Argument::getParamByRefType() const {
  assert(getType()->isPointerTy() && "Only pointers have byref types");
  return getParent()->getParamByRefType(getArgNo());
}

Type *Argument::getParamInAllocaType() const {
  assert(getType()->isPointerTy() && "Only pointers have inalloca types");
  return getParent()->getParamInAllocaType(getArgNo());
}

uint64_t Argument::getDereferenceableBytes() const {
  assert(getType()->isPointerTy() &&
         "Only pointers have dereferenceable bytes");
  return getParent()->getParamDereferenceableBytes(getArgNo());
}

uint64_t Argument::getDereferenceableOrNullBytes() const {
  assert(getType()->isPointerTy() &&
         "Only pointers have dereferenceable bytes");
  return getParent()->getParamDereferenceableOrNullBytes(getArgNo());
}

FPClassTest Argument::getNoFPClass() const {
  return getParent()->getParamNoFPClass(getArgNo());
}

std::optional<ConstantRange> Argument::getRange() const {
  const Attribute RangeAttr = getAttribute(llvm::Attribute::Range);
  if (RangeAttr.isValid())
    return RangeAttr.getRange();
  return std::nullopt;
}

bool Argument::hasNestAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::Nest);
}

bool Argument::hasNoAliasAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::NoAlias);
}

bool Argument::hasNoCaptureAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::NoCapture);
}

bool Argument::hasNoFreeAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::NoFree);
}

bool Argument::hasStructRetAttr() const {
  if (!getType()->isPointerTy()) return false;
  return hasAttribute(Attribute::StructRet);
}

bool Argument::hasInRegAttr() const {
  return hasAttribute(Attribute::InReg);
}

bool Argument::hasReturnedAttr() const {
  return hasAttribute(Attribute::Returned);
}

bool Argument::hasZExtAttr() const {
  return hasAttribute(Attribute::ZExt);
}

bool Argument::hasSExtAttr() const {
  return hasAttribute(Attribute::SExt);
}

bool Argument::onlyReadsMemory() const {
  AttributeList Attrs = getParent()->getAttributes();
  return Attrs.hasParamAttr(getArgNo(), Attribute::ReadOnly) ||
         Attrs.hasParamAttr(getArgNo(), Attribute::ReadNone);
}

void Argument::addAttrs(AttrBuilder &B) {
  AttributeList AL = getParent()->getAttributes();
  AL = AL.addParamAttributes(Parent->getContext(), getArgNo(), B);
  getParent()->setAttributes(AL);
}

void Argument::addAttr(Attribute::AttrKind Kind) {
  getParent()->addParamAttr(getArgNo(), Kind);
}

void Argument::addAttr(Attribute Attr) {
  getParent()->addParamAttr(getArgNo(), Attr);
}

void Argument::removeAttr(Attribute::AttrKind Kind) {
  getParent()->removeParamAttr(getArgNo(), Kind);
}

void Argument::removeAttrs(const AttributeMask &AM) {
  AttributeList AL = getParent()->getAttributes();
  AL = AL.removeParamAttributes(Parent->getContext(), getArgNo(), AM);
  getParent()->setAttributes(AL);
}

bool Argument::hasAttribute(Attribute::AttrKind Kind) const {
  return getParent()->hasParamAttribute(getArgNo(), Kind);
}

bool Argument::hasAttribute(StringRef Kind) const {
  return getParent()->hasParamAttribute(getArgNo(), Kind);
}

Attribute Argument::getAttribute(Attribute::AttrKind Kind) const {
  return getParent()->getParamAttribute(getArgNo(), Kind);
}

AttributeSet Argument::getAttributes() const {
  return getParent()->getAttributes().getParamAttrs(getArgNo());
}

//===----------------------------------------------------------------------===//
// Helper Methods in Function
//===----------------------------------------------------------------------===//

LLVMContext &Function::getContext() const {
  return getType()->getContext();
}

const DataLayout &Function::getDataLayout() const {
  return getParent()->getDataLayout();
}

unsigned Function::getInstructionCount() const {
  unsigned NumInstrs = 0;
  for (const BasicBlock &BB : BasicBlocks)
    NumInstrs += std::distance(BB.instructionsWithoutDebug().begin(),
                               BB.instructionsWithoutDebug().end());
  return NumInstrs;
}

Function *Function::Create(FunctionType *Ty, LinkageTypes Linkage,
                           const Twine &N, Module &M) {
  return Create(Ty, Linkage, M.getDataLayout().getProgramAddressSpace(), N, &M);
}

Function *Function::createWithDefaultAttr(FunctionType *Ty,
                                          LinkageTypes Linkage,
                                          unsigned AddrSpace, const Twine &N,
                                          Module *M) {
  auto *F = new (AllocMarker) Function(Ty, Linkage, AddrSpace, N, M);
  AttrBuilder B(F->getContext());
  UWTableKind UWTable = M->getUwtable();
  if (UWTable != UWTableKind::None)
    B.addUWTableAttr(UWTable);
  switch (M->getFramePointer()) {
  case FramePointerKind::None:
    // 0 ("none") is the default.
    break;
  case FramePointerKind::Reserved:
    B.addAttribute("frame-pointer", "reserved");
    break;
  case FramePointerKind::NonLeaf:
    B.addAttribute("frame-pointer", "non-leaf");
    break;
  case FramePointerKind::All:
    B.addAttribute("frame-pointer", "all");
    break;
  }
  if (M->getModuleFlag("function_return_thunk_extern"))
    B.addAttribute(Attribute::FnRetThunkExtern);
  StringRef DefaultCPU = F->getContext().getDefaultTargetCPU();
  if (!DefaultCPU.empty())
    B.addAttribute("target-cpu", DefaultCPU);
  StringRef DefaultFeatures = F->getContext().getDefaultTargetFeatures();
  if (!DefaultFeatures.empty())
    B.addAttribute("target-features", DefaultFeatures);

  // Check if the module attribute is present and not zero.
  auto isModuleAttributeSet = [&](const StringRef &ModAttr) -> bool {
    const auto *Attr =
        mdconst::extract_or_null<ConstantInt>(M->getModuleFlag(ModAttr));
    return Attr && !Attr->isZero();
  };

  auto AddAttributeIfSet = [&](const StringRef &ModAttr) {
    if (isModuleAttributeSet(ModAttr))
      B.addAttribute(ModAttr);
  };

  StringRef SignType = "none";
  if (isModuleAttributeSet("sign-return-address"))
    SignType = "non-leaf";
  if (isModuleAttributeSet("sign-return-address-all"))
    SignType = "all";
  if (SignType != "none") {
    B.addAttribute("sign-return-address", SignType);
    B.addAttribute("sign-return-address-key",
                   isModuleAttributeSet("sign-return-address-with-bkey")
                       ? "b_key"
                       : "a_key");
  }
  AddAttributeIfSet("branch-target-enforcement");
  AddAttributeIfSet("branch-protection-pauth-lr");
  AddAttributeIfSet("guarded-control-stack");

  F->addFnAttrs(B);
  return F;
}

void Function::removeFromParent() {
  getParent()->getFunctionList().remove(getIterator());
}

void Function::eraseFromParent() {
  getParent()->getFunctionList().erase(getIterator());
}

void Function::splice(Function::iterator ToIt, Function *FromF,
                      Function::iterator FromBeginIt,
                      Function::iterator FromEndIt) {
#ifdef EXPENSIVE_CHECKS
  // Check that FromBeginIt is before FromEndIt.
  auto FromFEnd = FromF->end();
  for (auto It = FromBeginIt; It != FromEndIt; ++It)
    assert(It != FromFEnd && "FromBeginIt not before FromEndIt!");
#endif // EXPENSIVE_CHECKS
  BasicBlocks.splice(ToIt, FromF->BasicBlocks, FromBeginIt, FromEndIt);
}

Function::iterator Function::erase(Function::iterator FromIt,
                                   Function::iterator ToIt) {
  return BasicBlocks.erase(FromIt, ToIt);
}

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//

static unsigned computeAddrSpace(unsigned AddrSpace, Module *M) {
  // If AS == -1 and we are passed a valid module pointer we place the function
  // in the program address space. Otherwise we default to AS0.
  if (AddrSpace == static_cast<unsigned>(-1))
    return M ? M->getDataLayout().getProgramAddressSpace() : 0;
  return AddrSpace;
}

Function::Function(FunctionType *Ty, LinkageTypes Linkage, unsigned AddrSpace,
                   const Twine &name, Module *ParentModule)
    : GlobalObject(Ty, Value::FunctionVal, AllocMarker, Linkage, name,
                   computeAddrSpace(AddrSpace, ParentModule)),
      NumArgs(Ty->getNumParams()), IsNewDbgInfoFormat(UseNewDbgInfoFormat) {
  assert(FunctionType::isValidReturnType(getReturnType()) &&
         "invalid return type");
  setGlobalObjectSubClassData(0);

  // We only need a symbol table for a function if the context keeps value names
  if (!getContext().shouldDiscardValueNames())
    SymTab = std::make_unique<ValueSymbolTable>(NonGlobalValueMaxNameSize);

  // If the function has arguments, mark them as lazily built.
  if (Ty->getNumParams())
    setValueSubclassData(1);   // Set the "has lazy arguments" bit.

  if (ParentModule) {
    ParentModule->getFunctionList().push_back(this);
    IsNewDbgInfoFormat = ParentModule->IsNewDbgInfoFormat;
  }

  HasLLVMReservedName = getName().starts_with("llvm.");
  // Ensure intrinsics have the right parameter attributes.
  // Note, the IntID field will have been set in Value::setName if this function
  // name is a valid intrinsic ID.
  if (IntID)
    setAttributes(Intrinsic::getAttributes(getContext(), IntID));
}

Function::~Function() {
  validateBlockNumbers();

  dropAllReferences();    // After this it is safe to delete instructions.

  // Delete all of the method arguments and unlink from symbol table...
  if (Arguments)
    clearArguments();

  // Remove the function from the on-the-side GC table.
  clearGC();
}

void Function::BuildLazyArguments() const {
  // Create the arguments vector, all arguments start out unnamed.
  auto *FT = getFunctionType();
  if (NumArgs > 0) {
    Arguments = std::allocator<Argument>().allocate(NumArgs);
    for (unsigned i = 0, e = NumArgs; i != e; ++i) {
      Type *ArgTy = FT->getParamType(i);
      assert(!ArgTy->isVoidTy() && "Cannot have void typed arguments!");
      new (Arguments + i) Argument(ArgTy, "", const_cast<Function *>(this), i);
    }
  }

  // Clear the lazy arguments bit.
  unsigned SDC = getSubclassDataFromValue();
  SDC &= ~(1 << 0);
  const_cast<Function*>(this)->setValueSubclassData(SDC);
  assert(!hasLazyArguments());
}

static MutableArrayRef<Argument> makeArgArray(Argument *Args, size_t Count) {
  return MutableArrayRef<Argument>(Args, Count);
}

bool Function::isConstrainedFPIntrinsic() const {
  return Intrinsic::isConstrainedFPIntrinsic(getIntrinsicID());
}

void Function::clearArguments() {
  for (Argument &A : makeArgArray(Arguments, NumArgs)) {
    A.setName("");
    A.~Argument();
  }
  std::allocator<Argument>().deallocate(Arguments, NumArgs);
  Arguments = nullptr;
}

void Function::stealArgumentListFrom(Function &Src) {
  assert(isDeclaration() && "Expected no references to current arguments");

  // Drop the current arguments, if any, and set the lazy argument bit.
  if (!hasLazyArguments()) {
    assert(llvm::all_of(makeArgArray(Arguments, NumArgs),
                        [](const Argument &A) { return A.use_empty(); }) &&
           "Expected arguments to be unused in declaration");
    clearArguments();
    setValueSubclassData(getSubclassDataFromValue() | (1 << 0));
  }

  // Nothing to steal if Src has lazy arguments.
  if (Src.hasLazyArguments())
    return;

  // Steal arguments from Src, and fix the lazy argument bits.
  assert(arg_size() == Src.arg_size());
  Arguments = Src.Arguments;
  Src.Arguments = nullptr;
  for (Argument &A : makeArgArray(Arguments, NumArgs)) {
    // FIXME: This does the work of transferNodesFromList inefficiently.
    SmallString<128> Name;
    if (A.hasName())
      Name = A.getName();
    if (!Name.empty())
      A.setName("");
    A.setParent(this);
    if (!Name.empty())
      A.setName(Name);
  }

  setValueSubclassData(getSubclassDataFromValue() & ~(1 << 0));
  assert(!hasLazyArguments());
  Src.setValueSubclassData(Src.getSubclassDataFromValue() | (1 << 0));
}

void Function::deleteBodyImpl(bool ShouldDrop) {
  setIsMaterializable(false);

  for (BasicBlock &BB : *this)
    BB.dropAllReferences();

  // Delete all basic blocks. They are now unused, except possibly by
  // blockaddresses, but BasicBlock's destructor takes care of those.
  while (!BasicBlocks.empty())
    BasicBlocks.begin()->eraseFromParent();

  if (getNumOperands()) {
    if (ShouldDrop) {
      // Drop uses of any optional data (real or placeholder).
      User::dropAllReferences();
      setNumHungOffUseOperands(0);
    } else {
      // The code needs to match Function::allocHungoffUselist().
      auto *CPN = ConstantPointerNull::get(PointerType::get(getContext(), 0));
      Op<0>().set(CPN);
      Op<1>().set(CPN);
      Op<2>().set(CPN);
    }
    setValueSubclassData(getSubclassDataFromValue() & ~0xe);
  }

  // Metadata is stored in a side-table.
  clearMetadata();
}

void Function::addAttributeAtIndex(unsigned i, Attribute Attr) {
  AttributeSets = AttributeSets.addAttributeAtIndex(getContext(), i, Attr);
}

void Function::addFnAttr(Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.addFnAttribute(getContext(), Kind);
}

void Function::addFnAttr(StringRef Kind, StringRef Val) {
  AttributeSets = AttributeSets.addFnAttribute(getContext(), Kind, Val);
}

void Function::addFnAttr(Attribute Attr) {
  AttributeSets = AttributeSets.addFnAttribute(getContext(), Attr);
}

void Function::addFnAttrs(const AttrBuilder &Attrs) {
  AttributeSets = AttributeSets.addFnAttributes(getContext(), Attrs);
}

void Function::addRetAttr(Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.addRetAttribute(getContext(), Kind);
}

void Function::addRetAttr(Attribute Attr) {
  AttributeSets = AttributeSets.addRetAttribute(getContext(), Attr);
}

void Function::addRetAttrs(const AttrBuilder &Attrs) {
  AttributeSets = AttributeSets.addRetAttributes(getContext(), Attrs);
}

void Function::addParamAttr(unsigned ArgNo, Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.addParamAttribute(getContext(), ArgNo, Kind);
}

void Function::addParamAttr(unsigned ArgNo, Attribute Attr) {
  AttributeSets = AttributeSets.addParamAttribute(getContext(), ArgNo, Attr);
}

void Function::addParamAttrs(unsigned ArgNo, const AttrBuilder &Attrs) {
  AttributeSets = AttributeSets.addParamAttributes(getContext(), ArgNo, Attrs);
}

void Function::removeAttributeAtIndex(unsigned i, Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.removeAttributeAtIndex(getContext(), i, Kind);
}

void Function::removeAttributeAtIndex(unsigned i, StringRef Kind) {
  AttributeSets = AttributeSets.removeAttributeAtIndex(getContext(), i, Kind);
}

void Function::removeFnAttr(Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.removeFnAttribute(getContext(), Kind);
}

void Function::removeFnAttr(StringRef Kind) {
  AttributeSets = AttributeSets.removeFnAttribute(getContext(), Kind);
}

void Function::removeFnAttrs(const AttributeMask &AM) {
  AttributeSets = AttributeSets.removeFnAttributes(getContext(), AM);
}

void Function::removeRetAttr(Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.removeRetAttribute(getContext(), Kind);
}

void Function::removeRetAttr(StringRef Kind) {
  AttributeSets = AttributeSets.removeRetAttribute(getContext(), Kind);
}

void Function::removeRetAttrs(const AttributeMask &Attrs) {
  AttributeSets = AttributeSets.removeRetAttributes(getContext(), Attrs);
}

void Function::removeParamAttr(unsigned ArgNo, Attribute::AttrKind Kind) {
  AttributeSets = AttributeSets.removeParamAttribute(getContext(), ArgNo, Kind);
}

void Function::removeParamAttr(unsigned ArgNo, StringRef Kind) {
  AttributeSets = AttributeSets.removeParamAttribute(getContext(), ArgNo, Kind);
}

void Function::removeParamAttrs(unsigned ArgNo, const AttributeMask &Attrs) {
  AttributeSets =
      AttributeSets.removeParamAttributes(getContext(), ArgNo, Attrs);
}

void Function::addDereferenceableParamAttr(unsigned ArgNo, uint64_t Bytes) {
  AttributeSets =
      AttributeSets.addDereferenceableParamAttr(getContext(), ArgNo, Bytes);
}

bool Function::hasFnAttribute(Attribute::AttrKind Kind) const {
  return AttributeSets.hasFnAttr(Kind);
}

bool Function::hasFnAttribute(StringRef Kind) const {
  return AttributeSets.hasFnAttr(Kind);
}

bool Function::hasRetAttribute(Attribute::AttrKind Kind) const {
  return AttributeSets.hasRetAttr(Kind);
}

bool Function::hasParamAttribute(unsigned ArgNo,
                                 Attribute::AttrKind Kind) const {
  return AttributeSets.hasParamAttr(ArgNo, Kind);
}

bool Function::hasParamAttribute(unsigned ArgNo, StringRef Kind) const {
  return AttributeSets.hasParamAttr(ArgNo, Kind);
}

Attribute Function::getAttributeAtIndex(unsigned i,
                                        Attribute::AttrKind Kind) const {
  return AttributeSets.getAttributeAtIndex(i, Kind);
}

Attribute Function::getAttributeAtIndex(unsigned i, StringRef Kind) const {
  return AttributeSets.getAttributeAtIndex(i, Kind);
}

bool Function::hasAttributeAtIndex(unsigned Idx,
                                   Attribute::AttrKind Kind) const {
  return AttributeSets.hasAttributeAtIndex(Idx, Kind);
}

Attribute Function::getFnAttribute(Attribute::AttrKind Kind) const {
  return AttributeSets.getFnAttr(Kind);
}

Attribute Function::getFnAttribute(StringRef Kind) const {
  return AttributeSets.getFnAttr(Kind);
}

Attribute Function::getRetAttribute(Attribute::AttrKind Kind) const {
  return AttributeSets.getRetAttr(Kind);
}

uint64_t Function::getFnAttributeAsParsedInteger(StringRef Name,
                                                 uint64_t Default) const {
  Attribute A = getFnAttribute(Name);
  uint64_t Result = Default;
  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, Result))
      getContext().emitError("cannot parse integer attribute " + Name);
  }

  return Result;
}

/// gets the specified attribute from the list of attributes.
Attribute Function::getParamAttribute(unsigned ArgNo,
                                      Attribute::AttrKind Kind) const {
  return AttributeSets.getParamAttr(ArgNo, Kind);
}

void Function::addDereferenceableOrNullParamAttr(unsigned ArgNo,
                                                 uint64_t Bytes) {
  AttributeSets = AttributeSets.addDereferenceableOrNullParamAttr(getContext(),
                                                                  ArgNo, Bytes);
}

void Function::addRangeRetAttr(const ConstantRange &CR) {
  AttributeSets = AttributeSets.addRangeRetAttr(getContext(), CR);
}

DenormalMode Function::getDenormalMode(const fltSemantics &FPType) const {
  if (&FPType == &APFloat::IEEEsingle()) {
    DenormalMode Mode = getDenormalModeF32Raw();
    // If the f32 variant of the attribute isn't specified, try to use the
    // generic one.
    if (Mode.isValid())
      return Mode;
  }

  return getDenormalModeRaw();
}

DenormalMode Function::getDenormalModeRaw() const {
  Attribute Attr = getFnAttribute("denormal-fp-math");
  StringRef Val = Attr.getValueAsString();
  return parseDenormalFPAttribute(Val);
}

DenormalMode Function::getDenormalModeF32Raw() const {
  Attribute Attr = getFnAttribute("denormal-fp-math-f32");
  if (Attr.isValid()) {
    StringRef Val = Attr.getValueAsString();
    return parseDenormalFPAttribute(Val);
  }

  return DenormalMode::getInvalid();
}

const std::string &Function::getGC() const {
  assert(hasGC() && "Function has no collector");
  return getContext().getGC(*this);
}

void Function::setGC(std::string Str) {
  setValueSubclassDataBit(14, !Str.empty());
  getContext().setGC(*this, std::move(Str));
}

void Function::clearGC() {
  if (!hasGC())
    return;
  getContext().deleteGC(*this);
  setValueSubclassDataBit(14, false);
}

bool Function::hasStackProtectorFnAttr() const {
  return hasFnAttribute(Attribute::StackProtect) ||
         hasFnAttribute(Attribute::StackProtectStrong) ||
         hasFnAttribute(Attribute::StackProtectReq);
}

/// Copy all additional attributes (those not needed to create a Function) from
/// the Function Src to this one.
void Function::copyAttributesFrom(const Function *Src) {
  GlobalObject::copyAttributesFrom(Src);
  setCallingConv(Src->getCallingConv());
  setAttributes(Src->getAttributes());
  if (Src->hasGC())
    setGC(Src->getGC());
  else
    clearGC();
  if (Src->hasPersonalityFn())
    setPersonalityFn(Src->getPersonalityFn());
  if (Src->hasPrefixData())
    setPrefixData(Src->getPrefixData());
  if (Src->hasPrologueData())
    setPrologueData(Src->getPrologueData());
}

MemoryEffects Function::getMemoryEffects() const {
  return getAttributes().getMemoryEffects();
}
void Function::setMemoryEffects(MemoryEffects ME) {
  addFnAttr(Attribute::getWithMemoryEffects(getContext(), ME));
}

/// Determine if the function does not access memory.
bool Function::doesNotAccessMemory() const {
  return getMemoryEffects().doesNotAccessMemory();
}
void Function::setDoesNotAccessMemory() {
  setMemoryEffects(MemoryEffects::none());
}

/// Determine if the function does not access or only reads memory.
bool Function::onlyReadsMemory() const {
  return getMemoryEffects().onlyReadsMemory();
}
void Function::setOnlyReadsMemory() {
  setMemoryEffects(getMemoryEffects() & MemoryEffects::readOnly());
}

/// Determine if the function does not access or only writes memory.
bool Function::onlyWritesMemory() const {
  return getMemoryEffects().onlyWritesMemory();
}
void Function::setOnlyWritesMemory() {
  setMemoryEffects(getMemoryEffects() & MemoryEffects::writeOnly());
}

/// Determine if the call can access memmory only using pointers based
/// on its arguments.
bool Function::onlyAccessesArgMemory() const {
  return getMemoryEffects().onlyAccessesArgPointees();
}
void Function::setOnlyAccessesArgMemory() {
  setMemoryEffects(getMemoryEffects() & MemoryEffects::argMemOnly());
}

/// Determine if the function may only access memory that is
///  inaccessible from the IR.
bool Function::onlyAccessesInaccessibleMemory() const {
  return getMemoryEffects().onlyAccessesInaccessibleMem();
}
void Function::setOnlyAccessesInaccessibleMemory() {
  setMemoryEffects(getMemoryEffects() & MemoryEffects::inaccessibleMemOnly());
}

/// Determine if the function may only access memory that is
///  either inaccessible from the IR or pointed to by its arguments.
bool Function::onlyAccessesInaccessibleMemOrArgMem() const {
  return getMemoryEffects().onlyAccessesInaccessibleOrArgMem();
}
void Function::setOnlyAccessesInaccessibleMemOrArgMem() {
  setMemoryEffects(getMemoryEffects() &
                   MemoryEffects::inaccessibleOrArgMemOnly());
}

bool Function::isTargetIntrinsic() const {
  return Intrinsic::isTargetIntrinsic(IntID);
}

void Function::updateAfterNameChange() {
  LibFuncCache = UnknownLibFunc;
  StringRef Name = getName();
  if (!Name.starts_with("llvm.")) {
    HasLLVMReservedName = false;
    IntID = Intrinsic::not_intrinsic;
    return;
  }
  HasLLVMReservedName = true;
  IntID = Intrinsic::lookupIntrinsicID(Name);
}

/// hasAddressTaken - returns true if there are any uses of this function
/// other than direct calls or invokes to it. Optionally ignores callback
/// uses, assume like pointer annotation calls, and references in llvm.used
/// and llvm.compiler.used variables.
bool Function::hasAddressTaken(const User **PutOffender,
                               bool IgnoreCallbackUses,
                               bool IgnoreAssumeLikeCalls, bool IgnoreLLVMUsed,
                               bool IgnoreARCAttachedCall,
                               bool IgnoreCastedDirectCall) const {
  for (const Use &U : uses()) {
    const User *FU = U.getUser();
    if (isa<BlockAddress>(FU))
      continue;

    if (IgnoreCallbackUses) {
      AbstractCallSite ACS(&U);
      if (ACS && ACS.isCallbackCall())
        continue;
    }

    const auto *Call = dyn_cast<CallBase>(FU);
    if (!Call) {
      if (IgnoreAssumeLikeCalls &&
          isa<BitCastOperator, AddrSpaceCastOperator>(FU) &&
          all_of(FU->users(), [](const User *U) {
            if (const auto *I = dyn_cast<IntrinsicInst>(U))
              return I->isAssumeLikeIntrinsic();
            return false;
          })) {
        continue;
      }

      if (IgnoreLLVMUsed && !FU->user_empty()) {
        const User *FUU = FU;
        if (isa<BitCastOperator, AddrSpaceCastOperator>(FU) &&
            FU->hasOneUse() && !FU->user_begin()->user_empty())
          FUU = *FU->user_begin();
        if (llvm::all_of(FUU->users(), [](const User *U) {
              if (const auto *GV = dyn_cast<GlobalVariable>(U))
                return GV->hasName() &&
                       (GV->getName() == "llvm.compiler.used" ||
                        GV->getName() == "llvm.used");
              return false;
            }))
          continue;
      }
      if (PutOffender)
        *PutOffender = FU;
      return true;
    }

    if (IgnoreAssumeLikeCalls) {
      if (const auto *I = dyn_cast<IntrinsicInst>(Call))
        if (I->isAssumeLikeIntrinsic())
          continue;
    }

    if (!Call->isCallee(&U) || (!IgnoreCastedDirectCall &&
                                Call->getFunctionType() != getFunctionType())) {
      if (IgnoreARCAttachedCall &&
          Call->isOperandBundleOfType(LLVMContext::OB_clang_arc_attachedcall,
                                      U.getOperandNo()))
        continue;

      if (PutOffender)
        *PutOffender = FU;
      return true;
    }
  }
  return false;
}

bool Function::isDefTriviallyDead() const {
  // Check the linkage
  if (!hasLinkOnceLinkage() && !hasLocalLinkage() &&
      !hasAvailableExternallyLinkage())
    return false;

  // Check if the function is used by anything other than a blockaddress.
  for (const User *U : users())
    if (!isa<BlockAddress>(U))
      return false;

  return true;
}

/// callsFunctionThatReturnsTwice - Return true if the function has a call to
/// setjmp or other function that gcc recognizes as "returning twice".
bool Function::callsFunctionThatReturnsTwice() const {
  for (const Instruction &I : instructions(this))
    if (const auto *Call = dyn_cast<CallBase>(&I))
      if (Call->hasFnAttr(Attribute::ReturnsTwice))
        return true;

  return false;
}

Constant *Function::getPersonalityFn() const {
  assert(hasPersonalityFn() && getNumOperands());
  return cast<Constant>(Op<0>());
}

void Function::setPersonalityFn(Constant *Fn) {
  setHungoffOperand<0>(Fn);
  setValueSubclassDataBit(3, Fn != nullptr);
}

Constant *Function::getPrefixData() const {
  assert(hasPrefixData() && getNumOperands());
  return cast<Constant>(Op<1>());
}

void Function::setPrefixData(Constant *PrefixData) {
  setHungoffOperand<1>(PrefixData);
  setValueSubclassDataBit(1, PrefixData != nullptr);
}

Constant *Function::getPrologueData() const {
  assert(hasPrologueData() && getNumOperands());
  return cast<Constant>(Op<2>());
}

void Function::setPrologueData(Constant *PrologueData) {
  setHungoffOperand<2>(PrologueData);
  setValueSubclassDataBit(2, PrologueData != nullptr);
}

void Function::allocHungoffUselist() {
  // If we've already allocated a uselist, stop here.
  if (getNumOperands())
    return;

  allocHungoffUses(3, /*IsPhi=*/ false);
  setNumHungOffUseOperands(3);

  // Initialize the uselist with placeholder operands to allow traversal.
  auto *CPN = ConstantPointerNull::get(PointerType::get(getContext(), 0));
  Op<0>().set(CPN);
  Op<1>().set(CPN);
  Op<2>().set(CPN);
}

template <int Idx>
void Function::setHungoffOperand(Constant *C) {
  if (C) {
    allocHungoffUselist();
    Op<Idx>().set(C);
  } else if (getNumOperands()) {
    Op<Idx>().set(ConstantPointerNull::get(PointerType::get(getContext(), 0)));
  }
}

void Function::setValueSubclassDataBit(unsigned Bit, bool On) {
  assert(Bit < 16 && "SubclassData contains only 16 bits");
  if (On)
    setValueSubclassData(getSubclassDataFromValue() | (1 << Bit));
  else
    setValueSubclassData(getSubclassDataFromValue() & ~(1 << Bit));
}

void Function::setEntryCount(ProfileCount Count,
                             const DenseSet<GlobalValue::GUID> *S) {
#if !defined(NDEBUG)
  auto PrevCount = getEntryCount();
  assert(!PrevCount || PrevCount->getType() == Count.getType());
#endif

  auto ImportGUIDs = getImportGUIDs();
  if (S == nullptr && ImportGUIDs.size())
    S = &ImportGUIDs;

  MDBuilder MDB(getContext());
  setMetadata(
      LLVMContext::MD_prof,
      MDB.createFunctionEntryCount(Count.getCount(), Count.isSynthetic(), S));
}

void Function::setEntryCount(uint64_t Count, Function::ProfileCountType Type,
                             const DenseSet<GlobalValue::GUID> *Imports) {
  setEntryCount(ProfileCount(Count, Type), Imports);
}

std::optional<ProfileCount> Function::getEntryCount(bool AllowSynthetic) const {
  MDNode *MD = getMetadata(LLVMContext::MD_prof);
  if (MD && MD->getOperand(0))
    if (MDString *MDS = dyn_cast<MDString>(MD->getOperand(0))) {
      if (MDS->getString() == "function_entry_count") {
        ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(1));
        uint64_t Count = CI->getValue().getZExtValue();
        // A value of -1 is used for SamplePGO when there were no samples.
        // Treat this the same as unknown.
        if (Count == (uint64_t)-1)
          return std::nullopt;
        return ProfileCount(Count, PCT_Real);
      } else if (AllowSynthetic &&
                 MDS->getString() == "synthetic_function_entry_count") {
        ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(1));
        uint64_t Count = CI->getValue().getZExtValue();
        return ProfileCount(Count, PCT_Synthetic);
      }
    }
  return std::nullopt;
}

DenseSet<GlobalValue::GUID> Function::getImportGUIDs() const {
  DenseSet<GlobalValue::GUID> R;
  if (MDNode *MD = getMetadata(LLVMContext::MD_prof))
    if (MDString *MDS = dyn_cast<MDString>(MD->getOperand(0)))
      if (MDS->getString() == "function_entry_count")
        for (unsigned i = 2; i < MD->getNumOperands(); i++)
          R.insert(mdconst::extract<ConstantInt>(MD->getOperand(i))
                       ->getValue()
                       .getZExtValue());
  return R;
}

void Function::setSectionPrefix(StringRef Prefix) {
  MDBuilder MDB(getContext());
  setMetadata(LLVMContext::MD_section_prefix,
              MDB.createFunctionSectionPrefix(Prefix));
}

std::optional<StringRef> Function::getSectionPrefix() const {
  if (MDNode *MD = getMetadata(LLVMContext::MD_section_prefix)) {
    assert(cast<MDString>(MD->getOperand(0))->getString() ==
               "function_section_prefix" &&
           "Metadata not match");
    return cast<MDString>(MD->getOperand(1))->getString();
  }
  return std::nullopt;
}

bool Function::nullPointerIsDefined() const {
  return hasFnAttribute(Attribute::NullPointerIsValid);
}

bool llvm::NullPointerIsDefined(const Function *F, unsigned AS) {
  if (F && F->nullPointerIsDefined())
    return true;

  if (AS != 0)
    return true;

  return false;
}
