//===- DebugInfoExprs.h - Debug Info Expression Manipulation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements DebugInfoExprs.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoExprs.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <optional>

using namespace llvm;

namespace llvm {

namespace DIOp {

#define HANDLE_OP0(NAME, ENCODING)                                             \
  bool NAME::operator==(const NAME &O) const { return true; }                  \
  void NAME::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {                 \
    Out.append({ENCODING});                                                    \
  }
#define HANDLE_OP1(NAME, ENCODING, TYPE1, NAME1)                               \
  bool NAME::operator==(const NAME &O) const { return NAME1 == O.NAME1; }      \
  void NAME::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {                 \
    Out.append({ENCODING, static_cast<uint64_t>(NAME1)});                      \
  }                                                                            \
  TYPE1 NAME::get##NAME1() const { return NAME1; }                             \
  void NAME::set##NAME1(TYPE1 NAME1) { this->NAME1 = NAME1; }
#define HANDLE_OP2(NAME, ENCODING, TYPE1, NAME1, TYPE2, NAME2)                 \
  bool NAME::operator==(const NAME &O) const {                                 \
    return NAME1 == O.NAME1 && NAME2 == O.NAME2;                               \
  }                                                                            \
  void NAME::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {                 \
    Out.append({ENCODING, static_cast<uint64_t>(NAME1),                        \
                static_cast<uint64_t>(NAME2)});                                \
  }                                                                            \
  TYPE1 NAME::get##NAME1() const { return NAME1; }                             \
  void NAME::set##NAME1(TYPE1 NAME1) { this->NAME1 = NAME1; }                  \
  TYPE2 NAME::get##NAME2() const { return NAME2; }                             \
  void NAME::set##NAME2(TYPE2 NAME2) { this->NAME2 = NAME2; }
#include "llvm/IR/DIOps.def"

#define HANDLE_OP(NAME, ENCODING)                                              \
  StringRef NAME::getAsmName() { return "DIOp" #NAME; }                        \
  uint64_t NAME::getDwarfEncoding() { return ENCODING; }
#include "llvm/IR/DIOps.def"

#define HANDLE_OP(NAME, ENCODING)                                              \
  template <> bool Op::holds<DIOp::NAME>() const {                             \
    return CIS.Tag == impl::TagT::NAME;                                        \
  }                                                                            \
  template <> DIOp::NAME Op::get<DIOp::NAME>() const {                         \
    assert(holds<DIOp::NAME>());                                               \
    return NAME;                                                               \
  }                                                                            \
  template <> std::optional<DIOp::NAME> Op::getIf<DIOp::NAME>() const {        \
    return (holds<DIOp::NAME>()) ? std::optional<DIOp::NAME>(NAME)             \
                                 : std::nullopt;                               \
  }
HANDLE_OP(LLVMEscape, void)
#include "llvm/IR/DIOps.def"

// The standard_layout is required to make use of the "Common Initial Sequence"
// rule, so that we can pack the discriminator "tag" into each alternative type.
//
// The trivially_destructible is a necessary (although perhaps not quite
// sufficient?) condition for the way we use a union, and it also enables
// cheaper versions of e.g. SmallVector methods.
//
// Finally the size check is a sanity check to make sure we don't accidentally
// balloon the struct. We are relying on the no-more-than-two-qword size to
// sneak into purely register passing on e.g. Itanium x86_64, and any increase
// also has a multiplicative effect on the size of buffers of Ops.
#define ASSERT_REQUIREMENTS(NAME)                                              \
  static_assert(std::is_standard_layout<NAME>::value);                         \
  static_assert(std::is_trivially_destructible<NAME>::value);                  \
  static_assert(sizeof(NAME) <= 16);
#define HANDLE_OP(NAME, ENCODING) ASSERT_REQUIREMENTS(NAME)
HANDLE_OP(LLVMEscape, void)
#include "llvm/IR/DIOps.def"
ASSERT_REQUIREMENTS(Op)
#undef ASSERT_REQUIREMENTS

} // end namespace DIOp

uint32_t DIOp::FromUIntIterator::getRemainingSize() const {
  assert(End > I);
  assert(End - I < std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(End - I);
}

uint32_t DIOp::FromUIntIterator::getCurrentOpSize() const {
  const uint32_t RemainingSize = getRemainingSize();
  uint32_t Ret;
  switch (*I) {
#define HANDLE_OP0(NAME, ENCODING)                                             \
  case ENCODING:                                                               \
    Ret = 1u;                                                                  \
    break;
#define HANDLE_OP1(NAME, ENCODING, ...)                                        \
  case ENCODING:                                                               \
    Ret = 2u;                                                                  \
    break;
#define HANDLE_OP2(NAME, ENCODING, ...)                                        \
  case ENCODING:                                                               \
    Ret = 3u;                                                                  \
    break;
#include "llvm/IR/DIOps.def"
  default:
    Ret = RemainingSize;
  }
  if (Ret > RemainingSize)
    return RemainingSize;
  return Ret;
}

DIOp::Op DIOp::FromUIntIterator::operator*() const {
  const uint32_t RemainingSize = getRemainingSize();
  switch (*I) {
#define HANDLE_OP0(NAME, ENCODING)                                             \
  case ENCODING:                                                               \
    return DIOp::NAME();
#define HANDLE_OP1(NAME, ENCODING, ...)                                        \
  case ENCODING:                                                               \
    if (RemainingSize < 2)                                                     \
      break;                                                                   \
    return DIOp::NAME(I[1]);
#define HANDLE_OP2(NAME, ENCODING, ...)                                        \
  case ENCODING:                                                               \
    if (RemainingSize < 3)                                                     \
      break;                                                                   \
    return DIOp::NAME(I[1], I[2]);
#include "llvm/IR/DIOps.def"
  }
  return LLVMEscape(I, RemainingSize);
}

void DIOp::LLVMEscape::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {
  auto Data = getData();
  Out.append(Data.begin(), Data.end());
}

void DIOp::Op::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {
  visitOverload([&](auto Op) { Op.toUIntVec(Out); });
}

DIExprRef::DIExprRef(const DIExpression *From)
    : Ops(DIOp::FromUIntIterator::makeRange(From->getElements())) {}

DIExprRef::ExtOps DIExprRef::getExtOps(unsigned FromSize, unsigned ToSize,
                                       bool Signed) {
  dwarf::TypeKind TK = Signed ? dwarf::DW_ATE_signed : dwarf::DW_ATE_unsigned;
  return {DIOp::LLVMConvert(FromSize, TK), DIOp::LLVMConvert(ToSize, TK)};
}

std::optional<DbgVariableFragmentInfo> DIExprRef::getFragmentInfo() const {
  auto I =
      find_if(Ops, [](DIOp::Op Op) { return Op.holds<DIOp::LLVMFragment>(); });
  if (I == Ops.end())
    return std::nullopt;
  assert(std::next(I) == Ops.end());
  auto Fragment = I->get<DIOp::LLVMFragment>();
  return DbgVariableFragmentInfo{Fragment.getSizeInBits(),
                                 Fragment.getOffsetInBits()};
}

bool DIExprRef::isValid() const {
  for (auto I = Ops.begin(), E = Ops.end(); I != E; ++I) {
    auto IsValid = I->visitOverload(
        [=](DIOp::LLVMFragment Fragment) { return std::next(I) == Ops.end(); },
        [=](DIOp::StackValue StackValue) {
          // Must be the last one or followed by a
          // DW_OP_LLVM_fragment.
          if (std::next(I) == Ops.end())
            return true;
          if (std::next(I)->holds<DIOp::LLVMFragment>())
            return true;
          return false;
        },
        [=](DIOp::Swap Swap) { return std::next(Ops.begin()) != Ops.end(); },
        [=](DIOp::LLVMEntryValue EntryValue) {
          // An entry value operator must appear at the
          // beginning or immediately following `DW_OP_LLVM_arg
          // 0`, and the number of operations it cover can
          // currently only be 1, because we support only entry
          // values of a simple register location. One reason
          // for this is that we currently can't calculate the
          // size of the resulting DWARF block for other
          // expressions.
          auto J = Ops.begin();
          if (auto Arg = J->getIf<DIOp::LLVMArg>())
            if (Arg->getIndex() == 0)
              ++J;
          return I == J && EntryValue.getOps() == 1;
        },
        [=](DIOp::LLVMEscape Op) { return false; },
        [=](auto Op) { return true; });
    if (!IsValid)
      return false;
  }
  return true;
}

bool DIExprRef::isSingleLocationExpression() const {
  if (!isValid())
    return false;

  if (Ops.empty())
    return true;

  auto I = Ops.begin();
  if (auto Arg = I->getIf<DIOp::LLVMArg>()) {
    if (Arg->getIndex() != 0u)
      return false;
    ++I;
  }

  return !std::any_of(I, Ops.end(),
                      [](DIOp::Op Op) { return Op.holds<DIOp::LLVMArg>(); });
}

std::optional<DIExprRef> DIExprRef::getSingleLocationExprRef() const {
  // Check for `isValid` covered by `isSingleLocationExpression`.
  if (!isSingleLocationExpression())
    return std::nullopt;

  if (Ops.empty())
    return *this;

  if (Ops.begin()->holds<DIOp::LLVMArg>())
    return DIExprRef{std::next(Ops.begin()), Ops.end()};
  return *this;
}

bool DIExprRef::startsWithDeref() const {
  if (auto SingleLocRef = getSingleLocationExprRef())
    return !SingleLocRef->Ops.empty() &&
           SingleLocRef->Ops.begin()->holds<DIOp::Deref>();
  return false;
}

bool DIExprRef::isDeref() const {
  if (auto SingleLocRef = getSingleLocationExprRef())
    return !SingleLocRef->Ops.empty() &&
           std::next(SingleLocRef->Ops.begin()) == SingleLocRef->Ops.end() &&
           SingleLocRef->Ops.begin()->holds<DIOp::Deref>();
  return false;
}

bool DIExprRef::isImplicit() const {
  return isValid() && find_if(Ops, [](DIOp::Op Op) {
                        return Op.holds<DIOp::StackValue>();
                      }) != Ops.end();
}

bool DIExprRef::isComplex() const {
  // If there are any elements other than fragment or tag_offset, then some
  // kind of complex computation occurs.
  return isValid() && find_if_not(Ops, [](DIOp::Op Op) {
                        return Op.holds<DIOp::LLVMTagOffset>() ||
                               Op.holds<DIOp::LLVMFragment>() ||
                               Op.holds<DIOp::LLVMArg>();
                      }) != Ops.end();
}

bool DIExprRef::isEntryValue() const {
  if (auto SingleLocRef = getSingleLocationExprRef())
    return !SingleLocRef->Ops.empty() &&
           SingleLocRef->Ops.begin()->holds<DIOp::LLVMEntryValue>();
  return false;
}

std::optional<SignedOrUnsignedConstant> DIExprRef::isConstant() const {
  // Recognize signed and unsigned constants.
  // An signed constants can be represented as:
  // DIOp::ConstS(N) (DIOp::StackValue (DIOp::Fragment(O, S)))
  // An unsigned constant can be represented as:
  // DIOp::ConstU(N) (DIOp::StackValue (DIOp::Fragment(O, S)))

  auto I = Ops.begin();
  auto Op0 = maybeAdvance(I);
  auto Op1 = maybeAdvance(I);
  auto Op2 = maybeAdvance(I);

  if (I != Ops.end())
    return std::nullopt;

  if (!Op0 || (!Op0->holds<DIOp::ConstU>() && !Op0->holds<DIOp::ConstS>()))
    return std::nullopt;

  if (Op1 && !Op1->holds<DIOp::StackValue>())
    return std::nullopt;

  if (Op2 && !Op2->holds<DIOp::LLVMFragment>())
    return std::nullopt;

  return Op0->holds<DIOp::ConstU>() ? SignedOrUnsignedConstant::UnsignedConstant
                                    : SignedOrUnsignedConstant::SignedConstant;
}

uint64_t DIExprRef::getNumLocationOperands() const {
  uint64_t Result = 0;
  for (auto Op : Ops)
    if (auto Arg = Op.getIf<DIOp::LLVMArg>())
      Result = std::max(Result, Arg->getIndex() + 1);
  /*FIXME:
  assert(hasAllLocationOps(Result) &&
         "Expression is missing one or more location operands.");
         */
  return Result;
}

std::optional<uint64_t> DIExprRef::getActiveBits(DIVariable *Var) const {
  std::optional<uint64_t> InitialActiveBits = Var->getSizeInBits();
  std::optional<uint64_t> ActiveBits = InitialActiveBits;
  auto UpdateActiveBits = [&](uint64_t NewActiveBits) {
    // Narrow the active bits
    if (ActiveBits)
      ActiveBits = std::min(*ActiveBits, NewActiveBits);
    else
      ActiveBits = NewActiveBits;
  };
  auto HandleExtOp = [&](DIBasicType::Signedness OpSignedness,
                         uint64_t OpActiveBits) {
    // We can't handle an extract whose sign doesn't match that of the
    // variable.
    std::optional<DIBasicType::Signedness> VarSignedness = Var->getSignedness();
    if (VarSignedness && *VarSignedness == OpSignedness)
      UpdateActiveBits(OpActiveBits);
    else
      ActiveBits = InitialActiveBits;
  };
  for (auto Op : Ops)
    Op.visitOverload(
        [&](DIOp::LLVMExtractBitsZExt ZExt) {
          HandleExtOp(DIBasicType::Signedness::Unsigned, ZExt.getSizeInBits());
        },
        [&](DIOp::LLVMExtractBitsSExt SExt) {
          HandleExtOp(DIBasicType::Signedness::Signed, SExt.getSizeInBits());
        },
        [&](DIOp::LLVMFragment Fragment) {
          UpdateActiveBits(Fragment.getSizeInBits());
        },
        [&](auto Op) {
          // We assume the worst case for anything we don't currently
          // handle and revert to the initial active bits.
          ActiveBits = InitialActiveBits;
        });
  return ActiveBits;
}

std::optional<int64_t> DIExprRef::extractIfOffset() const {
  auto SLExprRefOpt = getSingleLocationExprRef();
  if (!SLExprRefOpt)
    return std::nullopt;
  auto SLExprRef = *SLExprRefOpt;

  auto I = SLExprRef.Ops.begin();
  auto Op0 = maybeAdvance(I);
  auto Op1 = maybeAdvance(I);
  if (I != SLExprRef.Ops.end())
    return std::nullopt;

  if (Op1) {
    if (auto ConstU = Op0->getIf<DIOp::ConstU>()) {
      if (Op1->holds<DIOp::Plus>())
        return ConstU->getValue();
      if (Op1->holds<DIOp::Minus>())
        return -ConstU->getValue();
    }
    return std::nullopt;
  }

  if (Op0) {
    if (auto PlusUConst = Op0->getIf<DIOp::PlusUConst>())
      return PlusUConst->getValue();
    return std::nullopt;
  }

  return 0;
}

std::optional<std::pair<int64_t, DIExprRef>>
DIExprRef::extractLeadingOffset() const {
  int64_t OffsetInBytes = 0u;

  auto SLExprRefOpt = getSingleLocationExprRef();
  if (!SLExprRefOpt)
    return std::nullopt;
  auto SLExprRef = *SLExprRefOpt;

  auto I = SLExprRef.Ops.begin();
  while (I != SLExprRef.Ops.end()) {
    if (I->holdsOneOf<DIOp::Deref, DIOp::DerefSize, DIOp::LLVMFragment,
                      DIOp::LLVMExtractBitsZExt, DIOp::LLVMExtractBitsSExt>())
      break;
    if (auto P = I->getIf<DIOp::PlusUConst>()) {
      OffsetInBytes += P->getValue();
    } else if (auto C = I->getIf<DIOp::ConstU>()) {
      uint64_t Value = C->getValue();
      auto J = maybeAdvance(I);
      if (!J)
        return std::nullopt;
      if (J->holds<DIOp::Plus>())
        OffsetInBytes += Value;
      else if (J->holds<DIOp::Minus>())
        OffsetInBytes -= Value;
      else
        return std::nullopt;
    } else {
      // Not a const plus/minus operation or deref.
      return std::nullopt;
    }
    ++I;
  }
  return std::make_pair(OffsetInBytes, DIExprRef{I, SLExprRef.Ops.end()});
}

bool DIExprRef::hasAllLocationOps(unsigned N) const {
  SmallDenseSet<uint64_t> SeenIndexes;
  for (auto Op : Ops)
    if (auto Arg = Op.getIf<DIOp::LLVMArg>())
      SeenIndexes.insert(Arg->getIndex());
  for (uint64_t Index = 0; Index < N; ++Index)
    if (!SeenIndexes.contains(Index))
      return false;
  return true;
}

void DIExprRef::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {
  for (auto Op : Ops)
    Op.toUIntVec(Out);
}
SmallVector<uint64_t> DIExprRef::toUIntVec() const {
  SmallVector<uint64_t> Ops;
  toUIntVec(Ops);
  return Ops;
}

namespace impl {
class DIExpr {
public:
  template <typename OpsT>
  static void appendRaw(SmallVectorImpl<uint64_t> &Buf, OpsT Ops) {
    for (auto Op : Ops)
      Op.toUIntVec(Buf);
  }
  template <typename OpsT>
  static void assignRaw(SmallVectorImpl<uint64_t> &Buf, OpsT Ops) {
    Buf.clear();
    appendRaw(Buf, Ops);
  }
  template <typename OpsT>
  static DIExprBuf &append(DIExprBuf &DIBuf, OpsT Ops) {
    auto ExistingOps = DIOp::FromUIntIterator::makeRange(DIBuf.Elements);
    auto InsertPoint = find_if(ExistingOps, [](DIOp::Op Op) {
      return Op.holds<DIOp::StackValue>() || Op.holds<DIOp::LLVMFragment>();
    });
    DIBuf.NewElements.append(ExistingOps.begin().I, InsertPoint.I);
    appendRaw(DIBuf.NewElements, Ops);
    DIBuf.NewElements.append(InsertPoint.I, ExistingOps.end().I);
    return DIBuf.swap();
  }
  template <typename OpsT>
  static DIExprBuf &prependOpcodes(DIExprBuf &DIBuf, OpsT Ops, bool StackValue,
                                   bool EntryValue) {
    appendRaw(DIBuf.NewElements, Ops);
    return DIBuf.prependOpcodesFinalize(StackValue, EntryValue);
  }
  template <typename OpsT>
  static DIExprBuf &appendOpsToArg(DIExprBuf &DIBuf, OpsT Ops,
                                   unsigned ArgIndex, bool StackValue = false) {
    // Handle non-variadic intrinsics by prepending the opcodes.
    if (!any_of(expr_ops(DIBuf.Elements),
                [](auto Op) { return Op.getOp() == dwarf::DW_OP_LLVM_arg; })) {
      assert(ArgIndex == 0 &&
             "Location Index must be 0 for a non-variadic expression.");
      appendRaw(DIBuf.NewElements, Ops);
      return DIBuf.prependOpcodesFinalize(StackValue, false);
    }

    for (auto Op : expr_ops(DIBuf.Elements)) {
      // A DW_OP_stack_value comes at the end, but before a DW_OP_LLVM_fragment.
      if (StackValue) {
        if (Op.getOp() == dwarf::DW_OP_stack_value)
          StackValue = false;
        else if (Op.getOp() == dwarf::DW_OP_LLVM_fragment) {
          DIBuf.NewElements.push_back(dwarf::DW_OP_stack_value);
          StackValue = false;
        }
      }
      Op.appendToVector(DIBuf.NewElements);
      if (Op.getOp() == dwarf::DW_OP_LLVM_arg && Op.getArg(0) == ArgIndex)
        for (auto Op : Ops)
          Op.toUIntVec(DIBuf.NewElements);
    }
    if (StackValue)
      DIBuf.NewElements.push_back(dwarf::DW_OP_stack_value);

    return DIBuf.swap();
  }
};
template <>
void DIExpr::appendRaw<DIExprRef>(SmallVectorImpl<uint64_t> &Buf,
                                  DIExprRef Ops) {
  Buf.append(Ops.Ops.begin().I, Ops.Ops.end().I);
}
template <>
void DIExpr::appendRaw<const DIExpression *>(SmallVectorImpl<uint64_t> &Buf,
                                             const DIExpression *Ops) {
  Buf.append(Ops->elements_begin(), Ops->elements_end());
}
} // end namespace impl
} // end namespace llvm

DIExprBuf &DIExprBuf::prependOpcodesFinalize(bool StackValue, bool EntryValue) {
  if (EntryValue) {
    NewElements.push_back(dwarf::DW_OP_LLVM_entry_value);
    // Use a block size of 1 for the target register operand.  The
    // DWARF backend currently cannot emit entry values with a block
    // size > 1.
    NewElements.push_back(1);
  }
  // If there are no ops to prepend, do not even add the DW_OP_stack_value.
  if (NewElements.empty())
    StackValue = false;
  for (auto Op : expr_ops(Elements)) {
    // A DW_OP_stack_value comes at the end, but before a DW_OP_LLVM_fragment.
    if (StackValue) {
      if (Op.getOp() == dwarf::DW_OP_stack_value)
        StackValue = false;
      else if (Op.getOp() == dwarf::DW_OP_LLVM_fragment) {
        NewElements.push_back(dwarf::DW_OP_stack_value);
        StackValue = false;
      }
    }
    Op.appendToVector(NewElements);
  }
  if (StackValue)
    NewElements.push_back(dwarf::DW_OP_stack_value);
  return swap();
}

DIExprBuf::DIExprBuf(LLVMContext *Ctx) : Ctx(Ctx) {}
DIExprBuf::DIExprBuf(const DIExpression *From)
    : Ctx(&From->getContext()), Elements(From->getElements()) {}

DIExprBuf &DIExprBuf::assign(const DIExpression *From) {
  Ctx = &From->getContext();
  impl::DIExpr::assignRaw(Elements, From);
  return *this;
}

DIExprBuf &DIExprBuf::assign(LLVMContext *Ctx, DIExprRef From) {
  this->Ctx = Ctx;
  impl::DIExpr::assignRaw(Elements, From);
  return *this;
}

/*
DIExprBuf DIExprBuf::canonicalize(LLVMContext *Ctx, DIExprRef From,
                              bool IsIndirect) {
bool NeedsDeref = IsIndirect;
DIExprBuf Result(Ctx);

if (!any_of(From.Ops, [](DIOp::Op Op) { return Op.holds<DIOp::LLVMArg>(); }))
Result.Ops.emplace_back(DIOp::LLVMArg(0u));
for (auto I = From.Ops.begin(), E = From.Ops.end(); I != E; ++I) {
if (NeedsDeref &&
    (I->holds<DIOp::StackValue>() || I->holds<DIOp::LLVMFragment>())) {
  Result.Ops.emplace_back(DIOp::Deref());
  NeedsDeref = false;
}
Result.Ops.emplace_back(*I);
}
if (NeedsDeref)
Result.Ops.emplace_back(DIOp::Deref());
return Result;
}

DIExprBuf DIExprBuf::canonicalize(const DIExpression *From, bool IsIndirect) {
return DIExprBuf::canonicalize(&From->getContext(), From->asRef(),
                             IsIndirect);
}
*/

DIExprBuf &DIExprBuf::appendRaw(std::initializer_list<DIOp::Op> NewOps) {
  impl::DIExpr::appendRaw(Elements, NewOps);
  return *this;
}
DIExprBuf &DIExprBuf::appendRaw(iterator_range<const DIOp::Op *> NewOps) {
  impl::DIExpr::appendRaw(Elements, NewOps);
  return *this;
}

DIExprBuf &DIExprBuf::clear() {
  Elements.clear();
  return *this;
}

DIExprBuf &DIExprBuf::convertToUndefExpression() {
  if (auto FragmentInfo = asRef().getFragmentInfo())
    NewElements.append({dwarf::DW_OP_LLVM_fragment, FragmentInfo->OffsetInBits,
                        FragmentInfo->SizeInBits});
  return swap();
}

DIExprBuf &DIExprBuf::convertToVariadicExpressionUnchecked() {
  NewElements.reserve(Elements.size() + 2);
  NewElements.append({dwarf::DW_OP_LLVM_arg, 0});
  NewElements.append(Elements);
  return swap();
}

DIExprBuf &DIExprBuf::convertToVariadicExpression() {
  if (any_of(expr_ops(Elements), [](auto ExprOp) {
        return ExprOp.getOp() == dwarf::DW_OP_LLVM_arg;
      }))
    return *this;
  return convertToVariadicExpressionUnchecked();
}

bool DIExprBuf::convertToNonVariadicExpression() {
  auto SingleLocRefOpt = asRef().getSingleLocationExprRef();
  if (!SingleLocRefOpt)
    return false;
  SingleLocRefOpt->toUIntVec(NewElements);
  swap();
  return true;
}

DIExprBuf &DIExprBuf::prepend(uint8_t Flags, int64_t Offset) {
  if (Flags & DIExpression::DerefBefore)
    NewElements.push_back(dwarf::DW_OP_deref);
  appendOffsetImpl(NewElements, Offset);
  if (Flags & DIExpression::DerefAfter)
    NewElements.push_back(dwarf::DW_OP_deref);
  bool StackValue = Flags & DIExpression::StackValue;
  bool EntryValue = Flags & DIExpression::EntryValue;
  return prependOpcodesFinalize(StackValue, EntryValue);
}

DIExprBuf &DIExprBuf::append(iterator_range<const DIOp::Op *> NewOps) {
  return impl::DIExpr::append(*this, NewOps);
}
DIExprBuf &DIExprBuf::append(std::initializer_list<DIOp::Op> NewOps) {
  return impl::DIExpr::append(*this, NewOps);
}
DIExprBuf &DIExprBuf::append(DIExprRef NewOps) {
  return impl::DIExpr::append(*this, NewOps);
}
DIExprBuf &DIExprBuf::append(iterator_range<DIOp::FromUIntIterator> NewOps) {
  return impl::DIExpr::append(*this, NewOps);
}

DIExprBuf &DIExprBuf::prependOpcodes(iterator_range<const DIOp::Op *> NewOps,
                                     bool StackValue, bool EntryValue) {
  return impl::DIExpr::prependOpcodes(*this, NewOps, StackValue, EntryValue);
}
DIExprBuf &DIExprBuf::prependOpcodes(std::initializer_list<DIOp::Op> NewOps,
                                     bool StackValue, bool EntryValue) {
  return impl::DIExpr::prependOpcodes(*this, NewOps, StackValue, EntryValue);
}
DIExprBuf &DIExprBuf::prependOpcodes(DIExprRef NewOps, bool StackValue,
                                     bool EntryValue) {
  return impl::DIExpr::prependOpcodes(*this, NewOps, StackValue, EntryValue);
}
DIExprBuf &
DIExprBuf::prependOpcodes(iterator_range<DIOp::FromUIntIterator> NewOps,
                          bool StackValue, bool EntryValue) {
  return impl::DIExpr::prependOpcodes(*this, NewOps, StackValue, EntryValue);
}

DIExprBuf &DIExprBuf::appendOpsToArg(iterator_range<const DIOp::Op *> NewOps,
                                     unsigned ArgIndex, bool StackValue) {
  return impl::DIExpr::appendOpsToArg(*this, NewOps, ArgIndex, StackValue);
}
DIExprBuf &DIExprBuf::appendOpsToArg(std::initializer_list<DIOp::Op> NewOps,
                                     unsigned ArgIndex, bool StackValue) {
  return impl::DIExpr::appendOpsToArg(*this, NewOps, ArgIndex, StackValue);
}
DIExprBuf &DIExprBuf::appendOpsToArg(DIExprRef NewOps, unsigned ArgIndex,
                                     bool StackValue) {
  return impl::DIExpr::appendOpsToArg(*this, NewOps.Ops, ArgIndex, StackValue);
}
DIExprBuf &
DIExprBuf::appendOpsToArg(iterator_range<DIOp::FromUIntIterator> NewOps,
                          unsigned ArgIndex, bool StackValue) {
  return impl::DIExpr::appendOpsToArg(*this, NewOps, ArgIndex, StackValue);
}

DIExprBuf &DIExprBuf::appendConstant(SignedOrUnsignedConstant SignedOrUnsigned,
                                     uint64_t Val) {
  uint64_t Opcode = SignedOrUnsigned == SignedOrUnsignedConstant::SignedConstant
                        ? dwarf::DW_OP_consts
                        : dwarf::DW_OP_constu;
  Elements.append({Opcode, Val});
  return *this;
}

DIExprBuf &DIExprBuf::appendOffset(int64_t Offset) {
  appendOffsetImpl(Elements, Offset);
  return *this;
}

DIExprBuf &DIExprBuf::replaceArg(uint64_t OldArgIndex, uint64_t NewArgIndex) {
  for (auto Op : expr_ops(Elements)) {
    if (Op.getOp() != dwarf::DW_OP_LLVM_arg || Op.getArg(0) < OldArgIndex) {
      Op.appendToVector(NewElements);
      continue;
    }
    NewElements.push_back(dwarf::DW_OP_LLVM_arg);
    uint64_t ArgIndex =
        Op.getArg(0) == OldArgIndex ? NewArgIndex : Op.getArg(0);
    // OldArg has been deleted from the Op list, so decrement all indices
    // greater than it.
    if (ArgIndex > OldArgIndex)
      --ArgIndex;
    NewElements.push_back(ArgIndex);
  }
  return swap();
}

bool DIExprBuf::createFragmentExpression(unsigned OffsetInBits,
                                         unsigned SizeInBits) {
  // Track whether it's safe to split the value at the top of the DWARF stack,
  // assuming that it'll be used as an implicit location value.
  bool CanSplitValue = true;
  // Track whether we need to add a fragment expression to the end of Expr.
  bool EmitFragment = true;
  // Copy over the expression, but leave off any trailing DW_OP_LLVM_fragment.
  for (auto Op : expr_ops(Elements)) {
    switch (Op.getOp()) {
    default:
      break;
    case dwarf::DW_OP_shr:
    case dwarf::DW_OP_shra:
    case dwarf::DW_OP_shl:
    case dwarf::DW_OP_plus:
    case dwarf::DW_OP_plus_uconst:
    case dwarf::DW_OP_minus:
      // We can't safely split arithmetic or shift operations into multiple
      // fragments because we can't express carry-over between fragments.
      //
      // FIXME: We *could* preserve the lowest fragment of a constant offset
      // operation if the offset fits into SizeInBits.
      CanSplitValue = false;
      break;
    case dwarf::DW_OP_deref:
    case dwarf::DW_OP_deref_size:
    case dwarf::DW_OP_deref_type:
    case dwarf::DW_OP_xderef:
    case dwarf::DW_OP_xderef_size:
    case dwarf::DW_OP_xderef_type:
      // Preceeding arithmetic operations have been applied to compute an
      // address. It's okay to split the value loaded from that address.
      CanSplitValue = true;
      break;
    case dwarf::DW_OP_stack_value:
      // Bail if this expression computes a value that cannot be split.
      if (!CanSplitValue)
        return drop();
      break;
    case dwarf::DW_OP_LLVM_fragment: {
      // If we've decided we don't need a fragment then give up if we see that
      // there's already a fragment expression.
      // FIXME: We could probably do better here
      if (!EmitFragment)
        return drop();
      // Make the new offset point into the existing fragment.
      uint64_t FragmentOffsetInBits = Op.getArg(0);
      uint64_t FragmentSizeInBits = Op.getArg(1);
      (void)FragmentSizeInBits;
      assert((OffsetInBits + SizeInBits <= FragmentSizeInBits) &&
             "new fragment outside of original fragment");
      OffsetInBits += FragmentOffsetInBits;
      continue;
    }
    case dwarf::DW_OP_LLVM_extract_bits_zext:
    case dwarf::DW_OP_LLVM_extract_bits_sext: {
      // If we're extracting bits from inside of the fragment that we're
      // creating then we don't have a fragment after all, and just need to
      // adjust the offset that we're extracting from.
      uint64_t ExtractOffsetInBits = Op.getArg(0);
      uint64_t ExtractSizeInBits = Op.getArg(1);
      if (ExtractOffsetInBits >= OffsetInBits &&
          ExtractOffsetInBits + ExtractSizeInBits <=
              OffsetInBits + SizeInBits) {
        NewElements.push_back(Op.getOp());
        NewElements.push_back(ExtractOffsetInBits - OffsetInBits);
        NewElements.push_back(ExtractSizeInBits);
        EmitFragment = false;
        continue;
      }
      // If the extracted bits aren't fully contained within the fragment then
      // give up.
      // FIXME: We could probably do better here
      return drop();
    }
    }
    Op.appendToVector(NewElements);
  }
  if (EmitFragment) {
    NewElements.push_back(dwarf::DW_OP_LLVM_fragment);
    NewElements.push_back(OffsetInBits);
    NewElements.push_back(SizeInBits);
  }
  swap();
  return true;
}

const ConstantInt *DIExprBuf::constantFold(const ConstantInt *CI) {
  assert(Ctx && "constantFold called on DIExprBuf without LLVMContext");
  // Copy the APInt so we can modify it.
  APInt NewInt = CI->getValue();
  const auto Ops = ops();
  const auto Begin = Ops.begin();
  const auto End = Ops.end();
  // Attempt to fold Op into NewInt. Returns true if successful, else false.
  auto MaybeFold = [](APInt &NewInt, DIOp::Op Op) {
    return Op.visitOverload(
        [&](DIOp::LLVMConvert Convert) {
          if (Convert.getEncoding() == dwarf::DW_ATE_signed) {
            NewInt = NewInt.sextOrTrunc(Convert.getSizeInBits());
          } else {
            assert(Convert.getEncoding() == dwarf::DW_ATE_unsigned &&
                   "Unexpected DIOp::LLVMConvert encoding");
            NewInt = NewInt.zextOrTrunc(Convert.getSizeInBits());
          }
          return true;
        },
        [=](auto) { return false; });
  };
  // We only fold prefixes of the expression, so we stop on the first
  // thing we cannot fold
  auto I = Begin;
  for (; I != End; ++I)
    if (!MaybeFold(NewInt, *I))
      break;
  // Short-circuit if we folded nothing
  if (I == Begin) {
    drop();
    return CI;
  }
  // Otherwise use the remainder of the expression
  impl::DIExpr::appendRaw(NewElements, make_range(I, End));
  swap();
  return ConstantInt::get(*Ctx, NewInt);
}

DIExprBuf &DIExprBuf::foldConstantMath() { return swap(); }

void DIExprBuf::toUIntVec(SmallVectorImpl<uint64_t> &Out) const {
  asRef().toUIntVec(Out);
}
SmallVector<uint64_t> DIExprBuf::toUIntVec() const {
  return asRef().toUIntVec();
}

DIExpression *DIExprBuf::toExpr() const {
  assert(Ctx && "toExpr called on DIExprBuf without LLVMContext");
  auto *Expr = DIExpression::get(*Ctx, Elements);
  /*
  assert(Expr->isValid() &&
         "toExpr called on DIExprBuf with invalid expression");
         */
  return Expr;
}
