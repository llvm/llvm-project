//===- DebugInfoExprs.h - Debug Info Expression Manipulation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines types for working with DIExpression in a type-safe way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFOEXPRS_H
#define LLVM_IR_DEBUGINFOEXPRS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DbgVariableFragmentInfo.h"
#include "llvm/IR/DebugInfoCommon.h"
#include <cassert>
#include <climits>
#include <cstdint>
#include <optional>

namespace llvm {

class ConstantInt;
class DIExpression;
class DIVariable;
class LLVMContext;

namespace impl {
class DIExpr;
} // namespace impl
class DIExprRef;
class DIExprBuf;

namespace DIOp {

namespace impl {
enum class TagT : uint8_t {
  LLVMEscape = 0,
#define HANDLE_OP(NAME, ENCODING) NAME
#define SEPARATOR ,
#include "llvm/IR/DIOps.def"
};
struct CommonInitialSequence {
  TagT Tag;
  constexpr CommonInitialSequence(TagT Tag) : Tag(Tag) {}
};
} // namespace impl

// Below are the concrete alternative types that a DIOp::Op encapsulates.

class LLVMEscape {
  impl::CommonInitialSequence CIS = impl::TagT::LLVMEscape;
  uint32_t Size;
  const uint64_t *Data;

public:
  constexpr explicit LLVMEscape(const uint64_t *Data, uint32_t Size)
      : Size(Size), Data(Data) {}
  explicit LLVMEscape(ArrayRef<uint64_t> A)
      : Size(static_cast<uint32_t>(A.size())), Data(A.data()) {}
  constexpr bool operator==(const LLVMEscape &O) const {
    return Size == O.Size && Data == O.Data;
  }
  static constexpr StringRef getAsmName() { return "DIOpLLVMEscape"; }
  void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;
  ArrayRef<uint64_t> getData() const { return {Data, Size}; }
};

#define HANDLE_OP0(NAME, ENCODING)                                             \
  class NAME {                                                                 \
    impl::CommonInitialSequence CIS = impl::TagT::NAME;                        \
                                                                               \
  public:                                                                      \
    explicit constexpr NAME() {}                                               \
    bool operator==(const NAME &O) const;                                      \
    void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;                      \
    static StringRef getAsmName();                                             \
    static uint64_t getDwarfEncoding();                                        \
  };
#define HANDLE_OP1(NAME, ENCODING, TYPE1, NAME1)                               \
  class NAME {                                                                 \
    impl::CommonInitialSequence CIS = impl::TagT::NAME;                        \
    TYPE1 NAME1;                                                               \
                                                                               \
  public:                                                                      \
    explicit constexpr NAME(TYPE1 NAME1) : NAME1(NAME1) {}                     \
    bool operator==(const NAME &O) const;                                      \
    void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;                      \
    TYPE1 get##NAME1() const;                                                  \
    void set##NAME1(TYPE1 NAME1);                                              \
    static StringRef getAsmName();                                             \
    static uint64_t getDwarfEncoding();                                        \
  };
#define HANDLE_OP2(NAME, ENCODING, TYPE1, NAME1, TYPE2, NAME2)                 \
  class NAME {                                                                 \
    impl::CommonInitialSequence CIS = impl::TagT::NAME;                        \
    TYPE1 NAME1;                                                               \
    TYPE2 NAME2;                                                               \
                                                                               \
  public:                                                                      \
    explicit constexpr NAME(TYPE1 NAME1, TYPE2 NAME2)                          \
        : NAME1(NAME1), NAME2(NAME2) {}                                        \
    bool operator==(const NAME &O) const;                                      \
    void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;                      \
    TYPE1 get##NAME1() const;                                                  \
    void set##NAME1(TYPE1 NAME1);                                              \
    TYPE2 get##NAME2() const;                                                  \
    void set##NAME2(TYPE2 NAME2);                                              \
    static StringRef getAsmName();                                             \
    static uint64_t getDwarfEncoding();                                        \
  };
#include "llvm/IR/DIOps.def"

class Op {
  union {
    impl::CommonInitialSequence CIS;
#define HANDLE_OP(NAME, ENCODING) DIOp::NAME NAME;
    HANDLE_OP(LLVMEscape, void)
#include "llvm/IR/DIOps.def"
  };

public:
#define HANDLE_OP(NAME, ENCODING)                                              \
  constexpr Op(DIOp::NAME V) : NAME(V) {}
  HANDLE_OP(LLVMEscape, void)
#include "llvm/IR/DIOps.def"

  template <typename T> bool holds() const;
  template <typename... T> bool holdsOneOf() const {
    return (holds<T>() || ...);
  }
  template <typename T> T get() const;
  template <typename T> std::optional<T> getIf() const;

  template <typename... CallableTs>
  decltype(auto) visitOverload(CallableTs &&...Callables) const;
  template <typename R, typename... CallableTs>
  R visitOverload(CallableTs &&...Callables) const;
  void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;
};
#define HANDLE_OP(NAME, ENCODING)                                              \
  template <> bool Op::holds<DIOp::NAME>() const;                              \
  template <> DIOp::NAME Op::get<DIOp::NAME>() const;                          \
  template <> std::optional<DIOp::NAME> Op::getIf<DIOp::NAME>() const;
#include "llvm/IR/DIOps.def"

template <typename... CallableTs>
inline decltype(auto) Op::visitOverload(CallableTs &&...Callables) const {
  auto Visitor = makeVisitor(std::forward<CallableTs>(Callables)...);
  switch (CIS.Tag) {
#define HANDLE_OP(NAME, ENCODING)                                              \
  case impl::TagT::NAME:                                                       \
    return Visitor(get<DIOp::NAME>());
    HANDLE_OP(LLVMEscape, void)
#include "llvm/IR/DIOps.def"
  }
  llvm_unreachable("DIOp::visitOverload does not handle all tags");
}
template <typename R, typename... CallableTs>
inline R Op::visitOverload(CallableTs &&...Callables) const {
  auto Visitor = makeVisitor(std::forward<CallableTs>(Callables)...);
  switch (CIS.Tag) {
#define HANDLE_OP(NAME, ENCODING)                                              \
  case impl::TagT::NAME:                                                       \
    return Visitor(get<DIOp::NAME>());
    HANDLE_OP(LLVMEscape, void)
#include "llvm/IR/DIOps.def"
  }
  llvm_unreachable("DIOp::visitOverload does not handle all tags");
}

class FromUIntIterator
    : public iterator_facade_base<FromUIntIterator, std::forward_iterator_tag,
                                  DIOp::Op, std::ptrdiff_t, DIOp::Op,
                                  DIOp::Op> {
  friend class llvm::impl::DIExpr;
  // Each iterator knows the End so we can transparently yield an LLVMEscape
  // of all remaining ops if we encounter an invalid expression. We don't
  // keep a lot of these iterators around, so the doubling in size shouldn't
  // be significant. If it ever becomes an issue, we could explore using
  // the "sentinel" support from RangeTS when/if that becomes available to
  // make the actual end() an empty struct.
  const uint64_t *I = nullptr;
  const uint64_t *End = nullptr;

  uint32_t getRemainingSize() const;
  uint32_t getCurrentOpSize() const;

  struct ArrowProxy {
    Op O;
    Op *operator->() { return &O; }
  };

public:
  static iterator_range<FromUIntIterator> makeRange(ArrayRef<uint64_t> From) {
    return {FromUIntIterator(From.begin(), From.end()),
            FromUIntIterator(From.end(), From.end())};
  }
  FromUIntIterator(const uint64_t *Op, const uint64_t *End) : I(Op), End(End) {}
  FromUIntIterator(const FromUIntIterator &R) : I(R.I), End(R.End) {}
  FromUIntIterator &operator=(const FromUIntIterator &R) {
    I = R.I;
    End = R.End;
    return *this;
  }
  bool operator==(const FromUIntIterator &R) const {
    return I == R.I && End == R.End;
  }
  Op operator*() const;
  ArrowProxy operator->() const { return ArrowProxy{**this}; }
  FromUIntIterator &operator++() {
    I += getCurrentOpSize();
    return *this;
  }
  FromUIntIterator operator++(int) {
    auto O = *this;
    I += getCurrentOpSize();
    return O;
  }
};

} // namespace DIOp

class DIExprRef {
  friend class impl::DIExpr;
  friend class DIExprBuf;
  iterator_range<DIOp::FromUIntIterator> Ops;

  explicit DIExprRef(iterator_range<DIOp::FromUIntIterator> Ops) : Ops(Ops) {};
  explicit DIExprRef(DIOp::FromUIntIterator Begin, DIOp::FromUIntIterator End)
      : Ops(make_range(Begin, End)) {};

  std::optional<DIOp::Op> maybeAdvance(DIOp::FromUIntIterator &I) const {
    if (I != Ops.end())
      return *I++;
    return std::nullopt;
  };
  std::optional<DIExprRef> getSingleLocationExprRef() const;

public:
  explicit DIExprRef(const DIExpression *From);

  using ExtOps = std::array<DIOp::Op, 2>;

  /// Returns the ops for a zero- or sign-extension in a DIExpression.
  static ExtOps getExtOps(unsigned FromSize, unsigned ToSize, bool Signed);

  std::optional<DbgVariableFragmentInfo> getFragmentInfo() const;
  bool isValid() const;
  bool isSingleLocationExpression() const;
  bool startsWithDeref() const;
  bool isDeref() const;
  bool isImplicit() const;
  bool isComplex() const;
  bool isEntryValue() const;
  std::optional<SignedOrUnsignedConstant> isConstant() const;
  uint64_t getNumLocationOperands() const;
  std::optional<uint64_t> getActiveBits(DIVariable *Var) const;
  std::optional<int64_t> extractIfOffset() const;
  std::optional<std::pair<int64_t, DIExprRef>> extractLeadingOffset() const;
  bool hasAllLocationOps(unsigned N) const;
  void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;
  SmallVector<uint64_t> toUIntVec() const;
};

class DIExprBuf {
  friend class impl::DIExpr;
  LLVMContext *Ctx = nullptr;
  SmallVector<uint64_t, 0> Elements;

  // Nearly all operations require double-buffering, so we bake it in.
  // This allows us to re-use a small number of allocations for the
  // processing of many expressions, even where each expression may require
  // multiple operations.
  //
  // Each method has as an implicit post-condition that the backing buffer
  // NewElements is empty, and so on entry it can be used without being cleared.
  SmallVector<uint64_t, 0> NewElements;
  // Helper for failure paths in fallible methods to clear backing buffer.
  bool drop() {
    NewElements.clear();
    return false;
  }
  // Helper for success paths to swap buffers and clear new backing buffer.
  DIExprBuf &swap() {
    std::swap(Elements, NewElements);
    drop();
    return *this;
  }

  iterator_range<DIOp::FromUIntIterator> ops() {
    return DIOp::FromUIntIterator::makeRange(Elements);
  }

  DIExprBuf &prependOpcodesFinalize(bool StackValue, bool EntryValue);

public:
  DIExprBuf() = default;
  explicit DIExprBuf(LLVMContext *Ctx);
  explicit DIExprBuf(const DIExpression *From);

  /// Clear the buffer and assign From into it as-if it were being newly
  /// constructed. Useful where many expressions are manipulated to amortize
  /// allocation costs.
  DIExprBuf &assign(const DIExpression *From);
  DIExprBuf &assign(LLVMContext *Ctx, DIExprRef From);

  DIExprRef asRef() const {
    return DIExprRef(DIOp::FromUIntIterator::makeRange(Elements));
  }

  /*
  static DIExprBuf canonicalize(LLVMContext *Ctx, DIExprRef From,
                                bool IsIndirect);
  static DIExprBuf canonicalize(const DIExpression *From, bool IsIndirect);
  */

  DIExprBuf &appendRaw(iterator_range<const DIOp::Op *> NewOps);
  DIExprBuf &appendRaw(std::initializer_list<DIOp::Op> NewOps);

  DIExprBuf &clear();
  DIExprBuf &convertToUndefExpression();
  DIExprBuf &convertToVariadicExpressionUnchecked();
  DIExprBuf &convertToVariadicExpression();
  bool convertToNonVariadicExpression();
  DIExprBuf &prepend(uint8_t Flags, int64_t Offset = 0);

  DIExprBuf &append(iterator_range<const DIOp::Op *> NewOps);
  DIExprBuf &append(std::initializer_list<DIOp::Op> NewOps);
  DIExprBuf &append(DIExprRef NewOps);
  DIExprBuf &append(iterator_range<DIOp::FromUIntIterator> NewOps);

  DIExprBuf &prependOpcodes(iterator_range<const DIOp::Op *> NewOps,
                            bool StackValue = false, bool EntryValue = false);
  DIExprBuf &prependOpcodes(std::initializer_list<DIOp::Op> NewOps,
                            bool StackValue = false, bool EntryValue = false);
  DIExprBuf &prependOpcodes(DIExprRef NewOps, bool StackValue = false,
                            bool EntryValue = false);
  DIExprBuf &prependOpcodes(iterator_range<DIOp::FromUIntIterator> NewOps,
                            bool StackValue = false, bool EntryValue = false);

  DIExprBuf &appendOpsToArg(iterator_range<const DIOp::Op *> NewOps,
                            unsigned ArgIndex, bool StackValue = false);
  DIExprBuf &appendOpsToArg(std::initializer_list<DIOp::Op> NewOps,
                            unsigned ArgIndex, bool StackValue = false);
  DIExprBuf &appendOpsToArg(DIExprRef NewOps, unsigned ArgIndex,
                            bool StackValue = false);
  DIExprBuf &appendOpsToArg(iterator_range<DIOp::FromUIntIterator> NewOps,
                            unsigned ArgIndex, bool StackValue = false);

  DIExprBuf &appendConstant(SignedOrUnsignedConstant SignedOrUnsigned,
                            uint64_t Value);
  DIExprBuf &appendOffset(int64_t Offset);

  DIExprBuf &replaceArg(uint64_t OldArgIndex, uint64_t NewArgIndex);
  bool createFragmentExpression(unsigned OffsetInBits, unsigned SizeInBits);

  const ConstantInt *constantFold(const ConstantInt *CI);

  DIExprBuf &foldConstantMath();

  void toUIntVec(SmallVectorImpl<uint64_t> &Out) const;
  SmallVector<uint64_t> toUIntVec() const;

  DIExpression *toExpr() const;
};

} // namespace llvm

#endif // LLVM_IR_DEBUGINFOEXPRS_H
