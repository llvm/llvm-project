//===- CfgTraits.h - Traits for generically working on CFGs -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a traits template \ref CfgTraits as well as the
/// \ref CfgInterface abstract interface and \ref CfgInterfaceImpl that help
/// in writing algorithms that are generic over CFGs, e.g. operating on both
/// LLVM IR and MachineIR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CFGTRAITS_H
#define LLVM_SUPPORT_CFGTRAITS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Printable.h"

namespace llvm {

template <typename Tag> class CfgOpaqueType;

template <typename Tag>
bool operator==(CfgOpaqueType<Tag> lhs, CfgOpaqueType<Tag> rhs);
template <typename Tag>
bool operator<(CfgOpaqueType<Tag> lhs, CfgOpaqueType<Tag> rhs);

/// \brief Type-erased references to CFG objects (blocks, values).
///
/// Use CfgTraits::{wrapRef, unwrapRef} to wrap and unwrap concrete object
/// references.
///
/// The most common use is to hold a pointer, but arbitrary uintptr_t values
/// may be stored by CFGs. Note that 0, -1, and -2 have special interpretations:
///  * 0 / nullptr: default-constructed value; evaluates to false in boolean
///                 contexts.
///  * -1: dense map empty marker
///  * -2: dense map tombstone
template <typename Tag> class CfgOpaqueType {
  friend class CfgTraitsBase;
  friend struct DenseMapInfo<CfgOpaqueType<Tag>>;
  template <typename BaseTraits, typename FullTraits> friend class CfgTraits;
  template <typename T>
  friend bool operator==(CfgOpaqueType<T>, CfgOpaqueType<T>);
  template <typename T>
  friend bool operator<(CfgOpaqueType<T>, CfgOpaqueType<T>);

  void *ptr = nullptr;

  explicit CfgOpaqueType(void *ptr) : ptr(ptr) {}
  void *get() const { return ptr; }

public:
  CfgOpaqueType() = default;

  explicit operator bool() const { return ptr != nullptr; }
};

template <typename Tag>
bool operator==(CfgOpaqueType<Tag> lhs, CfgOpaqueType<Tag> rhs) {
  return lhs.get() == rhs.get();
}

template <typename Tag>
bool operator!=(CfgOpaqueType<Tag> lhs, CfgOpaqueType<Tag> rhs) {
  return !(lhs == rhs);
}

template <typename Tag>
bool operator<(CfgOpaqueType<Tag> lhs, CfgOpaqueType<Tag> rhs) {
  return lhs.get() < rhs.get();
}

template <typename Tag> struct DenseMapInfo<CfgOpaqueType<Tag>> {
  using Type = CfgOpaqueType<Tag>;

  static Type getEmptyKey() {
    uintptr_t val = static_cast<uintptr_t>(-1);
    return Type(reinterpret_cast<void *>(val));
  }

  static Type getTombstoneKey() {
    uintptr_t val = static_cast<uintptr_t>(-2);
    return Type(reinterpret_cast<void *>(val));
  }

  static unsigned getHashValue(Type val) {
    return llvm::DenseMapInfo<void *>::getHashValue(val.get());
  }
  static bool isEqual(Type lhs, Type rhs) { return lhs == rhs; }
};

class CfgParentRefTag;
using CfgParentRef = CfgOpaqueType<CfgParentRefTag>;

class CfgBlockRefTag;
using CfgBlockRef = CfgOpaqueType<CfgBlockRefTag>;

class CfgValueRefTag;
using CfgValueRef = CfgOpaqueType<CfgValueRefTag>;

/// \brief Base class for CFG traits
///
/// Derive from this base class to define the mapping between opaque types and
/// concrete CFG types. Then derive from \ref CfgTraits to implement
/// operations such as traversal of the CFG.
class CfgTraitsBase {
protected:
  template <typename Tag> static auto makeOpaque(void *ptr) {
    CfgOpaqueType<Tag> ref;
    ref.ptr = ptr;
    return ref;
  }

  template <typename Tag> static void *getOpaque(CfgOpaqueType<Tag> opaque) {
    return opaque.ptr;
  }

public:
  // To be implemented by derived classes:
  //
  // - The type of the "parent" of the CFG, e.g. `llvm::Function`
  //   using ParentType = ...;
  //
  // - The type of block references in the CFG, e.g. `llvm::BasicBlock *`
  //   using BlockRef = ...;
  //
  // - The type of value references in the CFG, e.g. `llvm::Value *`
  //   using ValueRef = ...;
  //
  // - Static methods for converting BlockRef and ValueRef to and from
  //   static CfgBlockRef wrapRef(BlockRef);
  //   static CfgValueRef wrapRef(ValueRef);
  //   static BlockRef unwrapRef(CfgBlockRef);
  //   static ValueRef unwrapRef(CfgValueRef);
};

/// \brief CFG traits
///
/// Implement CFG traits by:
///  - Deriving from CfgTraitsBase to designate block and value types and
///    implementing wrapRef / unwrapRef
///  - Deriving from CfgTraits using CRTP and implement / override additional
///    methods for CFG traversal, printing, etc.
///
/// This somewhat surprising two-step helps with the implementation of
/// (un)wrapping_iterators.
///
template <typename BaseTraits, typename FullTraits>
class CfgTraits : public BaseTraits {
public:
  using typename BaseTraits::BlockRef;
  using typename BaseTraits::ParentType;
  using typename BaseTraits::ValueRef;

  /// Functionality to be provided by implementations:
  ///@{

  // Constructor: initialize from a pointer to the parent.
  //   explicit CfgTraits(ParentType *parent);

  // Find the parent for a given block.
  //   static ParentType *getBlockParent(BlockRef block);

  // Iterate over blocks in the CFG containing the given block in an arbitrary
  // order (start with entry block, return a range of iterators dereferencing
  // to BlockRef):
  //   static auto blocks(ParentType *parent);

  // Iterate over the predecessors / successors of a block (return a range
  // of iterators dereferencing to BlockRef):
  //   static auto predecessors(BlockRef block);
  //   static auto successors(BlockRef block);

  // Iterate over the values defined in a basic block in program order (return
  // a range of iterators dereferencing to ValueRef):
  //   static auto blockdefs(BlockRef block);

  // Get the block in which a given value is defined. Returns a null-like
  // BlockRef if the value is not defined in a block (e.g. it is a constant or
  // function argument).
  //   BlockRef getValueDefBlock(ValueRef value) const;

  // struct Printer {
  //   explicit Printer(const CfgTraits &traits);
  //   void printBlockName(raw_ostream &out, BlockRef block) const;
  //   void printValue(raw_ostream &out, ValueRef value) const;
  // };

  ///@}

  static CfgParentRef wrapRef(ParentType *parent) {
    return CfgParentRef{parent};
  }

  static ParentType *unwrapRef(CfgParentRef parent) {
    return static_cast<ParentType *>(parent.get());
  }

  using BaseTraits::unwrapRef;
  using BaseTraits::wrapRef;

  template <typename BaseIteratorT> struct unwrapping_iterator;

  template <typename BaseIteratorT>
  using unwrapping_iterator_base = iterator_adaptor_base<
      unwrapping_iterator<BaseIteratorT>, BaseIteratorT,
      typename std::iterator_traits<BaseIteratorT>::iterator_category,
      // value_type
      decltype(BaseTraits::unwrapRef(*std::declval<BaseIteratorT>())),
      typename std::iterator_traits<BaseIteratorT>::difference_type,
      // pointer (not really usable, but we need to put something here)
      decltype(BaseTraits::unwrapRef(*std::declval<BaseIteratorT>())) *,
      // reference (not a true reference, because operator* doesn't return one)
      decltype(BaseTraits::unwrapRef(*std::declval<BaseIteratorT>()))>;

  template <typename BaseIteratorT>
  struct unwrapping_iterator : unwrapping_iterator_base<BaseIteratorT> {
    using Base = unwrapping_iterator_base<BaseIteratorT>;

    unwrapping_iterator() = default;
    explicit unwrapping_iterator(BaseIteratorT &&it)
        : Base(std::forward<BaseIteratorT>(it)) {}

    auto operator*() const { return BaseTraits::unwrapRef(*this->I); }
  };

  template <typename BaseIteratorT> struct wrapping_iterator;

  template <typename BaseIteratorT>
  using wrapping_iterator_base = iterator_adaptor_base<
      wrapping_iterator<BaseIteratorT>, BaseIteratorT,
      typename std::iterator_traits<BaseIteratorT>::iterator_category,
      // value_type
      decltype(BaseTraits::wrapRef(*std::declval<BaseIteratorT>())),
      typename std::iterator_traits<BaseIteratorT>::difference_type,
      // pointer (not really usable, but we need to put something here)
      decltype(BaseTraits::wrapRef(*std::declval<BaseIteratorT>())) *,
      // reference (not a true reference, because operator* doesn't return one)
      decltype(BaseTraits::wrapRef(*std::declval<BaseIteratorT>()))>;

  template <typename BaseIteratorT>
  struct wrapping_iterator : wrapping_iterator_base<BaseIteratorT> {
    using Base = wrapping_iterator_base<BaseIteratorT>;

    wrapping_iterator() = default;
    explicit wrapping_iterator(BaseIteratorT &&it)
        : Base(std::forward<BaseIteratorT>(it)) {}

    auto operator*() const { return BaseTraits::wrapRef(*this->I); }
  };

  /// Convert an iterator of CfgBlockRef or CfgValueRef into an iterator of
  /// BlockRef or ValueRef.
  template <typename IteratorT> static auto unwrapIterator(IteratorT &&it) {
    return unwrapping_iterator<IteratorT>(std::forward<IteratorT>(it));
  }

  /// Convert a range of CfgBlockRef or CfgValueRef into a range of
  /// BlockRef or ValueRef.
  template <typename RangeT> static auto unwrapRange(RangeT &&range) {
    return llvm::make_range(
        unwrapIterator(adl_begin(std::forward<RangeT>(range))),
        unwrapIterator(adl_end(std::forward<RangeT>(range))));
  }

  /// Convert an iterator of BlockRef or ValueRef into an iterator of
  /// CfgBlockRef or CfgValueRef.
  template <typename IteratorT> static auto wrapIterator(IteratorT &&it) {
    return wrapping_iterator<IteratorT>(std::forward<IteratorT>(it));
  }

  /// Convert a range of BlockRef or ValueRef into a range of CfgBlockRef or
  /// CfgValueRef.
  template <typename RangeT> static auto wrapRange(RangeT &&range) {
    return llvm::make_range(
        wrapIterator(adl_begin(std::forward<RangeT>(range))),
        wrapIterator(adl_end(std::forward<RangeT>(range))));
  }
};

/// \brief Obtain CfgTraits given the basic block type.
///
/// This template is provided to ease the transition to the use of CfgTraits.
/// Existing templates e.g. over the basic block type can use this to derive
/// the appropriate CfgTraits implementation via
/// typename CfgTraitsFor<BlockT>::CfgTraits.
template <typename CfgRelatedTypeT> struct CfgTraitsFor;
// Specializations need to include:
//   using CfgTraits = ...;

class CfgPrinter;

/// \brief Type-erased "CFG traits"
///
/// Non-template algorithms that operate generically over CFG types can use this
/// interface to query for CFG-specific functionality.
///
/// Note: This interface should only be implemented by \ref CfgInterfaceImpl.
class CfgInterface {
  virtual void anchor();

public:
  virtual ~CfgInterface() = default;

  /// Escape-hatch for obtaining a printer e.g. in debug code. Prefer to
  /// explicitly pass a CfgPrinter where possible.
  virtual std::unique_ptr<CfgPrinter> makePrinter() const = 0;

  virtual CfgParentRef getBlockParent(CfgBlockRef block) const = 0;

  virtual void appendBlocks(CfgParentRef parent,
                            SmallVectorImpl<CfgBlockRef> &list) const = 0;

  virtual void appendPredecessors(CfgBlockRef block,
                                  SmallVectorImpl<CfgBlockRef> &list) const = 0;
  virtual void appendSuccessors(CfgBlockRef block,
                                SmallVectorImpl<CfgBlockRef> &list) const = 0;
  virtual ArrayRef<CfgBlockRef>
  getPredecessors(CfgBlockRef block,
                  SmallVectorImpl<CfgBlockRef> &store) const = 0;
  virtual ArrayRef<CfgBlockRef>
  getSuccessors(CfgBlockRef block,
                SmallVectorImpl<CfgBlockRef> &store) const = 0;

  virtual void appendBlockDefs(CfgBlockRef block,
                               SmallVectorImpl<CfgValueRef> &list) const = 0;
  virtual CfgBlockRef getValueDefBlock(CfgValueRef value) const = 0;
};

/// \brief Type-erased "CFG printer"
///
/// Separate from CfgInterface because some CFG printing requires tracking
/// expensive data structures, and we'd like to avoid the cost of
/// (conditionally) tearing them down in the common case.
class CfgPrinter {
  virtual void anchor();

protected:
  const CfgInterface &m_iface;

  CfgPrinter(const CfgInterface &iface) : m_iface(iface) {}

public:
  virtual ~CfgPrinter() {}

  const CfgInterface &getInterface() const { return m_iface; }

  virtual void printBlockName(raw_ostream &out, CfgBlockRef block) const = 0;
  virtual void printValue(raw_ostream &out, CfgValueRef value) const = 0;

  Printable printableBlockName(CfgBlockRef block) const {
    return Printable(
        [this, block](raw_ostream &out) { printBlockName(out, block); });
  }
  Printable printableValue(CfgValueRef value) const {
    return Printable(
        [this, value](raw_ostream &out) { printValue(out, value); });
  }
};

template <typename CfgTraitsT> class CfgPrinterImpl;

/// \brief Implementation of type-erased "CFG traits"
///
/// Note: Do not specialize this template; adjust the CfgTraits type instead
/// where necessary.
template <typename CfgTraitsT>
class CfgInterfaceImpl final : public CfgInterface,
                               private CfgTraitsT { // empty base optimization
public:
  using CfgTraits = CfgTraitsT;
  using BlockRef = typename CfgTraits::BlockRef;
  using ValueRef = typename CfgTraits::ValueRef;
  using ParentType = typename CfgTraits::ParentType;

  friend CfgPrinterImpl<CfgTraits>;

public:
  explicit CfgInterfaceImpl(ParentType *parent) : CfgTraits(parent) {}

  std::unique_ptr<CfgPrinter> makePrinter() const final {
    return std::make_unique<CfgPrinterImpl<CfgTraits>>(*this);
  }

  CfgParentRef getBlockParent(CfgBlockRef block) const final {
    return CfgTraits::wrapRef(
        CfgTraits::getBlockParent(CfgTraits::unwrapRef(block)));
  }

  void appendBlocks(CfgParentRef parent,
                    SmallVectorImpl<CfgBlockRef> &list) const final {
    auto range = CfgTraits::blocks(CfgTraits::unwrapRef(parent));
    list.insert(list.end(), CfgTraits::wrapIterator(std::begin(range)),
                CfgTraits::wrapIterator(std::end(range)));
  }

  void appendPredecessors(CfgBlockRef block,
                          SmallVectorImpl<CfgBlockRef> &list) const final {
    auto range = CfgTraits::predecessors(CfgTraits::unwrapRef(block));
    list.insert(list.end(), CfgTraits::wrapIterator(std::begin(range)),
                CfgTraits::wrapIterator(std::end(range)));
  }
  void appendSuccessors(CfgBlockRef block,
                        SmallVectorImpl<CfgBlockRef> &list) const final {
    auto range = CfgTraits::successors(CfgTraits::unwrapRef(block));
    list.insert(list.end(), CfgTraits::wrapIterator(std::begin(range)),
                CfgTraits::wrapIterator(std::end(range)));
  }
  ArrayRef<CfgBlockRef>
  getPredecessors(CfgBlockRef block,
                  SmallVectorImpl<CfgBlockRef> &store) const final {
    // TODO: Can this be optimized for concrete CFGs that already have the
    //       "right" in-memory representation of predecessors / successors?
    store.clear();
    appendPredecessors(block, store);
    return store;
  }
  ArrayRef<CfgBlockRef>
  getSuccessors(CfgBlockRef block,
                SmallVectorImpl<CfgBlockRef> &store) const final {
    // TODO: Can this be optimized for concrete CFGs that already have the
    //       "right" in-memory representation of predecessors / successors?
    store.clear();
    appendSuccessors(block, store);
    return store;
  }

  void appendBlockDefs(CfgBlockRef block,
                       SmallVectorImpl<CfgValueRef> &list) const final {
    auto range = CfgTraits::blockdefs(CfgTraits::unwrapRef(block));
    list.insert(list.end(), CfgTraits::wrapIterator(std::begin(range)),
                CfgTraits::wrapIterator(std::end(range)));
  }

  CfgBlockRef getValueDefBlock(CfgValueRef value) const final {
    return CfgTraits::wrapRef(
        CfgTraits::getValueDefBlock(CfgTraits::unwrapRef(value)));
  }
};

/// \brief Implementation of type-erased "CFG traits"
///
/// Note: Do not specialize this template; adjust the CfgTraits type instead
/// where necessary.
template <typename CfgTraitsT>
class CfgPrinterImpl : public CfgPrinter,
                       private CfgTraitsT::Printer { // empty base optimization
public:
  using CfgTraits = CfgTraitsT;
  using BlockRef = typename CfgTraits::BlockRef;
  using ValueRef = typename CfgTraits::ValueRef;

public:
  explicit CfgPrinterImpl(const CfgInterfaceImpl<CfgTraits> &impl)
      : CfgPrinter(impl), CfgTraitsT::Printer(impl) {}

  void printBlockName(raw_ostream &out, CfgBlockRef block) const final {
    CfgTraits::Printer::printBlockName(out, CfgTraits::unwrapRef(block));
  }
  void printValue(raw_ostream &out, CfgValueRef value) const final {
    CfgTraits::Printer::printValue(out, CfgTraits::unwrapRef(value));
  }
};

} // namespace llvm

#endif // LLVM_SUPPORT_CFGTRAITS_H
