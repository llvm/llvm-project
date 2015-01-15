//===-- llvm/Metadata.h - Metadata definitions ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations for metadata subclasses.
/// They represent the different flavors of metadata that live in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_METADATA_H
#define LLVM_IR_METADATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/MetadataTracking.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

namespace llvm {
class LLVMContext;
class Module;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;


enum LLVMConstants : uint32_t {
  DEBUG_METADATA_VERSION = 2  // Current debug info version number.
};

/// \brief Root of the metadata hierarchy.
///
/// This is a root class for typeless data in the IR.
class Metadata {
  friend class ReplaceableMetadataImpl;

  /// \brief RTTI.
  const unsigned char SubclassID;

protected:
  /// \brief Storage flag for non-uniqued, otherwise unowned, metadata.
  bool IsDistinctInContext : 1;
  // TODO: expose remaining bits to subclasses.

  unsigned short SubclassData16;
  unsigned SubclassData32;

public:
  enum MetadataKind {
    MDTupleKind,
    MDLocationKind,
    MDNodeFwdDeclKind,
    ConstantAsMetadataKind,
    LocalAsMetadataKind,
    MDStringKind
  };

protected:
  Metadata(unsigned ID)
      : SubclassID(ID), IsDistinctInContext(false), SubclassData16(0),
        SubclassData32(0) {}
  ~Metadata() {}

  /// \brief Store this in a big non-uniqued untyped bucket.
  bool isStoredDistinctInContext() const { return IsDistinctInContext; }

  /// \brief Default handling of a changed operand, which asserts.
  ///
  /// If subclasses pass themselves in as owners to a tracking node reference,
  /// they must provide an implementation of this method.
  void handleChangedOperand(void *, Metadata *) {
    llvm_unreachable("Unimplemented in Metadata subclass");
  }

public:
  unsigned getMetadataID() const { return SubclassID; }

  /// \brief User-friendly dump.
  void dump() const;
  void print(raw_ostream &OS) const;
  void printAsOperand(raw_ostream &OS, bool PrintType = true,
                      const Module *M = nullptr) const;
};

#define HANDLE_METADATA(CLASS) class CLASS;
#include "llvm/IR/Metadata.def"

inline raw_ostream &operator<<(raw_ostream &OS, const Metadata &MD) {
  MD.print(OS);
  return OS;
}

/// \brief Metadata wrapper in the Value hierarchy.
///
/// A member of the \a Value hierarchy to represent a reference to metadata.
/// This allows, e.g., instrinsics to have metadata as operands.
///
/// Notably, this is the only thing in either hierarchy that is allowed to
/// reference \a LocalAsMetadata.
class MetadataAsValue : public Value {
  friend class ReplaceableMetadataImpl;
  friend class LLVMContextImpl;

  Metadata *MD;

  MetadataAsValue(Type *Ty, Metadata *MD);
  ~MetadataAsValue();

  /// \brief Drop use of metadata (during teardown).
  void dropUse() { MD = nullptr; }

public:
  static MetadataAsValue *get(LLVMContext &Context, Metadata *MD);
  static MetadataAsValue *getIfExists(LLVMContext &Context, Metadata *MD);
  Metadata *getMetadata() const { return MD; }

  static bool classof(const Value *V) {
    return V->getValueID() == MetadataAsValueVal;
  }

private:
  void handleChangedMetadata(Metadata *MD);
  void track();
  void untrack();
};

/// \brief Shared implementation of use-lists for replaceable metadata.
///
/// Most metadata cannot be RAUW'ed.  This is a shared implementation of
/// use-lists and associated API for the two that support it (\a ValueAsMetadata
/// and \a TempMDNode).
class ReplaceableMetadataImpl {
  friend class MetadataTracking;

public:
  typedef MetadataTracking::OwnerTy OwnerTy;

private:
  uint64_t NextIndex;
  SmallDenseMap<void *, std::pair<OwnerTy, uint64_t>, 4> UseMap;

public:
  ReplaceableMetadataImpl() : NextIndex(0) {}
  ~ReplaceableMetadataImpl() {
    assert(UseMap.empty() && "Cannot destroy in-use replaceable metadata");
  }

  /// \brief Replace all uses of this with MD.
  ///
  /// Replace all uses of this with \c MD, which is allowed to be null.
  void replaceAllUsesWith(Metadata *MD);

  /// \brief Resolve all uses of this.
  ///
  /// Resolve all uses of this, turning off RAUW permanently.  If \c
  /// ResolveUsers, call \a UniquableMDNode::resolve() on any users whose last
  /// operand is resolved.
  void resolveAllUses(bool ResolveUsers = true);

private:
  void addRef(void *Ref, OwnerTy Owner);
  void dropRef(void *Ref);
  void moveRef(void *Ref, void *New, const Metadata &MD);

  static ReplaceableMetadataImpl *get(Metadata &MD);
};

/// \brief Value wrapper in the Metadata hierarchy.
///
/// This is a custom value handle that allows other metadata to refer to
/// classes in the Value hierarchy.
///
/// Because of full uniquing support, each value is only wrapped by a single \a
/// ValueAsMetadata object, so the lookup maps are far more efficient than
/// those using ValueHandleBase.
class ValueAsMetadata : public Metadata, ReplaceableMetadataImpl {
  friend class ReplaceableMetadataImpl;
  friend class LLVMContextImpl;

  Value *V;

  /// \brief Drop users without RAUW (during teardown).
  void dropUsers() {
    ReplaceableMetadataImpl::resolveAllUses(/* ResolveUsers */ false);
  }

protected:
  ValueAsMetadata(unsigned ID, Value *V)
      : Metadata(ID), V(V) {
    assert(V && "Expected valid value");
  }
  ~ValueAsMetadata() {}

public:
  static ValueAsMetadata *get(Value *V);
  static ConstantAsMetadata *getConstant(Value *C) {
    return cast<ConstantAsMetadata>(get(C));
  }
  static LocalAsMetadata *getLocal(Value *Local) {
    return cast<LocalAsMetadata>(get(Local));
  }

  static ValueAsMetadata *getIfExists(Value *V);
  static ConstantAsMetadata *getConstantIfExists(Value *C) {
    return cast_or_null<ConstantAsMetadata>(getIfExists(C));
  }
  static LocalAsMetadata *getLocalIfExists(Value *Local) {
    return cast_or_null<LocalAsMetadata>(getIfExists(Local));
  }

  Value *getValue() const { return V; }
  Type *getType() const { return V->getType(); }
  LLVMContext &getContext() const { return V->getContext(); }

  static void handleDeletion(Value *V);
  static void handleRAUW(Value *From, Value *To);

protected:
  /// \brief Handle collisions after \a Value::replaceAllUsesWith().
  ///
  /// RAUW isn't supported directly for \a ValueAsMetadata, but if the wrapped
  /// \a Value gets RAUW'ed and the target already exists, this is used to
  /// merge the two metadata nodes.
  void replaceAllUsesWith(Metadata *MD) {
    ReplaceableMetadataImpl::replaceAllUsesWith(MD);
  }

public:
  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == LocalAsMetadataKind ||
           MD->getMetadataID() == ConstantAsMetadataKind;
  }
};

class ConstantAsMetadata : public ValueAsMetadata {
  friend class ValueAsMetadata;

  ConstantAsMetadata(Constant *C)
      : ValueAsMetadata(ConstantAsMetadataKind, C) {}

public:
  static ConstantAsMetadata *get(Constant *C) {
    return ValueAsMetadata::getConstant(C);
  }
  static ConstantAsMetadata *getIfExists(Constant *C) {
    return ValueAsMetadata::getConstantIfExists(C);
  }

  Constant *getValue() const {
    return cast<Constant>(ValueAsMetadata::getValue());
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == ConstantAsMetadataKind;
  }
};

class LocalAsMetadata : public ValueAsMetadata {
  friend class ValueAsMetadata;

  LocalAsMetadata(Value *Local)
      : ValueAsMetadata(LocalAsMetadataKind, Local) {
    assert(!isa<Constant>(Local) && "Expected local value");
  }

public:
  static LocalAsMetadata *get(Value *Local) {
    return ValueAsMetadata::getLocal(Local);
  }
  static LocalAsMetadata *getIfExists(Value *Local) {
    return ValueAsMetadata::getLocalIfExists(Local);
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == LocalAsMetadataKind;
  }
};

/// \brief Transitional API for extracting constants from Metadata.
///
/// This namespace contains transitional functions for metadata that points to
/// \a Constants.
///
/// In prehistory -- when metadata was a subclass of \a Value -- \a MDNode
/// operands could refer to any \a Value.  There's was a lot of code like this:
///
/// \code
///     MDNode *N = ...;
///     auto *CI = dyn_cast<ConstantInt>(N->getOperand(2));
/// \endcode
///
/// Now that \a Value and \a Metadata are in separate hierarchies, maintaining
/// the semantics for \a isa(), \a cast(), \a dyn_cast() (etc.) requires three
/// steps: cast in the \a Metadata hierarchy, extraction of the \a Value, and
/// cast in the \a Value hierarchy.  Besides creating boiler-plate, this
/// requires subtle control flow changes.
///
/// The end-goal is to create a new type of metadata, called (e.g.) \a MDInt,
/// so that metadata can refer to numbers without traversing a bridge to the \a
/// Value hierarchy.  In this final state, the code above would look like this:
///
/// \code
///     MDNode *N = ...;
///     auto *MI = dyn_cast<MDInt>(N->getOperand(2));
/// \endcode
///
/// The API in this namespace supports the transition.  \a MDInt doesn't exist
/// yet, and even once it does, changing each metadata schema to use it is its
/// own mini-project.  In the meantime this API prevents us from introducing
/// complex and bug-prone control flow that will disappear in the end.  In
/// particular, the above code looks like this:
///
/// \code
///     MDNode *N = ...;
///     auto *CI = mdconst::dyn_extract<ConstantInt>(N->getOperand(2));
/// \endcode
///
/// The full set of provided functions includes:
///
///   mdconst::hasa                <=> isa
///   mdconst::extract             <=> cast
///   mdconst::extract_or_null     <=> cast_or_null
///   mdconst::dyn_extract         <=> dyn_cast
///   mdconst::dyn_extract_or_null <=> dyn_cast_or_null
///
/// The target of the cast must be a subclass of \a Constant.
namespace mdconst {

namespace detail {
template <class T> T &make();
template <class T, class Result> struct HasDereference {
  typedef char Yes[1];
  typedef char No[2];
  template <size_t N> struct SFINAE {};

  template <class U, class V>
  static Yes &hasDereference(SFINAE<sizeof(static_cast<V>(*make<U>()))> * = 0);
  template <class U, class V> static No &hasDereference(...);

  static const bool value =
      sizeof(hasDereference<T, Result>(nullptr)) == sizeof(Yes);
};
template <class V, class M> struct IsValidPointer {
  static const bool value = std::is_base_of<Constant, V>::value &&
                            HasDereference<M, const Metadata &>::value;
};
template <class V, class M> struct IsValidReference {
  static const bool value = std::is_base_of<Constant, V>::value &&
                            std::is_convertible<M, const Metadata &>::value;
};
} // end namespace detail

/// \brief Check whether Metadata has a Value.
///
/// As an analogue to \a isa(), check whether \c MD has an \a Value inside of
/// type \c X.
template <class X, class Y>
inline typename std::enable_if<detail::IsValidPointer<X, Y>::value, bool>::type
hasa(Y &&MD) {
  assert(MD && "Null pointer sent into hasa");
  if (auto *V = dyn_cast<ConstantAsMetadata>(MD))
    return isa<X>(V->getValue());
  return false;
}
template <class X, class Y>
inline
    typename std::enable_if<detail::IsValidReference<X, Y &>::value, bool>::type
    hasa(Y &MD) {
  return hasa(&MD);
}

/// \brief Extract a Value from Metadata.
///
/// As an analogue to \a cast(), extract the \a Value subclass \c X from \c MD.
template <class X, class Y>
inline typename std::enable_if<detail::IsValidPointer<X, Y>::value, X *>::type
extract(Y &&MD) {
  return cast<X>(cast<ConstantAsMetadata>(MD)->getValue());
}
template <class X, class Y>
inline
    typename std::enable_if<detail::IsValidReference<X, Y &>::value, X *>::type
    extract(Y &MD) {
  return extract(&MD);
}

/// \brief Extract a Value from Metadata, allowing null.
///
/// As an analogue to \a cast_or_null(), extract the \a Value subclass \c X
/// from \c MD, allowing \c MD to be null.
template <class X, class Y>
inline typename std::enable_if<detail::IsValidPointer<X, Y>::value, X *>::type
extract_or_null(Y &&MD) {
  if (auto *V = cast_or_null<ConstantAsMetadata>(MD))
    return cast<X>(V->getValue());
  return nullptr;
}

/// \brief Extract a Value from Metadata, if any.
///
/// As an analogue to \a dyn_cast_or_null(), extract the \a Value subclass \c X
/// from \c MD, return null if \c MD doesn't contain a \a Value or if the \a
/// Value it does contain is of the wrong subclass.
template <class X, class Y>
inline typename std::enable_if<detail::IsValidPointer<X, Y>::value, X *>::type
dyn_extract(Y &&MD) {
  if (auto *V = dyn_cast<ConstantAsMetadata>(MD))
    return dyn_cast<X>(V->getValue());
  return nullptr;
}

/// \brief Extract a Value from Metadata, if any, allowing null.
///
/// As an analogue to \a dyn_cast_or_null(), extract the \a Value subclass \c X
/// from \c MD, return null if \c MD doesn't contain a \a Value or if the \a
/// Value it does contain is of the wrong subclass, allowing \c MD to be null.
template <class X, class Y>
inline typename std::enable_if<detail::IsValidPointer<X, Y>::value, X *>::type
dyn_extract_or_null(Y &&MD) {
  if (auto *V = dyn_cast_or_null<ConstantAsMetadata>(MD))
    return dyn_cast<X>(V->getValue());
  return nullptr;
}

} // end namespace mdconst

//===----------------------------------------------------------------------===//
/// \brief A single uniqued string.
///
/// These are used to efficiently contain a byte sequence for metadata.
/// MDString is always unnamed.
class MDString : public Metadata {
  friend class StringMapEntry<MDString>;

  MDString(const MDString &) LLVM_DELETED_FUNCTION;
  MDString &operator=(MDString &&) LLVM_DELETED_FUNCTION;
  MDString &operator=(const MDString &) LLVM_DELETED_FUNCTION;

  StringMapEntry<MDString> *Entry;
  MDString() : Metadata(MDStringKind), Entry(nullptr) {}
  MDString(MDString &&) : Metadata(MDStringKind) {}

public:
  static MDString *get(LLVMContext &Context, StringRef Str);
  static MDString *get(LLVMContext &Context, const char *Str) {
    return get(Context, Str ? StringRef(Str) : StringRef());
  }

  StringRef getString() const;

  unsigned getLength() const { return (unsigned)getString().size(); }

  typedef StringRef::iterator iterator;

  /// \brief Pointer to the first byte of the string.
  iterator begin() const { return getString().begin(); }

  /// \brief Pointer to one byte past the end of the string.
  iterator end() const { return getString().end(); }

  const unsigned char *bytes_begin() const { return getString().bytes_begin(); }
  const unsigned char *bytes_end() const { return getString().bytes_end(); }

  /// \brief Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDStringKind;
  }
};

/// \brief A collection of metadata nodes that might be associated with a
/// memory access used by the alias-analysis infrastructure.
struct AAMDNodes {
  explicit AAMDNodes(MDNode *T = nullptr, MDNode *S = nullptr,
                     MDNode *N = nullptr)
      : TBAA(T), Scope(S), NoAlias(N) {}

  bool operator==(const AAMDNodes &A) const {
    return TBAA == A.TBAA && Scope == A.Scope && NoAlias == A.NoAlias;
  }

  bool operator!=(const AAMDNodes &A) const { return !(*this == A); }

  LLVM_EXPLICIT operator bool() const { return TBAA || Scope || NoAlias; }

  /// \brief The tag for type-based alias analysis.
  MDNode *TBAA;

  /// \brief The tag for alias scope specification (used with noalias).
  MDNode *Scope;

  /// \brief The tag specifying the noalias scope.
  MDNode *NoAlias;
};

// Specialize DenseMapInfo for AAMDNodes.
template<>
struct DenseMapInfo<AAMDNodes> {
  static inline AAMDNodes getEmptyKey() {
    return AAMDNodes(DenseMapInfo<MDNode *>::getEmptyKey(), 0, 0);
  }
  static inline AAMDNodes getTombstoneKey() {
    return AAMDNodes(DenseMapInfo<MDNode *>::getTombstoneKey(), 0, 0);
  }
  static unsigned getHashValue(const AAMDNodes &Val) {
    return DenseMapInfo<MDNode *>::getHashValue(Val.TBAA) ^
           DenseMapInfo<MDNode *>::getHashValue(Val.Scope) ^
           DenseMapInfo<MDNode *>::getHashValue(Val.NoAlias);
  }
  static bool isEqual(const AAMDNodes &LHS, const AAMDNodes &RHS) {
    return LHS == RHS;
  }
};

/// \brief Tracking metadata reference owned by Metadata.
///
/// Similar to \a TrackingMDRef, but it's expected to be owned by an instance
/// of \a Metadata, which has the option of registering itself for callbacks to
/// re-unique itself.
///
/// In particular, this is used by \a MDNode.
class MDOperand {
  MDOperand(MDOperand &&) LLVM_DELETED_FUNCTION;
  MDOperand(const MDOperand &) LLVM_DELETED_FUNCTION;
  MDOperand &operator=(MDOperand &&) LLVM_DELETED_FUNCTION;
  MDOperand &operator=(const MDOperand &) LLVM_DELETED_FUNCTION;

  Metadata *MD;

public:
  MDOperand() : MD(nullptr) {}
  ~MDOperand() { untrack(); }

  Metadata *get() const { return MD; }
  operator Metadata *() const { return get(); }
  Metadata *operator->() const { return get(); }
  Metadata &operator*() const { return *get(); }

  void reset() {
    untrack();
    MD = nullptr;
  }
  void reset(Metadata *MD, Metadata *Owner) {
    untrack();
    this->MD = MD;
    track(Owner);
  }

private:
  void track(Metadata *Owner) {
    if (MD) {
      if (Owner)
        MetadataTracking::track(this, *MD, *Owner);
      else
        MetadataTracking::track(MD);
    }
  }
  void untrack() {
    assert(static_cast<void *>(this) == &MD && "Expected same address");
    if (MD)
      MetadataTracking::untrack(MD);
  }
};

template <> struct simplify_type<MDOperand> {
  typedef Metadata *SimpleType;
  static SimpleType getSimplifiedValue(MDOperand &MD) { return MD.get(); }
};

template <> struct simplify_type<const MDOperand> {
  typedef Metadata *SimpleType;
  static SimpleType getSimplifiedValue(const MDOperand &MD) { return MD.get(); }
};

//===----------------------------------------------------------------------===//
/// \brief Tuple of metadata.
class MDNode : public Metadata {
  MDNode(const MDNode &) LLVM_DELETED_FUNCTION;
  void operator=(const MDNode &) LLVM_DELETED_FUNCTION;
  void *operator new(size_t) LLVM_DELETED_FUNCTION;

  LLVMContext &Context;
  unsigned NumOperands;

protected:
  unsigned MDNodeSubclassData;

  void *operator new(size_t Size, unsigned NumOps);
  void operator delete(void *Mem);

  /// \brief Required by std, but never called.
  void operator delete(void *, unsigned) {
    llvm_unreachable("Constructor throws?");
  }

  /// \brief Required by std, but never called.
  void operator delete(void *, unsigned, bool) {
    llvm_unreachable("Constructor throws?");
  }

  MDNode(LLVMContext &Context, unsigned ID, ArrayRef<Metadata *> MDs);
  ~MDNode() {}

  void dropAllReferences();

  MDOperand *mutable_begin() { return mutable_end() - NumOperands; }
  MDOperand *mutable_end() { return reinterpret_cast<MDOperand *>(this); }

public:
  static inline MDNode *get(LLVMContext &Context, ArrayRef<Metadata *> MDs);
  static inline MDNode *getIfExists(LLVMContext &Context,
                                    ArrayRef<Metadata *> MDs);
  static inline MDNode *getDistinct(LLVMContext &Context,
                                    ArrayRef<Metadata *> MDs);

  /// \brief Return a temporary MDNode
  ///
  /// For use in constructing cyclic MDNode structures. A temporary MDNode is
  /// not uniqued, may be RAUW'd, and must be manually deleted with
  /// deleteTemporary.
  static MDNodeFwdDecl *getTemporary(LLVMContext &Context,
                                     ArrayRef<Metadata *> MDs);

  /// \brief Deallocate a node created by getTemporary.
  ///
  /// The node must not have any users.
  static void deleteTemporary(MDNode *N);

  LLVMContext &getContext() const { return Context; }

  /// \brief Replace a specific operand.
  void replaceOperandWith(unsigned I, Metadata *New);

  /// \brief Check if node is fully resolved.
  bool isResolved() const;

  /// \brief Check if node is distinct.
  ///
  /// Distinct nodes are not uniqued, and will not be returned by \a
  /// MDNode::get().
  bool isDistinct() const {
    return isStoredDistinctInContext() || isa<MDNodeFwdDecl>(this);
  }

protected:
  /// \brief Set an operand.
  ///
  /// Sets the operand directly, without worrying about uniquing.
  void setOperand(unsigned I, Metadata *New);

public:
  typedef const MDOperand *op_iterator;
  typedef iterator_range<op_iterator> op_range;

  op_iterator op_begin() const {
    return const_cast<MDNode *>(this)->mutable_begin();
  }
  op_iterator op_end() const {
    return const_cast<MDNode *>(this)->mutable_end();
  }
  op_range operands() const { return op_range(op_begin(), op_end()); }

  const MDOperand &getOperand(unsigned I) const {
    assert(I < NumOperands && "Out of range");
    return op_begin()[I];
  }

  /// \brief Return number of MDNode operands.
  unsigned getNumOperands() const { return NumOperands; }

  /// \brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDTupleKind ||
           MD->getMetadataID() == MDLocationKind ||
           MD->getMetadataID() == MDNodeFwdDeclKind;
  }

  /// \brief Check whether MDNode is a vtable access.
  bool isTBAAVtableAccess() const;

  /// \brief Methods for metadata merging.
  static MDNode *concatenate(MDNode *A, MDNode *B);
  static MDNode *intersect(MDNode *A, MDNode *B);
  static MDNode *getMostGenericTBAA(MDNode *A, MDNode *B);
  static AAMDNodes getMostGenericAA(const AAMDNodes &A, const AAMDNodes &B);
  static MDNode *getMostGenericFPMath(MDNode *A, MDNode *B);
  static MDNode *getMostGenericRange(MDNode *A, MDNode *B);
};

/// \brief Uniquable metadata node.
///
/// A uniquable metadata node.  This contains the basic functionality
/// for implementing sub-types of \a MDNode that can be uniqued like
/// constants.
///
/// There is limited support for RAUW at construction time.  At
/// construction time, if any operands are an instance of \a
/// MDNodeFwdDecl (or another unresolved \a UniquableMDNode, which
/// indicates an \a MDNodeFwdDecl in its path), the node itself will be
/// unresolved.  As soon as all operands become resolved, it will drop
/// RAUW support permanently.
///
/// If an unresolved node is part of a cycle, \a resolveCycles() needs
/// to be called on some member of the cycle when each \a MDNodeFwdDecl
/// has been removed.
class UniquableMDNode : public MDNode {
  friend class ReplaceableMetadataImpl;
  friend class MDNode;
  friend class LLVMContextImpl;

  /// \brief Support RAUW as long as one of its arguments is replaceable.
  ///
  /// FIXME: Save memory by storing this in a pointer union with the
  /// LLVMContext, and adding an LLVMContext reference to RMI.
  std::unique_ptr<ReplaceableMetadataImpl> ReplaceableUses;

protected:
  /// \brief Create a new node.
  ///
  /// If \c AllowRAUW, then if any operands are unresolved support RAUW.  RAUW
  /// will be dropped once all operands have been resolved (or if \a
  /// resolveCycles() is called).
  UniquableMDNode(LLVMContext &C, unsigned ID, ArrayRef<Metadata *> Vals,
                  bool AllowRAUW);
  ~UniquableMDNode() {}

  void storeDistinctInContext();

public:
  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDTupleKind ||
           MD->getMetadataID() == MDLocationKind;
  }

  /// \brief Check whether any operands are forward declarations.
  ///
  /// Returns \c true as long as any operands (or their operands, etc.) are \a
  /// MDNodeFwdDecl.
  ///
  /// As forward declarations are resolved, their containers should get
  /// resolved automatically.  However, if this (or one of its operands) is
  /// involved in a cycle, \a resolveCycles() needs to be called explicitly.
  bool isResolved() const { return !ReplaceableUses; }

  /// \brief Resolve cycles.
  ///
  /// Once all forward declarations have been resolved, force cycles to be
  /// resolved.
  ///
  /// \pre No operands (or operands' operands, etc.) are \a MDNodeFwdDecl.
  void resolveCycles();

private:
  void handleChangedOperand(void *Ref, Metadata *New);

  void resolve();
  void resolveAfterOperandChange(Metadata *Old, Metadata *New);
  void decrementUnresolvedOperandCount();

  void deleteAsSubclass();
  UniquableMDNode *uniquify();
  void eraseFromStore();
};

/// \brief Tuple of metadata.
///
/// This is the simple \a MDNode arbitrary tuple.  Nodes are uniqued by
/// default based on their operands.
class MDTuple : public UniquableMDNode {
  friend class LLVMContextImpl;
  friend class UniquableMDNode;

  MDTuple(LLVMContext &C, ArrayRef<Metadata *> Vals, bool AllowRAUW)
      : UniquableMDNode(C, MDTupleKind, Vals, AllowRAUW) {}
  ~MDTuple() { dropAllReferences(); }

  void setHash(unsigned Hash) { MDNodeSubclassData = Hash; }
  void recalculateHash();

  static MDTuple *getImpl(LLVMContext &Context, ArrayRef<Metadata *> MDs,
                          bool ShouldCreate);

public:
  /// \brief Get the hash, if any.
  unsigned getHash() const { return MDNodeSubclassData; }

  static MDTuple *get(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
    return getImpl(Context, MDs, /* ShouldCreate */ true);
  }
  static MDTuple *getIfExists(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
    return getImpl(Context, MDs, /* ShouldCreate */ false);
  }

  /// \brief Return a distinct node.
  ///
  /// Return a distinct node -- i.e., a node that is not uniqued.
  static MDTuple *getDistinct(LLVMContext &Context, ArrayRef<Metadata *> MDs);

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDTupleKind;
  }

private:
  MDTuple *uniquifyImpl();
  void eraseFromStoreImpl();
};

MDNode *MDNode::get(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
  return MDTuple::get(Context, MDs);
}
MDNode *MDNode::getIfExists(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
  return MDTuple::getIfExists(Context, MDs);
}
MDNode *MDNode::getDistinct(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
  return MDTuple::getDistinct(Context, MDs);
}

/// \brief Debug location.
///
/// A debug location in source code, used for debug info and otherwise.
class MDLocation : public UniquableMDNode {
  friend class LLVMContextImpl;
  friend class UniquableMDNode;

  MDLocation(LLVMContext &C, unsigned Line, unsigned Column,
             ArrayRef<Metadata *> MDs, bool AllowRAUW);
  ~MDLocation() { dropAllReferences(); }

  static MDLocation *constructHelper(LLVMContext &Context, unsigned Line,
                                     unsigned Column, Metadata *Scope,
                                     Metadata *InlinedAt, bool AllowRAUW);

  static MDLocation *getImpl(LLVMContext &Context, unsigned Line,
                             unsigned Column, Metadata *Scope,
                             Metadata *InlinedAt, bool ShouldCreate);

  // Disallow replacing operands.
  void replaceOperandWith(unsigned I, Metadata *New) LLVM_DELETED_FUNCTION;

public:
  static MDLocation *get(LLVMContext &Context, unsigned Line, unsigned Column,
                         Metadata *Scope, Metadata *InlinedAt = nullptr) {
    return getImpl(Context, Line, Column, Scope, InlinedAt,
                   /* ShouldCreate */ true);
  }
  static MDLocation *getIfExists(LLVMContext &Context, unsigned Line,
                                 unsigned Column, Metadata *Scope,
                                 Metadata *InlinedAt = nullptr) {
    return getImpl(Context, Line, Column, Scope, InlinedAt,
                   /* ShouldCreate */ false);
  }
  static MDLocation *getDistinct(LLVMContext &Context, unsigned Line,
                                 unsigned Column, Metadata *Scope,
                                 Metadata *InlinedAt = nullptr);

  unsigned getLine() const { return MDNodeSubclassData; }
  unsigned getColumn() const { return SubclassData16; }
  Metadata *getScope() const { return getOperand(0); }
  Metadata *getInlinedAt() const {
    if (getNumOperands() == 2)
      return getOperand(1);
    return nullptr;
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocationKind;
  }

private:
  MDLocation *uniquifyImpl();
  void eraseFromStoreImpl();
};

/// \brief Forward declaration of metadata.
///
/// Forward declaration of metadata, in the form of a basic tuple.  Unlike \a
/// MDTuple, this class has full support for RAUW, is not owned, is not
/// uniqued, and is suitable for forward references.
class MDNodeFwdDecl : public MDNode, ReplaceableMetadataImpl {
  friend class Metadata;
  friend class ReplaceableMetadataImpl;

  MDNodeFwdDecl(LLVMContext &C, ArrayRef<Metadata *> Vals)
      : MDNode(C, MDNodeFwdDeclKind, Vals) {}

public:
  ~MDNodeFwdDecl() { dropAllReferences(); }

  // MSVC doesn't see the alternative: "using MDNode::operator delete".
  void operator delete(void *Mem) { MDNode::operator delete(Mem); }

  static MDNodeFwdDecl *get(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
    return new (MDs.size()) MDNodeFwdDecl(Context, MDs);
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDNodeFwdDeclKind;
  }

  using ReplaceableMetadataImpl::replaceAllUsesWith;
};

//===----------------------------------------------------------------------===//
/// \brief A tuple of MDNodes.
///
/// Despite its name, a NamedMDNode isn't itself an MDNode. NamedMDNodes belong
/// to modules, have names, and contain lists of MDNodes.
///
/// TODO: Inherit from Metadata.
class NamedMDNode : public ilist_node<NamedMDNode> {
  friend class SymbolTableListTraits<NamedMDNode, Module>;
  friend struct ilist_traits<NamedMDNode>;
  friend class LLVMContextImpl;
  friend class Module;
  NamedMDNode(const NamedMDNode &) LLVM_DELETED_FUNCTION;

  std::string Name;
  Module *Parent;
  void *Operands; // SmallVector<TrackingMDRef, 4>

  void setParent(Module *M) { Parent = M; }

  explicit NamedMDNode(const Twine &N);

  template<class T1, class T2>
  class op_iterator_impl :
      public std::iterator<std::bidirectional_iterator_tag, T2> {
    const NamedMDNode *Node;
    unsigned Idx;
    op_iterator_impl(const NamedMDNode *N, unsigned i) : Node(N), Idx(i) { }

    friend class NamedMDNode;

  public:
    op_iterator_impl() : Node(nullptr), Idx(0) { }

    bool operator==(const op_iterator_impl &o) const { return Idx == o.Idx; }
    bool operator!=(const op_iterator_impl &o) const { return Idx != o.Idx; }
    op_iterator_impl &operator++() {
      ++Idx;
      return *this;
    }
    op_iterator_impl operator++(int) {
      op_iterator_impl tmp(*this);
      operator++();
      return tmp;
    }
    op_iterator_impl &operator--() {
      --Idx;
      return *this;
    }
    op_iterator_impl operator--(int) {
      op_iterator_impl tmp(*this);
      operator--();
      return tmp;
    }

    T1 operator*() const { return Node->getOperand(Idx); }
  };

public:
  /// \brief Drop all references and remove the node from parent module.
  void eraseFromParent();

  /// \brief Remove all uses and clear node vector.
  void dropAllReferences();

  ~NamedMDNode();

  /// \brief Get the module that holds this named metadata collection.
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }

  MDNode *getOperand(unsigned i) const;
  unsigned getNumOperands() const;
  void addOperand(MDNode *M);
  void setOperand(unsigned I, MDNode *New);
  StringRef getName() const;
  void print(raw_ostream &ROS) const;
  void dump() const;

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef op_iterator_impl<MDNode *, MDNode> op_iterator;
  op_iterator op_begin() { return op_iterator(this, 0); }
  op_iterator op_end()   { return op_iterator(this, getNumOperands()); }

  typedef op_iterator_impl<const MDNode *, MDNode> const_op_iterator;
  const_op_iterator op_begin() const { return const_op_iterator(this, 0); }
  const_op_iterator op_end()   const { return const_op_iterator(this, getNumOperands()); }

  inline iterator_range<op_iterator>  operands() {
    return iterator_range<op_iterator>(op_begin(), op_end());
  }
  inline iterator_range<const_op_iterator> operands() const {
    return iterator_range<const_op_iterator>(op_begin(), op_end());
  }
};

} // end llvm namespace

#endif
