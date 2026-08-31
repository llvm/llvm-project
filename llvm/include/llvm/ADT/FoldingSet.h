//===- llvm/ADT/FoldingSet.h - Uniquing Hash Set ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines a hash set that can be used to remove duplication of nodes
/// in a graph.  This code was originally created by Chris Lattner for use with
/// SelectionDAGCSEMap, but was isolated to provide use across the llvm code
/// set.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_FOLDINGSET_H
#define LLVM_ADT_FOLDINGSET_H

#include "llvm/ADT/EpochTracker.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/xxhash.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace llvm {

/// This folding set is used for two purposes:
///   1. Given information about a node we want to create, look up the unique
///      instance of the node in the set.  If the node already exists, return
///      it, otherwise return a token that makes the insertion cheap.
///   2. Given a node that has already been created, remove it from the set.
///
/// The hash table is linear-probing open addressing with tombstone-free
/// deletion, power-of-two capacity, and a 0.75 maximum load factor.
///
/// Any node that is to be included in the folding set must be a subclass of
/// FoldingSetNode.  The node class must also define a Profile method used to
/// establish the unique bits of data for the node.  The Profile method is
/// passed a FoldingSetNodeID object which is used to gather the bits.  Just
/// call one of the Add* functions defined in the FoldingSetNodeID class.
/// NOTE: That the folding set does not own the nodes and it is the
/// responsibility of the user to dispose of the nodes.
///
/// Eg.
///    class MyNode : public FoldingSetNode {
///    private:
///      std::string Name;
///      unsigned Value;
///    public:
///      MyNode(const char *N, unsigned V) : Name(N), Value(V) {}
///       ...
///      void Profile(FoldingSetNodeID &ID) const {
///        ID.AddString(Name);
///        ID.AddInteger(Value);
///      }
///      ...
///    };
///
/// To define the folding set itself use the FoldingSet template;
///
/// Eg.
///    FoldingSet<MyNode> MyFoldingSet;
///
/// Four public methods are available to manipulate the folding set;
///
/// 1) If you have an existing node that you want add to the set but unsure
/// that the node might already exist then call;
///
///    MyNode *M = MyFoldingSet.getOrInsert(N);
///
/// If The result is equal to the input then the node has been inserted.
/// Otherwise, the result is the node existing in the folding set, and the
/// input can be discarded (use the result instead.)
///
/// 2) If you are ready to construct a node but want to check if it already
/// exists, then call lookup with a FoldingSetNodeID of the bits to check;
///
///   FoldingSetNodeID ID;
///   ID.AddString(Name);
///   ID.AddInteger(Value);
///   FoldingSetInsertToken Token;
///
///    MyNode *M = MyFoldingSet.lookup(ID, Token);
///
/// If found then M will be non-NULL, else Token holds what insert needs to
/// place the node.
///
/// 3) If you get a NULL result from lookup then you can insert a new node with
/// insert;
///
///    MyNode *N = new MyNode(Name, Value);
///    MyFoldingSet.insert(N, Token);
///
/// Token survives intervening insertions, but N must profile identically to
/// the ID that produced it, or N becomes unfindable.
///
/// 4) Finally, if you want to remove a node from the folding set call;
///
///    bool WasRemoved = MyFoldingSet.erase(M);
///
/// The result indicates whether the node existed in the folding set.

class FoldingSetNodeID;
class StringRef;

//===----------------------------------------------------------------------===//

/// This class provides default implementations for FoldingSetTrait
/// implementations.
template <typename T> struct DefaultFoldingSetTrait {
  struct ContextStorage {};

  static void Profile(const T &X, FoldingSetNodeID &ID) { X.Profile(ID); }
  static void Profile(T &X, FoldingSetNodeID &ID) { X.Profile(ID); }

  // Equals - Test if the profile for X would match ID, using TempID
  // to compute a temporary ID if necessary. The default implementation
  // just calls Profile and does a regular comparison. Implementations
  // can override this to provide more efficient implementations.
  static inline bool Equals(T &X, const FoldingSetNodeID &ID,
                            FoldingSetNodeID &TempID);
};

/// This trait class is used to define behavior of how to "profile" (in the
/// FoldingSet parlance) an object of a given type.
/// The default behavior is to invoke a 'Profile' method on an object, but
/// through template specialization the behavior can be tailored for specific
/// types.  Combined with the FoldingSetNodeWrapper class, one can add objects
/// to FoldingSets that were not originally designed to have that behavior.
template <typename T, typename Enable = void>
struct FoldingSetTrait : public DefaultFoldingSetTrait<T> {};

/// Like DefaultFoldingSetTrait, but for ContextualFoldingSets.
template <typename T, typename Ctx> struct DefaultContextualFoldingSetTrait {
  struct ContextStorage {
    Ctx Context;
    explicit ContextStorage(Ctx Context) : Context(Context) {}
    Ctx getContext() const { return Context; }
  };

  static void Profile(T &X, FoldingSetNodeID &ID, Ctx Context) {
    X.Profile(ID, Context);
  }

  static inline bool Equals(T &X, const FoldingSetNodeID &ID,
                            FoldingSetNodeID &TempID, Ctx Context);
};

/// Like FoldingSetTrait, but for ContextualFoldingSets.
template <typename T, typename Ctx>
struct ContextualFoldingSetTrait : DefaultContextualFoldingSetTrait<T, Ctx> {};

//===--------------------------------------------------------------------===//
/// This class describes a reference to an interned FoldingSetNodeID, which can
/// be a useful to store node id data rather than using plain FoldingSetNodeIDs,
/// since the 32-element SmallVector is often much larger than necessary, and
/// the possibility of heap allocation means it requires a non-trivial
/// destructor call.
class FoldingSetNodeIDRef {
  const unsigned *Data = nullptr;
  size_t Size = 0;

public:
  FoldingSetNodeIDRef() = default;
  FoldingSetNodeIDRef(const unsigned *D, size_t S) : Data(D), Size(S) {}

  static constexpr unsigned NotAHash = 0;

  // Compute a strong hash value used to lookup the node in the FoldingSetBase.
  // The hash value is not guaranteed to be deterministic across processes.
  // Never returns NotAHash: FoldingSetBase reserves it for the empty insert
  // token and for a node belonging to no set.
  unsigned computeHash() const {
    unsigned Hash =
        static_cast<unsigned>(hash_combine_range(Data, Data + Size));
    return Hash == NotAHash ? 1 : Hash;
  }

  // Compute a deterministic hash value across processes that is suitable for
  // on-disk serialization.
  unsigned computeStableHash() const {
    return static_cast<unsigned>(xxh3_64bits(
        reinterpret_cast<const uint8_t *>(Data), sizeof(unsigned) * Size));
  }

  bool operator==(FoldingSetNodeIDRef RHS) const {
    return Size == RHS.Size &&
           memcmp(Data, RHS.Data, Size * sizeof(*Data)) == 0;
  }

  bool operator!=(FoldingSetNodeIDRef RHS) const { return !(*this == RHS); }

  /// Used to compare the "ordering" of two nodes as defined by the
  /// profiled bits and their ordering defined by memcmp().
  LLVM_ABI bool operator<(FoldingSetNodeIDRef) const;

  const unsigned *getData() const { return Data; }
  size_t getSize() const { return Size; }
};

//===--------------------------------------------------------------------===//
/// This class is used to gather all the unique data bits of a node.  When all
/// the bits are gathered this class is used to produce a hash value for the
/// node.
class FoldingSetNodeID {
  /// Vector of all the data bits that make the node unique.
  /// Use a SmallVector to avoid a heap allocation in the common case.
  SmallVector<unsigned, 32> Bits;

  template <typename T> void AddIntegerImpl(T I) {
    static_assert(std::is_integral_v<T> && sizeof(T) <= sizeof(unsigned) * 2,
                  "T must be an integer type no wider than 64 bits");
    Bits.push_back(static_cast<unsigned>(I));
    if constexpr (sizeof(unsigned) < sizeof(T))
      Bits.push_back(static_cast<unsigned long long>(I) >> 32);
  }

public:
  FoldingSetNodeID() = default;

  FoldingSetNodeID(FoldingSetNodeIDRef Ref)
      : Bits(Ref.getData(), Ref.getData() + Ref.getSize()) {}

  /// Add* - Add various data types to Bit data.
  void AddPointer(const void *Ptr) {
    // Note: this adds pointers to the hash using sizes and endianness that
    // depend on the host. It doesn't matter, however, because hashing on
    // pointer values is inherently unstable. Nothing should depend on the
    // ordering of nodes in the folding set.
    static_assert(sizeof(uintptr_t) <= sizeof(unsigned long long),
                  "unexpected pointer size");
    AddInteger(reinterpret_cast<uintptr_t>(Ptr));
  }
  void AddInteger(signed I) { AddIntegerImpl(I); }
  void AddInteger(unsigned I) { AddIntegerImpl(I); }
  void AddInteger(long I) { AddIntegerImpl(I); }
  void AddInteger(unsigned long I) { AddIntegerImpl(I); }
  void AddInteger(long long I) { AddIntegerImpl(I); }
  void AddInteger(unsigned long long I) { AddIntegerImpl(I); }
  void AddBoolean(bool B) { AddInteger(B ? 1U : 0U); }
  LLVM_ABI void AddString(StringRef String);
  LLVM_ABI void AddNodeID(const FoldingSetNodeID &ID);

  template <typename T> inline void Add(const T &x) {
    FoldingSetTrait<T>::Profile(x, *this);
  }

  /// Clear the accumulated profile, allowing this FoldingSetNodeID
  /// object to be used to compute a new profile.
  inline void clear() { Bits.clear(); }

  // Compute a strong hash value for this FoldingSetNodeID, used to lookup the
  // node in the FoldingSetBase. The hash value is not guaranteed to be
  // deterministic across processes.
  unsigned computeHash() const {
    return FoldingSetNodeIDRef(Bits.data(), Bits.size()).computeHash();
  }

  // Compute a deterministic hash value across processes that is suitable for
  // on-disk serialization.
  unsigned computeStableHash() const {
    return FoldingSetNodeIDRef(Bits.data(), Bits.size()).computeStableHash();
  }

  /// operator== - Used to compare two nodes to each other.
  bool operator==(const FoldingSetNodeID &RHS) const {
    return *this == FoldingSetNodeIDRef(RHS.Bits.data(), RHS.Bits.size());
  }
  bool operator==(const FoldingSetNodeIDRef RHS) const {
    return FoldingSetNodeIDRef(Bits.data(), Bits.size()) == RHS;
  }

  bool operator!=(const FoldingSetNodeID &RHS) const { return !(*this == RHS); }
  bool operator!=(const FoldingSetNodeIDRef RHS) const {
    return !(*this == RHS);
  }

  /// Used to compare the "ordering" of two nodes as defined by the
  /// profiled bits and their ordering defined by memcmp().
  LLVM_ABI bool operator<(const FoldingSetNodeID &RHS) const;
  LLVM_ABI bool operator<(const FoldingSetNodeIDRef RHS) const;

  /// Copy this node's data to a memory region allocated from the
  /// given allocator and return a FoldingSetNodeIDRef describing the
  /// interned data.
  LLVM_ABI FoldingSetNodeIDRef Intern(BumpPtrAllocator &Allocator) const;
};

/// Insertion token: a failed lookup fills it in, the matching insert consumes
/// it.
class FoldingSetInsertToken {
  uint32_t Hash = FoldingSetNodeIDRef::NotAHash;

  explicit FoldingSetInsertToken(uint32_t Hash) : Hash(Hash) {
    assert(Hash != FoldingSetNodeIDRef::NotAHash && "Invalid insert token");
  }

  friend class FoldingSetBase;

public:
  FoldingSetInsertToken() = default;
  explicit operator bool() const {
    return Hash != FoldingSetNodeIDRef::NotAHash;
  }

  friend bool operator==(FoldingSetInsertToken A, FoldingSetInsertToken B) {
    return A.Hash == B.Hash;
  }
  friend bool operator!=(FoldingSetInsertToken A, FoldingSetInsertToken B) {
    return !(A == B);
  }
};

//===----------------------------------------------------------------------===//
/// Non-templated base class for FoldingSet and ContextualFoldingSet, holding
/// the memory management and probing that does not depend on the node type.
class FoldingSetBase : public DebugEpochBase {
protected:
  /// Array of node pointers; a null entry marks an empty slot.
  void **Buckets = nullptr;

  /// Length of the Buckets array.  Always a power of 2.
  unsigned NumBuckets = 0;

  /// Number of nodes in the folding set.
  unsigned NumNodes = 0;

  LLVM_ABI explicit FoldingSetBase(unsigned Log2InitSize);
  LLVM_ABI FoldingSetBase(FoldingSetBase &&Arg);
  LLVM_ABI FoldingSetBase &operator=(FoldingSetBase &&RHS);
  LLVM_ABI ~FoldingSetBase();

public:
  //===--------------------------------------------------------------------===//
  /// This class is used to maintain node state in a folding set.
  class Node {
  private:
    // Hash of the node's profile, cached so that growth and removal never
    // re-run Profile(). NotAHash while the node is in no folding set.
    uint32_t FoldingSetHash = FoldingSetNodeIDRef::NotAHash;

  public:
    Node() = default;

    // Accessors
    uint32_t getFoldingSetHash() const { return FoldingSetHash; }
    void setFoldingSetHash(uint32_t Hash) { FoldingSetHash = Hash; }
  };

  /// Remove all nodes from the folding set.
  LLVM_ABI void clear();

  /// Returns the number of nodes in the folding set.
  unsigned size() const { return NumNodes; }

  /// Returns true if there are no nodes in the folding set.
  [[nodiscard]] bool empty() const { return NumNodes == 0; }

  /// Grow the number of buckets so that we can hold at least \p N nodes
  /// before rebucketing. May allocate more space than requested.
  LLVM_ABI void reserve(unsigned N);

private:
  /// Put \p N in the first empty slot following its home, without checking
  /// capacity. Does not touch \p N, so a rehash need not dirty every node.
  void placeNode(Node *N, uint32_t Hash);

  /// Rehash into at least \p MinNumBuckets buckets, rounded up to a power of
  /// two and floored at the constructor's minimum.
  void grow(unsigned MinNumBuckets);

protected:
  // The below methods are protected to encourage subclasses to provide a more
  // type-safe API.

  /// Remove a node from the folding set, returning true if one
  /// was removed or false if the node was not in the folding set.
  LLVM_ABI bool erase(Node *N);

  /// Walk the probe chain for \p Hash, offering each node whose cached hash
  /// matches to \p IsMatch. \p IsMatch is a template parameter so that it, and
  /// the profile it may build, inline into the loop.
  template <typename MatchFn>
  Node *probe(uint32_t Hash, FoldingSetInsertToken &Token, MatchFn IsMatch) {
    assert(Hash != FoldingSetNodeIDRef::NotAHash && "Hash must be normalized");
    unsigned Mask = NumBuckets - 1;
    for (unsigned I = Hash & Mask; Buckets[I]; I = (I + 1) & Mask) {
      Node *N = static_cast<Node *>(Buckets[I]);
      if (N->getFoldingSetHash() == Hash && IsMatch(N)) {
        Token = {};
        return N;
      }
    }

    Token = FoldingSetInsertToken(Hash);
    return nullptr;
  }

  /// Insert the specified node into the folding set, knowing that it is not
  /// already in the folding set.  \p Token must come from lookup for an ID that
  /// \p N profiles identically to.
  LLVM_ABI void insert(Node *N, FoldingSetInsertToken Token);
};

// Convenience type to hide the implementation of the folding set.
using FoldingSetNode = FoldingSetBase::Node;
template <class T> class FoldingSetIterator;

// Definitions of FoldingSetTrait and ContextualFoldingSetTrait functions, which
// require the definition of FoldingSetNodeID.
template <typename T>
inline bool DefaultFoldingSetTrait<T>::Equals(T &X, const FoldingSetNodeID &ID,
                                              FoldingSetNodeID &TempID) {
  FoldingSetTrait<T>::Profile(X, TempID);
  return TempID == ID;
}
template <typename T, typename Ctx>
inline bool DefaultContextualFoldingSetTrait<T, Ctx>::Equals(
    T &X, const FoldingSetNodeID &ID, FoldingSetNodeID &TempID, Ctx Context) {
  ContextualFoldingSetTrait<T, Ctx>::Profile(X, TempID, Context);
  return TempID == ID;
}

//===----------------------------------------------------------------------===//
/// An implementation detail that lets us share code between FoldingSet and
/// ContextualFoldingSet.
template <class T, class Trait = FoldingSetTrait<T>>
class FoldingSetImpl : public FoldingSetBase, public Trait::ContextStorage {
  void nodeProfile(FoldingSetNode *N, FoldingSetNodeID &ID) const {
    if constexpr (std::is_empty_v<typename Trait::ContextStorage>)
      Trait::Profile(*static_cast<T *>(N), ID);
    else
      Trait::Profile(*static_cast<T *>(N), ID, this->getContext());
  }

  bool nodeEquals(FoldingSetNode *N, const FoldingSetNodeID &ID) const {
    // Trait::Equals profiles into TempID without clearing it first, so each
    // candidate needs its own.
    FoldingSetNodeID TempID;
    if constexpr (std::is_empty_v<typename Trait::ContextStorage>)
      return Trait::Equals(*static_cast<T *>(N), ID, TempID);
    else
      return Trait::Equals(*static_cast<T *>(N), ID, TempID,
                           this->getContext());
  }

public:
  explicit FoldingSetImpl(unsigned Log2InitSize = 6)
      : FoldingSetBase(Log2InitSize) {}

  template <typename C, typename = std::enable_if_t<std::is_constructible_v<
                            typename Trait::ContextStorage, C>>>
  explicit FoldingSetImpl(C &&Context, unsigned Log2InitSize = 6)
      : FoldingSetBase(Log2InitSize),
        Trait::ContextStorage(std::forward<C>(Context)) {}

  FoldingSetImpl(FoldingSetImpl &&Arg) = default;
  FoldingSetImpl &operator=(FoldingSetImpl &&RHS) = default;
  ~FoldingSetImpl() = default;

public:
  using iterator = FoldingSetIterator<T>;

  iterator begin() { return iterator(Buckets, Buckets + NumBuckets, this); }
  iterator end() {
    return iterator(Buckets + NumBuckets, Buckets + NumBuckets, this);
  }

  using const_iterator = FoldingSetIterator<const T>;

  const_iterator begin() const {
    return const_iterator(Buckets, Buckets + NumBuckets, this);
  }
  const_iterator end() const {
    return const_iterator(Buckets + NumBuckets, Buckets + NumBuckets, this);
  }

  /// Remove a node from the folding set, returning true if one
  /// was removed or false if the node was not in the folding set.
  bool erase(T *N) { return FoldingSetBase::erase(N); }

  /// If there is an existing node exactly equal to the specified node,
  /// return it.  Otherwise, insert 'N' and return it instead.
  ///
  /// Out of line so that callers do not inherit the ID's inline storage; some
  /// of them recurse.
  LLVM_ATTRIBUTE_NOINLINE T *getOrInsert(T *N) {
    FoldingSetNodeID ID;
    nodeProfile(N, ID);
    FoldingSetInsertToken Token;
    if (T *E = lookup(ID, Token))
      return E;
    FoldingSetBase::insert(N, Token);
    return N;
  }

  /// Look up the node specified by ID. If it exists, return it and clear
  /// \p Token; otherwise return null and set \p Token for a subsequent insert.
  T *lookup(const FoldingSetNodeID &ID, FoldingSetInsertToken &Token) {
    return static_cast<T *>(
        probe(ID.computeHash(), Token,
              [&](FoldingSetNode *N) { return nodeEquals(N, ID); }));
  }

  /// Insert the specified node into the folding set, knowing that it is not
  /// already in the folding set.  \p Token must come from lookup for an ID that
  /// \p N profiles identically to.
  void insert(T *N, FoldingSetInsertToken Token) {
    FoldingSetBase::insert(N, Token);
  }

  /// Insert the specified node into the folding set, knowing that it is not
  /// already in the folding set.
  void insert(T *N) {
    T *Inserted = getOrInsert(N);
    (void)Inserted;
    assert(Inserted == N && "Node already inserted!");
  }
};

//===----------------------------------------------------------------------===//
/// This template class is used to instantiate a specialized
/// implementation of the folding set to the node class T.  T must be a
/// subclass of FoldingSetNode and implement a Profile function.
///
/// Note that this set type is movable and move-assignable. However, its
/// moved-from state is not a valid state for anything other than
/// move-assigning and destroying. This is primarily to enable movable APIs
/// that incorporate these objects.
template <class T, class Trait = FoldingSetTrait<T>>
using FoldingSet = FoldingSetImpl<T, Trait>;

//===----------------------------------------------------------------------===//
/// This template class is a further refinement of FoldingSet which provides a
/// context argument when calling Profile on its nodes.  Currently, that
/// argument is fixed at initialization time.
///
/// T must be a subclass of FoldingSetNode and implement a Profile
/// function with signature
///   void Profile(FoldingSetNodeID &, Ctx);
template <class T, class Ctx>
using ContextualFoldingSet =
    FoldingSetImpl<T, ContextualFoldingSetTrait<T, Ctx>>;

//===----------------------------------------------------------------------===//
/// This template class combines a FoldingSet and a vector to provide the
/// interface of FoldingSet but with deterministic iteration order based on the
/// insertion order. T must be a subclass of FoldingSetNode and implement a
/// Profile function.
template <class T, class VectorT = SmallVector<T *, 8>> class FoldingSetVector {
  FoldingSet<T> Set;
  VectorT Vector;

public:
  explicit FoldingSetVector(unsigned Log2InitSize = 6) : Set(Log2InitSize) {}

  using iterator = pointee_iterator<typename VectorT::iterator>;

  iterator begin() { return Vector.begin(); }
  iterator end() { return Vector.end(); }

  using const_iterator = pointee_iterator<typename VectorT::const_iterator>;

  const_iterator begin() const { return Vector.begin(); }
  const_iterator end() const { return Vector.end(); }

  /// Remove all nodes from the folding set.
  void clear() {
    Set.clear();
    Vector.clear();
  }

  /// Look up the node specified by ID. If it exists, return it and clear
  /// \p Token; otherwise return null and set \p Token for a subsequent insert.
  T *lookup(const FoldingSetNodeID &ID, FoldingSetInsertToken &Token) {
    return Set.lookup(ID, Token);
  }

  /// If there is an existing node exactly equal to the specified node,
  /// return it.  Otherwise, insert 'N' and return it instead.
  T *getOrInsert(T *N) {
    T *Result = Set.getOrInsert(N);
    if (Result == N)
      Vector.push_back(N);
    return Result;
  }

  /// Insert the specified node into the folding set, knowing that it is not
  /// already in the folding set.  \p Token must come from lookup for an ID that
  /// \p N profiles identically to.
  void insert(T *N, FoldingSetInsertToken Token) {
    Set.insert(N, Token);
    Vector.push_back(N);
  }

  /// Insert the specified node into the folding set, knowing that
  /// it is not already in the folding set.
  void insert(T *N) {
    Set.insert(N);
    Vector.push_back(N);
  }

  /// Returns the number of nodes in the folding set.
  unsigned size() const { return Set.size(); }

  /// Returns true if there are no nodes in the folding set.
  [[nodiscard]] bool empty() const { return Set.empty(); }
};

//===----------------------------------------------------------------------===//
/// Forward iterator for FoldingSet and ContextualFoldingSet.
template <class T> class FoldingSetIterator : DebugEpochBase::HandleBase {
  void **Bucket = nullptr;
  void **End = nullptr;

  void advance() {
    assert(isHandleInSync() && "invalid iterator access!");
    do
      ++Bucket;
    while (Bucket != End && *Bucket == nullptr);
  }

public:
  FoldingSetIterator(void **Bucket, void **End, const DebugEpochBase *Epoch)
      : DebugEpochBase::HandleBase(Epoch), Bucket(Bucket), End(End) {
    while (this->Bucket != this->End && *this->Bucket == nullptr)
      ++this->Bucket;
  }

  T &operator*() const {
    assert(isHandleInSync() && "invalid iterator access!");
    return *static_cast<T *>(static_cast<FoldingSetNode *>(*Bucket));
  }

  T *operator->() const { return &operator*(); }

  inline FoldingSetIterator &operator++() { // Preincrement
    advance();
    return *this;
  }
  FoldingSetIterator operator++(int) { // Postincrement
    FoldingSetIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool operator==(const FoldingSetIterator &RHS) const {
    assert(isComparableWith(RHS) && "incomparable iterators!");
    return Bucket == RHS.Bucket;
  }
  bool operator!=(const FoldingSetIterator &RHS) const {
    return !(*this == RHS);
  }
};

//===----------------------------------------------------------------------===//
/// This template class is used to "wrap" arbitrary types in an enclosing object
/// so that they can be inserted into FoldingSets.
template <typename T> class FoldingSetNodeWrapper : public FoldingSetNode {
  T data;

public:
  template <typename... Ts>
  explicit FoldingSetNodeWrapper(Ts &&...Args)
      : data(std::forward<Ts>(Args)...) {}

  void Profile(FoldingSetNodeID &ID) { FoldingSetTrait<T>::Profile(data, ID); }

  T &getValue() { return data; }
  const T &getValue() const { return data; }

  operator T &() { return data; }
  operator const T &() const { return data; }
};

//===----------------------------------------------------------------------===//
/// This is a subclass of FoldingSetNode which stores a FoldingSetNodeID value
/// rather than requiring the node to recompute it each time it is needed. This
/// trades space for speed (which can be significant if the ID is long), and it
/// also permits nodes to drop information that would otherwise only be required
/// for recomputing an ID.
class FastFoldingSetNode : public FoldingSetNode {
  FoldingSetNodeID FastID;

protected:
  explicit FastFoldingSetNode(const FoldingSetNodeID &ID) : FastID(ID) {}

public:
  void Profile(FoldingSetNodeID &ID) const { ID.AddNodeID(FastID); }
};

//===----------------------------------------------------------------------===//
// Partial specializations of FoldingSetTrait.

template <typename T> struct FoldingSetTrait<T *> {
  static inline void Profile(T *X, FoldingSetNodeID &ID) { ID.AddPointer(X); }
};
template <typename T1, typename T2> struct FoldingSetTrait<std::pair<T1, T2>> {
  static inline void Profile(const std::pair<T1, T2> &P, FoldingSetNodeID &ID) {
    ID.Add(P.first);
    ID.Add(P.second);
  }
};

template <typename T>
struct FoldingSetTrait<T, std::enable_if_t<std::is_enum<T>::value>> {
  static void Profile(const T &X, FoldingSetNodeID &ID) {
    ID.AddInteger(llvm::to_underlying(X));
  }
};

} // namespace llvm

#endif // LLVM_ADT_FOLDINGSET_H
