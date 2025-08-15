//===- InMemoryCAS.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/ADT/LazyAtomicPointer.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/TrieRawHashMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ThreadSafeAllocator.h"
#include "llvm/Support/TrailingObjects.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::builtin;

namespace {

class InMemoryObject;

/// Index of referenced IDs (map: Hash -> InMemoryObject*). Uses
/// LazyAtomicPointer to coordinate creation of objects.
using InMemoryIndexT =
    ThreadSafeTrieRawHashMap<LazyAtomicPointer<const InMemoryObject>,
                             sizeof(HashType)>;

/// Values in \a InMemoryIndexT. \a InMemoryObject's point at this to access
/// their hash.
using InMemoryIndexValueT = InMemoryIndexT::value_type;

/// Builtin InMemory CAS that stores CAS object in the memory.
class InMemoryObject {
public:
  enum class Kind {
    /// Node with refs and data.
    RefNode,

    /// Node with refs and data co-allocated.
    InlineNode,

    Max = InlineNode,
  };

  Kind getKind() const { return IndexAndKind.getInt(); }
  const InMemoryIndexValueT &getIndex() const {
    assert(IndexAndKind.getPointer());
    return *IndexAndKind.getPointer();
  }

  ArrayRef<uint8_t> getHash() const { return getIndex().Hash; }

  InMemoryObject() = delete;
  InMemoryObject(InMemoryObject &&) = delete;
  InMemoryObject(const InMemoryObject &) = delete;

protected:
  InMemoryObject(Kind K, const InMemoryIndexValueT &I) : IndexAndKind(&I, K) {}

private:
  enum Counts : int {
    NumKindBits = 2,
  };
  PointerIntPair<const InMemoryIndexValueT *, NumKindBits, Kind> IndexAndKind;
  static_assert((1U << NumKindBits) <= alignof(InMemoryIndexValueT),
                "Kind will clobber pointer");
  static_assert(((int)Kind::Max >> NumKindBits) == 0, "Kind will be truncated");

public:
  ArrayRef<char> getData() const;

  ArrayRef<const InMemoryObject *> getRefs() const;
};

class InMemoryRefObject final : public InMemoryObject {
public:
  static constexpr Kind KindValue = Kind::RefNode;
  static bool classof(const InMemoryObject *O) {
    return O->getKind() == KindValue;
  }

  ArrayRef<const InMemoryObject *> getRefsImpl() const { return Refs; }
  ArrayRef<const InMemoryObject *> getRefs() const { return Refs; }
  ArrayRef<char> getDataImpl() const { return Data; }
  ArrayRef<char> getData() const { return Data; }

  static InMemoryRefObject &create(function_ref<void *(size_t Size)> Allocate,
                                   const InMemoryIndexValueT &I,
                                   ArrayRef<const InMemoryObject *> Refs,
                                   ArrayRef<char> Data) {
    void *Mem = Allocate(sizeof(InMemoryRefObject));
    return *new (Mem) InMemoryRefObject(I, Refs, Data);
  }

private:
  InMemoryRefObject(const InMemoryIndexValueT &I,
                    ArrayRef<const InMemoryObject *> Refs, ArrayRef<char> Data)
      : InMemoryObject(KindValue, I), Refs(Refs), Data(Data) {
    assert(isAddrAligned(Align(8), this) && "Expected 8-byte alignment");
    assert(isAddrAligned(Align(8), Data.data()) && "Expected 8-byte alignment");
    assert(*Data.end() == 0 && "Expected null-termination");
  }

  ArrayRef<const InMemoryObject *> Refs;
  ArrayRef<char> Data;
};

class InMemoryInlineObject final
    : public InMemoryObject,
      public TrailingObjects<InMemoryInlineObject, const InMemoryObject *,
                             char> {
public:
  static constexpr Kind KindValue = Kind::InlineNode;
  static bool classof(const InMemoryObject *O) {
    return O->getKind() == KindValue;
  }

  ArrayRef<const InMemoryObject *> getRefs() const { return getRefsImpl(); }
  ArrayRef<const InMemoryObject *> getRefsImpl() const {
    return ArrayRef(getTrailingObjects<const InMemoryObject *>(), NumRefs);
  }

  ArrayRef<char> getData() const { return getDataImpl(); }
  ArrayRef<char> getDataImpl() const {
    return ArrayRef(getTrailingObjects<char>(), DataSize);
  }

  static InMemoryInlineObject &
  create(function_ref<void *(size_t Size)> Allocate,
         const InMemoryIndexValueT &I, ArrayRef<const InMemoryObject *> Refs,
         ArrayRef<char> Data) {
    void *Mem = Allocate(sizeof(InMemoryInlineObject) +
                         sizeof(uintptr_t) * Refs.size() + Data.size() + 1);
    return *new (Mem) InMemoryInlineObject(I, Refs, Data);
  }

  size_t numTrailingObjects(OverloadToken<const InMemoryObject *>) const {
    return NumRefs;
  }

private:
  InMemoryInlineObject(const InMemoryIndexValueT &I,
                       ArrayRef<const InMemoryObject *> Refs,
                       ArrayRef<char> Data)
      : InMemoryObject(KindValue, I), NumRefs(Refs.size()),
        DataSize(Data.size()) {
    auto *BeginRefs = reinterpret_cast<const InMemoryObject **>(this + 1);
    llvm::copy(Refs, BeginRefs);
    auto *BeginData = reinterpret_cast<char *>(BeginRefs + NumRefs);
    llvm::copy(Data, BeginData);
    BeginData[Data.size()] = 0;
  }
  uint32_t NumRefs;
  uint32_t DataSize;
};

/// In-memory CAS database and action cache (the latter should be separated).
class InMemoryCAS : public BuiltinCAS {
public:
  Expected<ObjectRef> storeImpl(ArrayRef<uint8_t> ComputedHash,
                                ArrayRef<ObjectRef> Refs,
                                ArrayRef<char> Data) final;

  Expected<ObjectRef>
  storeFromNullTerminatedRegion(ArrayRef<uint8_t> ComputedHash,
                                sys::fs::mapped_file_region Map) override;

  CASID getID(const InMemoryIndexValueT &I) const {
    StringRef Hash = toStringRef(I.Hash);
    return CASID::create(&getContext(), Hash);
  }
  CASID getID(const InMemoryObject &O) const { return getID(O.getIndex()); }

  ObjectHandle getObjectHandle(const InMemoryObject &Node) const {
    assert(!(reinterpret_cast<uintptr_t>(&Node) & 0x1ULL));
    return makeObjectHandle(reinterpret_cast<uintptr_t>(&Node));
  }

  Expected<std::optional<ObjectHandle>> loadIfExists(ObjectRef Ref) override {
    return getObjectHandle(asInMemoryObject(Ref));
  }

  InMemoryIndexValueT &indexHash(ArrayRef<uint8_t> Hash) {
    return *Index.insertLazy(
        Hash, [](auto ValueConstructor) { ValueConstructor.emplace(nullptr); });
  }

  /// TODO: Consider callers to actually do an insert and to return a handle to
  /// the slot in the trie.
  const InMemoryObject *getInMemoryObject(CASID ID) const {
    assert(ID.getContext().getHashSchemaIdentifier() ==
               getContext().getHashSchemaIdentifier() &&
           "Expected ID from same hash schema");
    if (InMemoryIndexT::const_pointer P = Index.find(ID.getHash()))
      return P->Data;
    return nullptr;
  }

  const InMemoryObject &getInMemoryObject(ObjectHandle OH) const {
    return *reinterpret_cast<const InMemoryObject *>(
        (uintptr_t)OH.getInternalRef(*this));
  }

  const InMemoryObject &asInMemoryObject(ReferenceBase Ref) const {
    uintptr_t P = Ref.getInternalRef(*this);
    return *reinterpret_cast<const InMemoryObject *>(P);
  }
  ObjectRef toReference(const InMemoryObject &O) const {
    return makeObjectRef(reinterpret_cast<uintptr_t>(&O));
  }

  CASID getID(ObjectRef Ref) const final { return getIDImpl(Ref); }
  CASID getIDImpl(ReferenceBase Ref) const {
    return getID(asInMemoryObject(Ref));
  }

  std::optional<ObjectRef> getReference(const CASID &ID) const final {
    if (const InMemoryObject *Object = getInMemoryObject(ID))
      return toReference(*Object);
    return std::nullopt;
  }

  Expected<bool> isMaterialized(ObjectRef Ref) const final { return true; }

  ArrayRef<char> getDataConst(ObjectHandle Node) const final {
    return cast<InMemoryObject>(asInMemoryObject(Node)).getData();
  }

  InMemoryCAS() = default;

private:
  size_t getNumRefs(ObjectHandle Node) const final {
    return getInMemoryObject(Node).getRefs().size();
  }
  ObjectRef readRef(ObjectHandle Node, size_t I) const final {
    return toReference(*getInMemoryObject(Node).getRefs()[I]);
  }
  Error forEachRef(ObjectHandle Node,
                   function_ref<Error(ObjectRef)> Callback) const final;

  /// Index of referenced IDs (map: Hash -> InMemoryObject*). Mapped to nullptr
  /// as a convenient way to store hashes.
  ///
  /// - Insert nullptr on lookups.
  /// - InMemoryObject points back to here.
  InMemoryIndexT Index;

  ThreadSafeAllocator<BumpPtrAllocator> Objects;
  ThreadSafeAllocator<SpecificBumpPtrAllocator<sys::fs::mapped_file_region>>
      MemoryMaps;
};

} // end anonymous namespace

ArrayRef<char> InMemoryObject::getData() const {
  if (auto *Derived = dyn_cast<InMemoryRefObject>(this))
    return Derived->getDataImpl();
  return cast<InMemoryInlineObject>(this)->getDataImpl();
}

ArrayRef<const InMemoryObject *> InMemoryObject::getRefs() const {
  if (auto *Derived = dyn_cast<InMemoryRefObject>(this))
    return Derived->getRefsImpl();
  return cast<InMemoryInlineObject>(this)->getRefsImpl();
}

Expected<ObjectRef>
InMemoryCAS::storeFromNullTerminatedRegion(ArrayRef<uint8_t> ComputedHash,
                                           sys::fs::mapped_file_region Map) {
  // Look up the hash in the index, initializing to nullptr if it's new.
  ArrayRef<char> Data(Map.data(), Map.size());
  auto &I = indexHash(ComputedHash);

  // Load or generate.
  auto Allocator = [&](size_t Size) -> void * {
    return Objects.Allocate(Size, alignof(InMemoryObject));
  };
  auto Generator = [&]() -> const InMemoryObject * {
    return &InMemoryRefObject::create(Allocator, I, {}, Data);
  };
  const InMemoryObject &Node =
      cast<InMemoryObject>(I.Data.loadOrGenerate(Generator));

  // Save Map if the winning node uses it.
  if (auto *RefNode = dyn_cast<InMemoryRefObject>(&Node))
    if (RefNode->getData().data() == Map.data())
      new (MemoryMaps.Allocate(1)) sys::fs::mapped_file_region(std::move(Map));

  return toReference(Node);
}

Expected<ObjectRef> InMemoryCAS::storeImpl(ArrayRef<uint8_t> ComputedHash,
                                           ArrayRef<ObjectRef> Refs,
                                           ArrayRef<char> Data) {
  // Look up the hash in the index, initializing to nullptr if it's new.
  auto &I = indexHash(ComputedHash);

  // Create the node.
  SmallVector<const InMemoryObject *> InternalRefs;
  for (ObjectRef Ref : Refs)
    InternalRefs.push_back(&asInMemoryObject(Ref));
  auto Allocator = [&](size_t Size) -> void * {
    return Objects.Allocate(Size, alignof(InMemoryObject));
  };
  auto Generator = [&]() -> const InMemoryObject * {
    return &InMemoryInlineObject::create(Allocator, I, InternalRefs, Data);
  };
  return toReference(cast<InMemoryObject>(I.Data.loadOrGenerate(Generator)));
}

Error InMemoryCAS::forEachRef(ObjectHandle Handle,
                              function_ref<Error(ObjectRef)> Callback) const {
  auto &Node = getInMemoryObject(Handle);
  for (const InMemoryObject *Ref : Node.getRefs())
    if (Error E = Callback(toReference(*Ref)))
      return E;
  return Error::success();
}

std::unique_ptr<ObjectStore> cas::createInMemoryCAS() {
  return std::make_unique<InMemoryCAS>();
}
