//===- llvm/CAS/ObjectStore.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_OBJECTSTORE_H
#define LLVM_CAS_OBJECTSTORE_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/CAS/TreeEntry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h" // FIXME: Split out sys::fs::file_status.
#include <cstddef>

namespace llvm {

class MemoryBuffer;

namespace cas {

class ObjectStore;

class ObjectProxy;

/// Content-addressable storage for objects.
///
/// Conceptually, objects are stored in a "unique set".
///
/// - Objects are immutable ("value objects") that are defined by their
///   content. They are implicitly deduplicated by content.
/// - Each object has a unique identifier (UID) that's derived from its content,
///   called a \a CASID.
///     - This UID is a fixed-size (strong) hash of the transitive content of a
///       CAS object.
///     - It's comparable between any two CAS instances that have the same \a
///       CASIDContext::getHashSchemaIdentifier().
///     - The UID can be printed (e.g., \a CASID::toString()) and it can parsed
///       by the same or a different CAS instance with \a
///       ObjectStore::parseID().
/// - An object can be looked up by content or by UID.
///     - \a store() is "get-or-create"  methods, writing an object if it
///       doesn't exist yet, and return a ref to it in any case.
///     - \a loadObject(const CASID&) looks up an object by its UID.
/// - Objects can reference other objects, forming an arbitrary DAG.
///
/// The \a ObjectStore interface has a few ways of referencing objects:
///
/// - \a ObjectRef encapsulates a reference to something in the CAS. It is an
///   opaque type that references an object inside a specific CAS. It is
///   implementation defined if the underlying object exists or not for an
///   ObjectRef, and it can used to speed up CAS lookup as an implementation
///   detail. However, you don't know anything about the underlying objects.
///   "Loading" the object is a separate step that may not have happened
///   yet, and which can fail (e.g. due to filesystem corruption) or introduce
///   latency (if downloading from a remote store).
/// - \a ObjectHandle encapulates a *loaded* object in the CAS. You need one of
///   these to inspect the content of an object: to look at its stored
///   data and references. This is internal to CAS implementation and not
///   availble from CAS public APIs.
/// - \a CASID: the UID for an object in the CAS, obtained through \a
///   ObjectStore::getID() or \a ObjectStore::parseID(). This is a valid CAS
///   identifier, but may reference an object that is unknown to this CAS
///   instance.
/// - \a ObjectProxy pairs an ObjectHandle (subclass) with a ObjectStore, and
///   wraps access APIs to avoid having to pass extra parameters. It is the
///   object used for accessing underlying data and refs by CAS users.
///
/// There are a few options for accessing content of objects, with different
/// lifetime tradeoffs:
///
/// - \a getData() accesses data without exposing lifetime at all.
/// - \a loadIndependentDataBuffer() returns a \a MemoryBuffer whose lifetime
///   is independent of the CAS (it can live longer).
/// - \a getDataString() return StringRef with lifetime is guaranteed to last as
///   long as \a ObjectStore.
/// - \a readRef() and \a forEachRef() iterate through the references in an
///   object. There is no lifetime assumption.
///
/// Both ObjectRef and ObjectHandle are lightweight, wrapping a `uint64_t`.
/// Doing anything with them requires a ObjectStore. As a convenience:
///
///
/// TODO: Remove CASID.
///
/// Here's how to remove CASID:
///
/// - Add APIs for bypassing CASID when parsing:
///     - Validate an ID without doing anything else (current check done by
///       `parseID()`).
///     - Get the hash for an object or StringRef-based ID.
///     - Get an ObjectRef or load an ObjectHandle from a StringRef-based ID.
/// - Update existing code using CASID to use the new ObjectRef,
///   ObjectHandle, and StringRef APIs.
/// - Remove CASID, changing `getObjectID()` to return `std::string`.
///
/// TODO: Consider optimizing small and/or string-like leaf objects:
///
/// - \a NodeBuilder and \a NodeReader interfaces can bring some of the same
///   gains without adding complexity to \a ObjectStore. E.g., \a NodeBuilder
///   could have an API to add a named field to a node under construction; if
///   the name is small enough, it's stored locally in the node's own data, but
///   if it's bigger then it's outlined to a separate CAS object. \a NodeReader
///   could handle the complications of reading.
/// - Implementations can do fast lookups of small objects by adding a
///   content-based index for them (prefix tree / suffix tree of content),
///   amortizing overhead of hash computation in \a storeNode().
/// - Implementations could remove small leaf objects from the main index,
///   indexing them separately with a partial hash (e.g., 4B prefix), to
///   optimize storage overhead (32B hash is big for small objects!). Lookups
///   by UID that miss the main index would get more expensive, requiring a
///   hash computation for each small object with a matching partial hash, but
///   maybe this would be rare. To mitigate this cost, small leaf objects could
///   get added to the main index lazily on first lookup-by-UID, lazily adding
///   the full overhead of the hash storage only when used by clients.
/// - NOTE: we tried adding an API to store "raw data" that can be optimized,
///   but it was very complicated to reason about.
///     - Introduced many opportunities for implementation bugs.
///     - Introduced many complications in the API.
///
/// FIXME: Split out ActionCache as a separate concept, and rename this
/// ObjectStore.
class ObjectStore {
  friend class ObjectProxy;
  void anchor();

public:
  /// Get a \p CASID from a \p ID, which should have been generated by \a
  /// CASID::print(). This succeeds as long as \a validateID() would pass. The
  /// object may be unknown to this CAS instance.
  ///
  /// TODO: Remove, and update callers to use \a validateID() or \a
  /// extractHashFromID().
  virtual Expected<CASID> parseID(StringRef ID) = 0;

  /// Store object into ObjectStore.
  virtual Expected<ObjectRef> store(ArrayRef<ObjectRef> Refs,
                                    ArrayRef<char> Data) = 0;
  /// Get an ID for \p Ref.
  virtual CASID getID(ObjectRef Ref) const = 0;
  /// Get an ID for \p Handle.
  virtual CASID getID(ObjectHandle Handle) const = 0;

  /// Get a reference to the object called \p ID.
  ///
  /// Returns \c None if not stored in this CAS.
  virtual Optional<ObjectRef> getReference(const CASID &ID) const = 0;

  /// Validate the underlying object referred by CASID.
  virtual Error validate(const CASID &ID) = 0;

protected:
  /// Get a Ref from Handle.
  virtual ObjectRef getReference(ObjectHandle Handle) const = 0;

  /// Load the object referenced by \p Ref.
  ///
  /// Errors if the object cannot be loaded.
  virtual Expected<ObjectHandle> load(ObjectRef Ref) = 0;

  /// Get the size of some data.
  virtual uint64_t getDataSize(ObjectHandle Node) const = 0;

  /// Methods for handling objects.
  virtual Error forEachRef(ObjectHandle Node,
                           function_ref<Error(ObjectRef)> Callback) const = 0;
  virtual ObjectRef readRef(ObjectHandle Node, size_t I) const = 0;
  virtual size_t getNumRefs(ObjectHandle Node) const = 0;
  virtual ArrayRef<char> getData(ObjectHandle Node,
                                 bool RequiresNullTerminator = false) const = 0;

  /// Get ObjectRef from open file.
  virtual Expected<ObjectRef>
  storeFromOpenFileImpl(sys::fs::file_t FD,
                        Optional<sys::fs::file_status> Status) = 0;

  /// Get a lifetime-extended StringRef pointing at \p Data.
  ///
  /// Depending on the CAS implementation, this may involve in-memory storage
  /// overhead.
  StringRef getDataString(ObjectHandle Node) {
    return toStringRef(getData(Node));
  }

  /// Get a lifetime-extended MemoryBuffer pointing at \p Data.
  ///
  /// Depending on the CAS implementation, this may involve in-memory storage
  /// overhead.
  std::unique_ptr<MemoryBuffer>
  getMemoryBuffer(ObjectHandle Node, StringRef Name = "",
                  bool RequiresNullTerminator = true);

  /// Read all the refs from object in a SmallVector.
  virtual void readRefs(ObjectHandle Node,
                        SmallVectorImpl<ObjectRef> &Refs) const;

  Expected<ObjectProxy> getProxy(Expected<ObjectHandle> Ref);

  /// Allow ObjectStore implementations to create internal handles.
#define MAKE_CAS_HANDLE_CONSTRUCTOR(HandleKind)                                \
  HandleKind make##HandleKind(uint64_t InternalRef) const {                    \
    return HandleKind(*this, InternalRef);                                     \
  }
  MAKE_CAS_HANDLE_CONSTRUCTOR(ObjectHandle)
  MAKE_CAS_HANDLE_CONSTRUCTOR(ObjectRef)
#undef MAKE_CAS_HANDLE_CONSTRUCTOR

public:
  /// Helper functions to store object and returns a ObjectProxy.
  Expected<ObjectProxy> createProxy(ArrayRef<ObjectRef> Refs, StringRef Data);

  /// Store object from StringRef.
  Expected<ObjectRef> storeFromString(ArrayRef<ObjectRef> Refs,
                                      StringRef String) {
    return store(Refs, arrayRefFromStringRef<char>(String));
  }

  /// Default implementation reads \p FD and calls \a storeNode(). Does not
  /// take ownership of \p FD; the caller is responsible for closing it.
  ///
  /// If \p Status is sent in it is to be treated as a hint. Implementations
  /// must protect against the file size potentially growing after the status
  /// was taken (i.e., they cannot assume that an mmap will be null-terminated
  /// where \p Status implies).
  ///
  /// Returns the \a CASID and the size of the file.
  Expected<ObjectRef>
  storeFromOpenFile(sys::fs::file_t FD,
                    Optional<sys::fs::file_status> Status = None) {
    return storeFromOpenFileImpl(FD, Status);
  }

  static Error createUnknownObjectError(const CASID &ID);

  /// Create ObjectProxy from CASID. If the object doesn't exit, get an error.
  Expected<ObjectProxy> getProxy(const CASID &ID);
  /// Create ObjectProxy from CASID. If the object doesn't exit, get None..
  Expected<Optional<ObjectProxy>> getProxyOrNone(const CASID &ID);
  /// Create ObjectProxy from ObjectRef. If the object can't be loaded, get an
  /// error.
  Expected<ObjectProxy> getProxy(ObjectRef Ref);

  /// Read the data from \p Data into \p OS.
  uint64_t readData(ObjectHandle Node, raw_ostream &OS, uint64_t Offset = 0,
                    uint64_t MaxBytes = -1ULL) const {
    ArrayRef<char> Data = getData(Node);
    assert(Offset < Data.size() && "Expected valid offset");
    Data = Data.drop_front(Offset).take_front(MaxBytes);
    OS << toStringRef(Data);
    return Data.size();
  }

  /// Get a MemoryBuffer with the contents of \p Data whose lifetime is
  /// independent of this CAS instance.
  virtual Expected<std::unique_ptr<MemoryBuffer>>
  loadIndependentDataBuffer(ObjectHandle Node, const Twine &Name = "",
                            bool NullTerminate = true) const;

  /// Print the ObjectStore internals for debugging purpose.
  virtual void print(raw_ostream &) const {}
  void dump() const;

  /// Get CASContext
  const CASContext &getContext() const { return Context; }

  virtual ~ObjectStore() = default;

protected:
  ObjectStore(const CASContext &Context) : Context(Context) {}

private:
  const CASContext &Context;
};

/// Reference to an abstract hierarchical node, with data and references.
/// Reference is passed by value and is expected to be valid as long as the \a
/// ObjectStore is.
///
/// TODO: Expose \a ObjectStore::readData() and only call \a
/// ObjectStore::getDataString() when asked.
class ObjectProxy {
public:
  const ObjectStore &getCAS() const { return *CAS; }
  ObjectStore &getCAS() { return *CAS; }
  CASID getID() const { return CAS->getID(H); }
  ObjectRef getRef() const { return CAS->getReference(H); }
  size_t getNumReferences() const { return CAS->getNumRefs(H); }
  ObjectRef getReference(size_t I) const { return CAS->readRef(H, I); }

  // FIXME: Remove this.
  operator CASID() const { return getID(); }
  CASID getReferenceID(size_t I) const {
    Optional<CASID> ID = getCAS().getID(getReference(I));
    assert(ID && "Expected reference to be first-class object");
    return *ID;
  }

  /// Visit each reference in order, returning an error from \p Callback to
  /// stop early.
  Error forEachReference(function_ref<Error(ObjectRef)> Callback) const {
    return CAS->forEachRef(H, Callback);
  }

  std::unique_ptr<MemoryBuffer>
  getMemoryBuffer(StringRef Name = "",
                  bool RequiresNullTerminator = true) const;

  /// Get the content of the node. Valid as long as the CAS is valid.
  StringRef getData() const { return CAS->getDataString(H); }

  friend bool operator==(const ObjectProxy &Proxy, ObjectRef Ref) {
    return Proxy.getRef() == Ref;
  }
  friend bool operator==(ObjectRef Ref, const ObjectProxy &Proxy) {
    return Proxy.getRef() == Ref;
  }
  friend bool operator!=(const ObjectProxy &Proxy, ObjectRef Ref) {
    return !(Proxy.getRef() == Ref);
  }
  friend bool operator!=(ObjectRef Ref, const ObjectProxy &Proxy) {
    return !(Proxy.getRef() == Ref);
  }

public:
  ObjectProxy() = delete;

  static ObjectProxy load(ObjectStore &CAS, ObjectHandle Node) {
    return ObjectProxy(CAS, Node);
  }

private:
  ObjectProxy(ObjectStore &CAS, ObjectHandle H) : CAS(&CAS), H(H) {}

  ObjectStore *CAS;
  ObjectHandle H;
};



std::unique_ptr<ObjectStore> createInMemoryCAS();

/// Gets or creates a persistent on-disk path at \p Path.
///
/// Deprecated: if \p Path resolves to \a getDefaultOnDiskCASStableID(),
/// automatically opens \a getDefaultOnDiskCASPath() instead.
///
/// FIXME: Remove the special behaviour for getDefaultOnDiskCASStableID(). The
/// client should handle this logic, if/when desired.
Expected<std::unique_ptr<ObjectStore>> createOnDiskCAS(const Twine &Path);

/// Set \p Path to a reasonable default on-disk path for a persistent CAS for
/// the current user.
void getDefaultOnDiskCASPath(SmallVectorImpl<char> &Path);

/// Get a reasonable default on-disk path for a persistent CAS for the current
/// user.
std::string getDefaultOnDiskCASPath();

/// FIXME: Remove.
void getDefaultOnDiskCASStableID(SmallVectorImpl<char> &Path);

/// FIXME: Remove.
std::string getDefaultOnDiskCASStableID();

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_OBJECTSTORE_H
