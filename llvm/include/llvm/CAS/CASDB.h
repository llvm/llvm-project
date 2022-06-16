//===- llvm/CAS/CASDB.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASDB_H
#define LLVM_CAS_CASDB_H

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

class CASDB;

/// Kind of CAS object.
///
/// FIXME: Remove.
enum class ObjectKind {
  /// A node, with data and zero or more references.
  Node,

  /// Filesystem tree, with named references and entry types.
  ///
  /// FIXME: Move into a Filesystem schema.
  Tree,
};

class LeafNodeProxy;
class NodeProxy;
class TreeProxy;

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
///       by the same or a different CAS instance with \a CASDB::parseID().
/// - An object can be looked up by content or by UID.
///     - \a storeNode() and \a storeTree() are "get-or-create"
///       methods, writing an object if it doesn't exist yet, and return a
///       handle to it in any case.
///     - \a loadObject(const CASID&) looks up an object by its UID.
/// - Objects can reference other objects, forming an arbitrary DAG.
///
/// There are currently three kinds of objects.
///
/// - ObjectKind::Node: Contains 0+ bytes of data and a list of 0+ \a
///   ObjectRef. A \a ObjectRef points at another CAS object. Created with \a
///   storeNode().
/// - ObjectKind::Tree: Sorted list of 0+ \a NamedTreeEntry, which associates a
///   name, another CAS object, and a \a TreeEntry::EntryKind. Designed to
///   represent a filesystem in the CAS. Created with \a storeTree().
///
/// The \a CASDB interface has a few ways of referencing objects:
///
/// - \a ObjectRef encapsulates a reference to something in the CAS. If you
///   have an ObjectRef, you know the object exists, but you don't know
///   anything about it. "Loading" the object is a separate step that may
///   not have happened yet, and which can fail (due to filesystem corruption)
///   or introduce latency (if downloading from a remote store).
/// - \a ObjectHandle encapulates a *loaded* object in the CAS. You need one of
///   these to inspect the content of an object: to look at its stored
///   data and references. In practice, right now you want a subclass:
///     - \a NodeHandle: a handle for a node. Returned by \a
///       storeNode() and the variant accessors.
///     - \a TreeHandle: a handle for a tree. Returned by \a
///       storeTree() and the variant accessors.
///     - \a AnyObjectHandle: a variant between \a ObjectHandle and its
///       non-variant subclasses. Returned by \a loadObject().
/// - \a CASID: the UID for an object in the CAS, obtained through \a
///   CASDB::getObjectID() or \a CASDB::parseID(). This is a valid CAS
///   identifier, but may reference an object that is unknown to this CAS
///   instance.
///
/// There are a few options for accessing content of objects, with different
/// lifetime tradeoffs:
///
/// - \a readData() accesses data without exposing lifetime at all.
/// - \a loadIndependentDataBuffer() returns a \a MemoryBuffer whose lifetime
///   is independent of the CAS (it can live longer).
/// - \a getDataString() and \a getDataArray() return StringRef/ArrayRef with
///   lifetime is guaranteed to last as long as \a CASDB.
/// - \a readRef() and \a forEachRef() iterate through the references in a
///   node. There is no lifetime assumption.
/// - \a loadTreeEntry(), \a lookupTreeEntry(), and \a forEachTreeEntry()
///   iterate through the entries in a tree. The names are assumed to have the
///   same lifetime as \a getDataString() would give.
///
/// Both ObjectRef and ObjectHandle are lightweight, wrapping a `uint64_t`.
/// Doing anything with them requires a CASDB. As a convenience:
///
/// - ObjectProxy (currently \a NodeProxy and \a TreeProxy) pairs
///   an ObjectHandle (subclass) with a CASDB, and wraps access APIs to avoid
///   having to pass extra parameters.
///
/// TODO: Remove Tree, moving these concepts to a filesystem schema
/// that sits on top of the CAS.
///
/// TODO: Remove CASID.
///
/// Here's how to remove CASID and Tree:
///
/// - Lift trees into a filesystem schema, dropping TreeHandle and making
///   TreeProxy inherit from NodeProxy.
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
///   gains without adding complexity to \a CASDB. E.g., \a NodeBuilder could
///   have an API to add a named field to a node under construction; if the
///   name is small enough, it's stored locally in the node's own data, but if
///   it's bigger then it's outlined to a separate CAS object. \a NodeReader
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
class CASDB : public CASIDContext {
  void anchor() override;

public:
  /// Get a \p CASID from a \p ID, which should have been generated by \a
  /// CASID::print(). This succeeds as long as \a validateID() would pass. The
  /// object may be unknown to this CAS instance.
  ///
  /// TODO: Remove, and update callers to use \a validateID() or \a
  /// extractHashFromID().
  virtual Expected<CASID> parseID(StringRef ID) = 0;

  /// FIXME: Remove these.
  Expected<LeafNodeProxy> createBlob(StringRef Data);
  Expected<TreeProxy> createTree(ArrayRef<NamedTreeEntry> Entries = None);
  Expected<NodeProxy> createNode(ArrayRef<CASID> References, StringRef Data);

  virtual Expected<TreeHandle> storeTree(ArrayRef<NamedTreeEntry> Entries) = 0;
  virtual Expected<NodeHandle> storeNode(ArrayRef<ObjectRef> Refs,
                                         ArrayRef<char> Data) = 0;

  Expected<NodeHandle> storeNodeFromString(ArrayRef<ObjectRef> Refs,
                                           StringRef String) {
    return storeNode(Refs, arrayRefFromStringRef<char>(String));
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
  Expected<NodeHandle>
  storeNodeFromOpenFile(sys::fs::file_t FD,
                        Optional<sys::fs::file_status> Status = None) {
    return storeNodeFromOpenFileImpl(FD, Status);
  }

protected:
  virtual Expected<NodeHandle>
  storeNodeFromOpenFileImpl(sys::fs::file_t FD,
                            Optional<sys::fs::file_status> Status);

  /// Allow CASDB implementations to create internal handles.
#define MAKE_CAS_HANDLE_CONSTRUCTOR(HandleKind)                                \
  HandleKind make##HandleKind(uint64_t InternalRef) const {                    \
    return HandleKind(*this, InternalRef);                                     \
  }
  MAKE_CAS_HANDLE_CONSTRUCTOR(NodeHandle)
  MAKE_CAS_HANDLE_CONSTRUCTOR(TreeHandle)
  MAKE_CAS_HANDLE_CONSTRUCTOR(ObjectRef)
#undef MAKE_CAS_HANDLE_CONSTRUCTOR

public:
  /// Get an ID for \p Ref.
  virtual CASID getObjectID(ObjectRef Ref) const = 0;
  virtual CASID getObjectID(ObjectHandle Handle) const = 0;

  /// Get a reference to the object called \p ID.
  ///
  /// Returns \c None if not stored in this CAS.
  virtual Optional<ObjectRef> getReference(const CASID &ID) const = 0;

  virtual ObjectRef getReference(ObjectHandle Handle) const = 0;

  /// Load the object referenced by \p Ref.
  ///
  /// Errors if the object cannot be loaded.
  virtual Expected<AnyObjectHandle> loadObject(ObjectRef Ref) = 0;

  /// Load the object called \p ID.
  ///
  /// Returns \c None if it's unknown in this CAS instance.
  ///
  /// Errors if the object cannot be loaded.
  Expected<Optional<AnyObjectHandle>> loadObject(const CASID &ID);

  static Error createUnknownObjectError(CASID ID);
  static Error createWrongKindError(CASID ID);

  template <class ProxyT, class HandleT>
  Expected<ProxyT> loadObjectProxy(CASID ID);

  template <class ProxyT, class HandleT>
  Expected<ProxyT> loadObjectProxy(Expected<HandleT> H);

  virtual Error validateObject(const CASID &ID) = 0;

public:
  /// FIXME: Delete these. Update callers to call \a loadObject() and create
  /// the proxy themselves.
  Expected<LeafNodeProxy> getBlob(CASID ID);
  Expected<TreeProxy> getTree(CASID ID);
  Expected<NodeProxy> getNode(CASID ID);

  /// FIXME: Move to the Filesystem schema once trees are removed from CASDB.
  Expected<TreeProxy> loadTree(ObjectRef Ref);
  Expected<LeafNodeProxy> loadBlob(ObjectRef Ref);
  Expected<NodeProxy> loadNode(ObjectRef Ref);

  /// Get the size of some data.
  virtual uint64_t getDataSize(NodeHandle Node) const = 0;

  /// Read the data from \p Data into \p OS.
  uint64_t readData(NodeHandle Node, raw_ostream &OS, uint64_t Offset = 0,
                    uint64_t MaxBytes = -1ULL) const {
    return readDataImpl(Node, OS, Offset, MaxBytes);
  }

protected:
  virtual uint64_t readDataImpl(NodeHandle Node, raw_ostream &OS,
                                uint64_t Offset, uint64_t MaxBytes) const = 0;

public:
  /// Get a lifetime-extended StringRef pointing at \p Data.
  ///
  /// Depending on the CAS implementation, this may involve in-memory storage
  /// overhead.
  StringRef getDataString(NodeHandle Node, bool NullTerminate = true) {
    return toStringRef(getDataImpl(Node, NullTerminate));
  }

  /// Get a lifetime-extended ArrayRef pointing at \p Data.
  ///
  /// Depending on the CAS implementation, this may involve in-memory storage
  /// overhead.
  template <class CharT = char>
  ArrayRef<CharT> getDataArray(NodeHandle Node, bool NullTerminate = true) {
    static_assert(std::is_same<CharT, char>::value ||
                      std::is_same<CharT, unsigned char>::value ||
                      std::is_same<CharT, signed char>::value,
                  "Expected byte type");
    ArrayRef<char> S = getDataImpl(Node, NullTerminate);
    return makeArrayRef(reinterpret_cast<const CharT *>(S.data()), S.size());
  }

protected:
  virtual ArrayRef<char> getDataImpl(NodeHandle Node, bool NullTerminate) = 0;

public:
  /// Get a MemoryBuffer with the contents of \p Data whose lifetime is
  /// independent of this CAS instance.
  Expected<std::unique_ptr<MemoryBuffer>>
  loadIndependentDataBuffer(NodeHandle Node, const Twine &Name = "",
                            bool NullTerminate = true) const;

protected:
  virtual Expected<std::unique_ptr<MemoryBuffer>>
  loadIndependentDataBufferImpl(NodeHandle Node, const Twine &Name,
                                bool NullTerminate) const;

public:
  virtual Error forEachRef(NodeHandle Node,
                           function_ref<Error(ObjectRef)> Callback) const = 0;
  virtual void readRefs(NodeHandle Node,
                        SmallVectorImpl<ObjectRef> &Refs) const;
  virtual ObjectRef readRef(NodeHandle Node, size_t I) const = 0;
  virtual size_t getNumRefs(NodeHandle Node) const = 0;

  virtual Error forEachTreeEntry(
      TreeHandle Tree,
      function_ref<Error(const NamedTreeEntry &)> Callback) const = 0;
  virtual NamedTreeEntry loadTreeEntry(TreeHandle Tree, size_t I) const = 0;
  virtual Optional<size_t> lookupTreeEntry(TreeHandle Tree,
                                           StringRef Name) const = 0;
  virtual size_t getNumTreeEntries(TreeHandle Tree) const = 0;

  virtual Expected<CASID> getCachedResult(CASID InputID) = 0;
  virtual Error putCachedResult(CASID InputID, CASID OutputID) = 0;

  virtual void print(raw_ostream &) const {}
  void dump() const;

  virtual ~CASDB() = default;
};

template <class HandleT> class ProxyBase : public HandleT {
public:
  const CASDB &getCAS() const { return *CAS; }
  CASID getID() const {
    return CAS->getObjectID(*static_cast<const ObjectHandle *>(this));
  }
  ObjectRef getRef() const {
    return CAS->getReference(*static_cast<const ObjectHandle *>(this));
  }

  /// FIXME: Remove this.
  operator CASID() const { return getID(); }

  friend bool operator==(const ProxyBase &Proxy, ObjectRef Ref) {
    return Proxy.CAS->getReference(
               *static_cast<const ObjectHandle *>(&Proxy)) == Ref;
  }
  friend bool operator==(ObjectRef Ref, const ProxyBase &Proxy) {
    return Proxy == Ref;
  }
  friend bool operator!=(const ProxyBase &Proxy, ObjectRef Ref) {
    return !(Proxy == Ref);
  }
  friend bool operator!=(ObjectRef Ref, const ProxyBase &Proxy) {
    return !(Proxy == Ref);
  }

protected:
  ProxyBase(const CASDB &CAS, HandleT H) : HandleT(H), CAS(&CAS) {}
  const CASDB *CAS;
};

/// Proxy of a tree CAS object. Reference is passed by value and is
/// expected to be valid as long as the \a CASDB is.
///
/// TODO: Add an API to expose a range of NamedTreeEntry.
///
/// FIXME: Turn into a reader API.
class TreeProxy : public ProxyBase<TreeHandle> {
public:
  bool empty() const { return NumEntries == 0; }
  size_t size() const { return NumEntries; }

  Optional<NamedTreeEntry> lookup(StringRef Name) const {
    if (Optional<size_t> I = getCAS().lookupTreeEntry(
            *static_cast<const TreeHandle *>(this), Name))
      return get(*I);
    return None;
  }

  NamedTreeEntry get(size_t I) const {
    return getCAS().loadTreeEntry(*static_cast<const TreeHandle *>(this), I);
  }

  /// Visit each tree entry in order, returning an error from \p Callback to
  /// stop early.
  Error
  forEachEntry(function_ref<Error(const NamedTreeEntry &)> Callback) const {
    return getCAS().forEachTreeEntry(*static_cast<const TreeHandle *>(this),
                                     Callback);
  }

  TreeProxy() = delete;

  static TreeProxy load(CASDB &CAS, TreeHandle Tree) {
    return TreeProxy(CAS, Tree, CAS.getNumTreeEntries(Tree));
  }

private:
  TreeProxy(const CASDB &CAS, TreeHandle H, size_t NumEntries)
      : ProxyBase::ProxyBase(CAS, H), NumEntries(NumEntries) {}

  size_t NumEntries;
};

/// Reference to an abstract hierarchical node, with data and references.
/// Reference is passed by value and is expected to be valid as long as the \a
/// CASDB is.
///
/// TODO: Expose \a CASDB::readData() and only call \a CASDB::getDataString()
/// when asked.
class NodeProxy : public ProxyBase<NodeHandle> {
public:
  size_t getNumReferences() const { return NumReferences; }
  ObjectRef getReference(size_t I) const {
    return getCAS().readRef(*static_cast<const NodeHandle *>(this), I);
  }

  // FIXME: Remove this.
  CASID getReferenceID(size_t I) const {
    Optional<CASID> ID = getCAS().getObjectID(getReference(I));
    assert(ID && "Expected reference to be first-class object");
    return *ID;
  }

  /// Visit each reference in order, returning an error from \p Callback to
  /// stop early.
  Error forEachReference(function_ref<Error(ObjectRef)> Callback) const {
    return getCAS().forEachRef(*static_cast<const NodeHandle *>(this),
                               Callback);
  }
  Error forEachReferenceID(function_ref<Error(CASID)> Callback) const {
    return getCAS().forEachRef(
        *static_cast<const NodeHandle *>(this), [&](ObjectRef Ref) {
          Optional<CASID> ID = getCAS().getObjectID(Ref);
          assert(ID && "Expected reference to be first-class object");
          return Callback(*ID);
        });
  }

  /// Get the content of the node. Valid as long as the CAS is valid.
  StringRef getData() const { return Data; }

protected:
  /// FIXME: Remove once LeafNodeProxy doesn't need this.
  const StringRef *getDataPtr() const { return &Data; }

public:
  NodeProxy() = delete;

  static NodeProxy load(CASDB &CAS, NodeHandle Node) {
    return NodeProxy(CAS, Node, CAS.getNumRefs(Node), CAS.getDataString(Node));
  }

private:
  NodeProxy(const CASDB &CAS, NodeHandle H, size_t NumReferences,
            StringRef Data)
      : ProxyBase::ProxyBase(CAS, H), NumReferences(NumReferences), Data(Data) {
  }

  size_t NumReferences;
  StringRef Data;
};

/// Proxy for a leaf node.
class LeafNodeProxy : public NodeProxy {
public:
  /// FIXME: Remove this after updating clients.
  StringRef operator*() const { return getData(); }

  /// FIXME: Remove this after updating clients.
  const StringRef *operator->() const { return getDataPtr(); }

  size_t getNumReferences() const = delete;
  ObjectRef getReference(size_t I) const = delete;
  CASID getReferenceID(size_t I) const = delete;
  Error forEachReference(function_ref<Error(ObjectRef)> Callback) const = delete;
  Error forEachReferenceID(function_ref<Error(CASID)> Callback) const = delete;

  explicit LeafNodeProxy(NodeProxy N) : NodeProxy(std::move(N)) {
    assert(this->NodeProxy::getNumReferences() == 0);
  }
};

/// FIXME: Remove this after updating callers.
using BlobProxy = LeafNodeProxy;

Expected<std::unique_ptr<CASDB>>
createPluginCAS(StringRef PluginPath, ArrayRef<std::string> PluginArgs = None);
std::unique_ptr<CASDB> createInMemoryCAS();

/// Gets or creates a persistent on-disk path at \p Path.
///
/// Deprecated: if \p Path resolves to \a getDefaultOnDiskCASStableID(),
/// automatically opens \a getDefaultOnDiskCASPath() instead.
///
/// FIXME: Remove the special behaviour for getDefaultOnDiskCASStableID(). The
/// client should handle this logic, if/when desired.
Expected<std::unique_ptr<CASDB>> createOnDiskCAS(const Twine &Path);

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

#endif // LLVM_CAS_CASDB_H
