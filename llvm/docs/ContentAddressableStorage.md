# Content Addressable Storage

## Introduction to CAS

Content Addressable Storage, or `CAS`, is a storage system where it assigns
unique addresses to the data stored. It is very useful for data deduplicaton
and creating unique identifiers.

Unlikely other kind of storage system like file system, CAS is immutable. It
is more reliable to model a computation when representing the inputs and outputs
of the computation using objects stored in CAS.

The basic unit of the CAS library is a CASObject, where it contains:

* Data: arbitrary data
* References: references to other CASObject

It can be conceptually modeled as something like:

```
struct CASObject {
  ArrayRef<char> Data;
  ArrayRef<CASObject*> Refs;
}
```

Such abstraction can allow simple composition of CASObjects into a DAG to
represent complicated data structure while still allowing data deduplication.
Note you can compare two DAGs by just comparing the CASObject hash of two
root nodes.



## LLVM CAS Library User Guide

The CAS-like storage provided in LLVM is `llvm::cas::ObjectStore`.
To reference a CASObject, there are few different abstractions provided
with different trade-offs:

### ObjectRef

`ObjectRef` is a lightweight reference to a CASObject stored in the CAS.
This is the most commonly used abstraction and it is cheap to copy/pass
along. It has following properties:

* `ObjectRef` is only meaningful within the `ObjectStore` that created the ref.
`ObjectRef` created by different `ObjectStore` cannot be cross-referenced or
compared.
* `ObjectRef` doesn't guarantee the existence of the CASObject it points to. An
explicitly load is required before accessing the data stored in CASObject.
This load can also fail, for reasons like but not limited to: object does
not exist, corrupted CAS storage, operation timeout, etc.
* If two `ObjectRef` are equal, it is guarantee that the object they point to
(if exists) are identical. If they are not equal, the underlying objects are
guaranteed to be not the same.

### ObjectProxy

`ObjectProxy` represents a loaded CASObject. With an `ObjectProxy`, the
underlying stored data and references can be accessed without the need
of error handling. The class APIs also provide convenient methods to
access underlying data. The lifetime of the underlying data is equal to
the lifetime of the instance of `ObjectStore` unless explicitly copied.

### CASID

`CASID` is the hash identifier for CASObjects. It owns the underlying
storage for hash value so it can be expensive to copy and compare depending
on the hash algorithm. `CASID` is generally only useful in rare situations
like printing raw hash value or exchanging hash values between different
CAS instances with the same hashing schema.

### ObjectStore

`ObjectStore` is the CAS-like object storage. It provides API to save
and load CASObjects, for example:

```
ObjectRef A, B, C;
Expected<ObjectRef> Stored = ObjectStore.store("data", {A, B});
Expected<ObjectProxy> Loaded = ObjectStore.getProxy(C);
```

It also provides APIs to convert between `ObjectRef`, `ObjectProxy` and
`CASID`.



## CAS Library Implementation Guide

The LLVM ObjectStore APIs are designed so that it is easy to add
customized CAS implementation that are interchangeable with builtin
CAS implementations.

To add your own implementation, you just need to add a subclass to
`llvm::cas::ObjectStore` and implement all its pure virtual methods.
To be interchangeable with LLVM ObjectStore, the new CAS implementation
needs to conform to following contracts:

* Different CASObject stored in the ObjectStore needs to have a different hash
and result in a different `ObjectRef`. Vice versa, same CASObject should have
same hash and same `ObjectRef`. Note two different CASObjects with identical
data but different references are considered different objects.
* `ObjectRef`s are comparable within the same `ObjectStore` instance, and can
be used to determine the equality of the underlying CASObjects.
* The loaded objects from the ObjectStore need to have the lifetime to be at
least as long as the ObjectStore itself.

If not specified, the behavior can be implementation defined. For example,
`ObjectRef` can be used to point to a loaded CASObject so
`ObjectStore` never fails to load. It is also legal to use a stricter model
than required. For example, an `ObjectRef` that can be used to compare
objects between different `ObjectStore` instances is legal but user
of the ObjectStore should not depend on this behavior.

For CAS library implementer, there is also a `ObjectHandle` class that
is an internal representation of a loaded CASObject reference.
`ObjectProxy` is just a pair of `ObjectHandle` and `ObjectStore`, because
just like `ObjectRef`, `ObjectHandle` is only useful when paired with
the ObjectStore that knows about the loaded CASObject.
