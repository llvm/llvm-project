# Content Addressable Storage

## Introduction to CAS

Content Addressable Storage, or `CAS`, is a storage system that assigns
unique addresses to the data stored. It is very useful for data deduplicaton
and creating unique identifiers.

Unlike other kinds of storage systems, like file systems, CAS is immutable. It
is more reliable to model a computation by representing the inputs and outputs
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

With this abstraction, it is possible to compose `CASObject`s into a DAG that is
capable of representing complicated data structures, while still allowing data
deduplication. Note you can compare two DAGs by just comparing the CASObject
hash of two root nodes.


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
explicit load is required before accessing the data stored in CASObject.
This load can also fail, for reasons like (but not limited to): object does
not exist, corrupted CAS storage, operation timeout, etc.
* If two `ObjectRef` are equal, it is guaranteed that the object they point to
are identical (if they exist). If they are not equal, the underlying objects are
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

The LLVM ObjectStore API was designed so that it is easy to add
customized CAS implementations that are interchangeable with the builtin
ones.

To add your own implementation, you just need to add a subclass to
`llvm::cas::ObjectStore` and implement all its pure virtual methods.
To be interchangeable with LLVM ObjectStore, the new CAS implementation
needs to conform to following contracts:

* Different CASObjects stored in the ObjectStore need to have a different hash
and result in a different `ObjectRef`. Similarly, the same CASObject should have
the same hash and the same `ObjectRef`. Note: two different CASObjects with
identical data but different references are considered different objects.
* `ObjectRef`s are only comparable within the same `ObjectStore` instance, and
can be used to determine the equality of the underlying CASObjects.
* The loaded objects from the ObjectStore need to have a lifetime at least as
long as the ObjectStore itself so it is always legal to access the loaded data
without holding on the `ObjectProxy` until the `ObjectStore` is destroyed.


If not specified, the behavior can be implementation defined. For example,
`ObjectRef` can be used to point to a loaded CASObject so
`ObjectStore` never fails to load. It is also legal to use a stricter model
than required. For example, the underlying value inside `ObjectRef` can be
the unique indentities of the objects across multiple `ObjectStore` instances,
but comparing such `ObjectRef` from different `ObjectStore` is still illegal.

For CAS library implementers, there is also an `ObjectHandle` class that
is an internal representation of a loaded CASObject reference.
`ObjectProxy` is just a pair of `ObjectHandle` and `ObjectStore`, and
just like `ObjectRef`, `ObjectHandle` is only useful when paired with
the `ObjectStore` that knows about the loaded CASObject.
