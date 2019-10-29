# Pointer Authentication

## Introduction

Pointer Authentication is a mechanism by which certain pointers are signed,
are modified to embed that signature in their unused bits, and are
authenticated (have their signature checked) when used, to prevent pointers
of unknown origin from being injected into a process.

To enforce Control Flow Integrity (CFI), this is mostly used for all code
pointers (function pointers, vtables, ...), but certain data pointers specified
by the ABI (vptr, ...) are also authenticated.

Additionally, with clang extensions, users can specify that a given pointer
be signed/authenticated.

At the IR level, it is represented using:

* a [set of intrinsics](#intrinsics) (to sign/authenticate pointers)
* a [special section and relocation](#authenticated-global-relocation)
  (to sign globals)
* a [call operand bundle](#operand-bundle) (to authenticate called pointers)

It is implemented by the [AArch64 target](#aarch64-support), using the
[ARMv8.3 Pointer Authentication Code](#armv8-3-pointer-authentication-code)
instructions, to support the Darwin [arm64e](#arm64e) ABI.


## Concepts

### Operations

Pointer Authentication is based on three fundamental operations:

#### Sign
* compute a cryptographic signature of a given pointer value
* embed it within the value
* return the signed value

#### Auth
* compute a cryptographic signature of a given value
* compare it against the embedded signature
* remove the embedded signature
* return the raw, unauthenticated, value

#### Strip
* remove the embedded signature
* return the unauthenticated value


### Diversity

To prevent any signed pointer from being used instead of any other signed
pointer, the signatures are diversified, using additional inputs:

* a key: one of a small, fixed set.  The value of the key itself is not
  directly accessible, but is referenced by ptrauth operations via an
  identifier.

* salt, or extra diversity data: additional data mixed in with the value and
  used by the ptrauth operations.
  A concrete value is called a "discriminator", and, in the special case where
  the diversity data is a pointer to the storage location of the signed value,
  the value is said to be "address-discriminated".
  Additionally, an arbitrary small integer can be blended into an address
  discriminator to produce a blended address discriminator.

Keys are not necessarily interchangeable, and keys can be specified to be
incompatible with certain kinds of pointers (e.g., code vs data keys/pointers).
Which keys are appropriate for a given kind of pointer is defined by the
target implementation.

## LLVM IR Representation

### Intrinsics

These intrinsics are provided by LLVM to expose pointer authentication
operations.


#### '``llvm.ptrauth.sign``'

##### Syntax:

```llvm
declare i64 @llvm.ptrauth.sign.i64(i64 <value>, i32 <key>, i64 <extra data>)
```

##### Overview:

The '``llvm.ptrauth.sign``' intrinsic signs an unauthenticated pointer.


##### Arguments:

The ``value`` argument is the unauthenticated (raw) pointer value to be signed.
The ``key`` argument is the identifier of the key to be used to generate the
signed value.
The ``extra data`` argument is the additional diversity data to be used as a
discriminator.

##### Semantics:

The '``llvm.ptrauth.sign``' intrinsic implements the `sign`_ operation.
It returns a signed value.

If ``value`` is already a signed value, the behavior is undefined.

If ``value`` is not a pointer value for which ``key`` is appropriate, the
behavior is undefined.


#### '``llvm.ptrauth.auth``'

##### Syntax:

```llvm
declare i64 @llvm.ptrauth.auth.i64(i64 <value>, i32 <key>, i64 <extra data>)
```

##### Overview:

The '``llvm.ptrauth.auth``' intrinsic authenticates a signed pointer.

##### Arguments:

The ``value`` argument is the signed pointer value to be authenticated.
The ``key`` argument is the identifier of the key that was used to generate
the signed value.
The ``extra data`` argument is the additional diversity data to be used as a
discriminator.

##### Semantics:

The '``llvm.ptrauth.auth``' intrinsic implements the `auth`_ operation.
It returns a raw, unauthenticated value.
If ``value`` does not have a correct signature for ``key`` and ``extra data``,
the returned value is an invalid, poison pointer.


#### '``llvm.ptrauth.strip``'

##### Syntax:

```llvm
declare i64 @llvm.ptrauth.strip.i64(i64 <value>, i32 <key>)
```

##### Overview:

The '``llvm.ptrauth.strip``' intrinsic strips the embedded signature out of a
possibly-signed pointer.


##### Arguments:

The ``value`` argument is the signed pointer value to be stripped.
The ``key`` argument is the identifier of the key that was used to generate
the signed value.

##### Semantics:

The '``llvm.ptrauth.strip``' intrinsic implements the `strip`_ operation.
It returns an unauthenticated value.  It does **not** check that the
signature is valid.

If ``value`` is an unauthenticated pointer value, it is returned as-is,
provided the ``key`` is appropriate for the pointer.

If ``value`` is not a pointer value for which ``key`` is appropriate, the
behavior is undefined.

If ``value`` is a signed pointer value, but ``key`` does not identify the
same ``key`` that was used to generate ``value``, the behavior is undefined.


#### '``llvm.ptrauth.resign``'

##### Syntax:

```llvm
declare i64 @llvm.ptrauth.resign.i64(i64 <value>,
                                     i32 <old key>, i64 <old extra data>,
                                     i32 <new key>, i64 <new extra data>)
```

##### Overview:

The '``llvm.ptrauth.resign``' intrinsic re-signs a signed pointer using
a different key and diversity data.

##### Arguments:

The ``value`` argument is the signed pointer value to be authenticated.
The ``old key`` argument is the identifier of the key that was used to generate
the signed value.
The ``old extra data`` argument is the additional diversity data to be used as a
discriminator in the auth operation.
The ``new key`` argument is the identifier of the key to use to generate the
resigned value.
The ``new extra data`` argument is the additional diversity data to be used as a
discriminator in the sign operation.

##### Semantics:

The '``llvm.ptrauth.resign``' intrinsic performs a combined `auth`_ and `sign`_
operation, without exposing the intermediate unauthenticated pointer.
It returns a signed value.
If ``value`` does not have a correct signature for ``old key`` and
``old extra data``, the returned value is an invalid, poison pointer.

#### '``llvm.ptrauth.sign_generic``'

##### Syntax:

```llvm
declare i64 @llvm.ptrauth.sign_generic.i64(i64 <value>, i64 <extra data>)
```

##### Overview:

The '``llvm.ptrauth.sign_generic``' intrinsic computes a generic signature of
arbitrary data.

##### Arguments:

The ``value`` argument is the arbitrary data value to be signed.
The ``extra data`` argument is the additional diversity data to be used as a
discriminator.

##### Semantics:

The '``llvm.ptrauth.sign_generic``' intrinsic computes the signature of a given
combination of value and additional diversity data.

It returns a full signature value (as opposed to a signed pointer value, with
an embedded signature).

As opposed to [``llvm.ptrauth.sign``](#llvm-ptrauth-sign), it does not interpret
``value`` as a pointer value.  Instead, it is an arbitrary data value.


#### '``llvm.ptrauth.blend``'

##### Syntax:

```llvm
declare i64 @llvm.ptrauth.blend.i64(i64 <address discriminator>, i64 <integer discriminator>)
```

##### Overview:

The '``llvm.ptrauth.blend``' intrinsic blends a pointer address discriminator
with a small integer discriminator to produce a new discriminator.

##### Arguments:

The ``address discriminator`` argument is a pointer.
The ``integer discriminator`` argument is a small integer.

##### Semantics:

The '``llvm.ptrauth.blend``' intrinsic combines a small integer discriminator
with a pointer address discriminator, in a way that is specified by the target
implementation.


### Operand Bundle

As a way to enforce CFI, function pointers used as indirect call targets are
signed when materialized, and authenticated before calls.

To prevent the intermediate, unauthenticated pointer from being exposed to
attackers (similar to [``llvm.ptrauth.resign``](#llvm-ptrauth-resign)), the
representation guarantees that the intermediate call target is never attackable
(e.g., by being spilled to memory), using the ``ptrauth`` operand bundle.

```llvm
define void @f(void ()* %fp) {
  call void %fp() [ "ptrauth"(i32 <key>, i64 <data>) ]
  ret void
}
```

is functionally equivalent to:

```llvm
define void @f(void ()* %fp) {
  %fp_i = ptrtoint void ()* %fp to i64
  %fp_auth = call i64 @llvm.ptrauth.auth.i64(i64 %fp_i, i32 <key>, i64 <data>)
  %fp_auth_p = inttoptr i64 %fp_auth to void ()*
  call void %fp_auth_p()
  ret void
}
```

but with the added guarantee that ``%fp_i``, ``%fp_auth``, and ``%fp_auth_p``
are never attackable.


### Function Attributes

Two function attributes are used to describe other pointer authentication
operations that are not otherwise explicitly expressed in IR.

#### ``ptrauth-returns``

``ptrauth-returns`` specifies that returns from functions should be
authenticated, and that saved return addresses should be signed.

Note that this describes the execution environment that can be assumed by
this function, not the semantics of return instructions in this function alone.

The semantics of
[``llvm.returnaddress``](LangRef.html#llvm-returnaddress-intrinsic) are not
changed (it still returns a raw, unauthenticated, return address), so it might
require an implicit strip/authenticate operation.  This applies to return
addresses stored in deeper stack frames.

#### ``ptrauth-calls``

``ptrauth-calls`` specifies that calls emitted in this function should be
authenticated according to the platform ABI.

Calls represented by ``call``/``invoke`` instructions in IR are not affected by
this attribute, as they should already be annotated with the
[``ptrauth`` operand bundle](#operand-bundle).

The ``ptrauth-calls`` attribute only describes calls emitted by the backend,
as part of target-specific lowering (e.g., runtime calls for TLS accesses).


### Authenticated Global Relocation

[Intrinsics](#intrinsics) can be used to produce signed pointers dynamically,
in code, but not for signed pointers referenced by constants, in, e.g., global
initializers.

The latter are represented using a special kind of global describing an
authenticated relocation (producing a signed pointer).

These special global must live in section '``llvm.ptrauth``', and have a
specific type.

```llvm
@fp.ptrauth = constant { i8*, i32, i64, i64 }
                       { i8* <value>,
                         i32 <key>,
                         i64 <address discriminator>,
                         i64 <integer discriminator>
                       }, section "llvm.ptrauth"
```

is equivalent to ``@fp.ptrauth`` being initialized with:

```llvm
  %disc = call i64 @llvm.ptrauth.blend.i64(i64 <address discriminator>, i64 <integer discriminator>)
  %signed_fp = call i64 @llvm.ptrauth.sign.i64(i64 bitcast (i8* <value> to i64), i32 <key>, i64 %disc)
  %fp_p_loc = bitcast { i8*, i32, i64, i64 }* @fp.ptrauth to i64*
  store i64 %signed_fp, i8* %fp_p_loc
```

Note that this is a temporary representation, chosen to minimize divergence with
upstream.  Ideally, this would simply be a new kind of ConstantExpr.



## AArch64 Support

AArch64 is currently the only target with full support of the pointer
authentication primitives, based on ARMv8.3 instructions.

### ARMv8.3 Pointer Authentication Code

[ARMv8.3] is an ISA extension that includes Pointer Authentication Code (PAC)
instructions.

[ARMv8.3]: https://developer.arm.com/products/architecture/cpu-architecture/a-profile/docs/ddi0487/latest

#### Keys

5 keys are supported by ARMv8.3.

Of those, 4 keys are interchangeably usable to specify the key used in IR
constructs:
* ``ASIA``/``ASIB`` are instruction keys (encoded as respectively 0 and 1).
* ``ASDA``/``ASDB`` are data keys (encoded as respectively 2 and 3).

``ASGA`` is a special key that cannot be explicitly specified, and is only ever
used implicitly, to implement the
[``llvm.ptrauth.sign_generic``](#llvm-ptrauth-sign-generic) intrinsic.

#### Instructions

The IR [Intrinsics](#intrinsics) described above map onto these
instructions as such:
* [``llvm.ptrauth.sign``](#llvm-ptrauth-sign): ``PAC{I,D}{A,B}{Z,SP,}``
* [``llvm.ptrauth.auth``](#llvm-ptrauth-auth): ``AUT{I,D}{A,B}{Z,SP,}``
* [``llvm.ptrauth.strip``](#llvm-ptrauth-strip): ``XPAC{I,D}``
* [``llvm.ptrauth.blend``](#llvm-ptrauth-blend): The semantics of the
  blend operation are, in effect, specified by the ABI.  arm64e specifies it as
  a ``MOVK`` into the high 16-bits.
* [``llvm.ptrauth.sign_generic``](#llvm-ptrauth-sign-generic): ``PACGA``
* [``llvm.ptrauth.resign``](#llvm-ptrauth-resign): ``AUT*+PAC*``.  These are
  represented as a single pseudo-instruction in the backend to guarantee that
  the intermediate unauthenticated value is not spilled and attackable.

### arm64e

Darwin supports ARMv8.3 Pointer Authentication Codes via the arm64e MachO
architecture slice.

#### CPU Subtype

The arm64e slice is an extension of the ``arm64`` slice (so uses the same
MachO ``cpu_type``, ``CPU_TYPE_ARM64``).

It is mainly represented using the ``cpu_subtype`` 2, or ``CPU_SUBTYPE_ARM64E``.

The subtype also encodes the version of the pointer authentication ABI used in
the object:

```
| 31-28 |     28-25    |      24-0      |
| ----- | ------------ | -------------- |
|  0000 |  ABI version | 0000 0000 0010 |
```


#### Assembly Representation

At the assembly level,
[Authenticated Relocations](#authenticated-global-relocation) are represented
using the ``@AUTH`` modifier:

```asm
    .quad _target@AUTH(<key>,<discriminator>[,addr])
```

where:
* ``key`` is the ARMv8.3 key identifier (``ia``, ``ib``, ``da``, ``db``)
* ``discriminator`` is the 16-bit unsigned discriminator value
* ``addr`` signifies that the authenticated pointer is address-discriminated
  (that is, that the relocation's target address is to be blended into the
  ``discriminator`` before it is used in the sign operation.

For example:
```asm
  _authenticated_reference_to_sym:
    .quad _sym@AUTH(db,0)

  _authenticated_reference_to_sym_addr_disc:
    .quad _sym@AUTH(ia,12,addr)
```

#### Object File Representation

At the binary object file level,
[Authenticated Relocations](#authenticated-global-relocation) are represented
using the ``ARM64_RELOC_AUTHENTICATED_POINTER`` relocation kind (with value
``11``).

The pointer authentication information is encoded into the addend, as such:

```
| 63 | 62 | 61-51 | 50-49 |   48   | 47     -     32 | 31  -  0 |
| -- | -- | ----- | ----- | ------ | --------------- | -------- |
|  1 |  0 |   0   |  key  |  addr  |  discriminator  |  addend  |
```
