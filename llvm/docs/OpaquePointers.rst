===============
Opaque Pointers
===============

The Opaque Pointer Type
=======================

Traditionally, LLVM IR pointer types have contained a pointee type. For example,
``i32*`` is a pointer that points to an ``i32`` somewhere in memory. However,
due to a lack of pointee type semantics and various issues with having pointee
types, there is a desire to remove pointee types from pointers.

The opaque pointer type project aims to replace all pointer types containing
pointee types in LLVM with an opaque pointer type. The new pointer type is
tentatively represented textually as ``ptr``.

Address spaces are still used to distinguish between different kinds of pointers
where the distinction is relevant for lowering (e.g. data vs function pointers
have different sizes on some architectures). Opaque pointers are not changing
anything related to address spaces and lowering. For more information, see
`DataLayout <LangRef.html#langref-datalayout>`_. Opaque pointers in non-default
address space are spelled ``ptr addrspace(N)``.

Issues with explicit pointee types
==================================

LLVM IR pointers can be cast back and forth between pointers with different
pointee types. The pointee type does not necessarily represent the actual
underlying type in memory. In other words, the pointee type carries no real
semantics.

Lots of operations do not actually care about the underlying type. These
operations, typically intrinsics, usually end up taking an ``i8*``. This causes
lots of redundant no-op bitcasts in the IR to and from a pointer with a
different pointee type. The extra bitcasts take up space and require extra work
to look through in optimizations. And more bitcasts increase the chances of
incorrect bitcasts, especially in regards to address spaces.

Some instructions still need to know what type to treat the memory pointed to by
the pointer as. For example, a load needs to know how many bytes to load from
memory. In these cases, instructions themselves contain a type argument. For
example the load instruction from older versions of LLVM

.. code-block:: llvm

  load i64* %p

becomes

.. code-block:: llvm

  load i64, ptr %p

A nice analogous transition that happened earlier in LLVM is integer signedness.
There is no distinction between signed and unsigned integer types, rather the
integer operations themselves contain what to treat the integer as. Initially,
LLVM IR distinguished between unsigned and signed integer types. The transition
from manifesting signedness in types to instructions happened early on in LLVM's
life to the betterment of LLVM IR.

Opaque Pointers Mode
====================

During the transition phase, LLVM can be used in two modes: In typed pointer
mode (currently still the default) all pointer types have a pointee type and
opaque pointers cannot be used. In opaque pointers mode, all pointers are
opaque. The opaque pointer mode can be enabled using ``-opaque-pointers`` in
LLVM tools like ``opt``, or ``-Xclang -opaque-pointers`` in clang. Additionally,
opaque pointer mode is automatically enabled for IR and bitcode files that use
the ``ptr`` type.

In opaque pointer mode, all typed pointers used in IR, bitcode, or created
using ``PointerType::get()`` and similar APIs are automatically converted into
opaque pointers. This simplifies migration and allows testing existing IR with
opaque pointers.

.. code-block:: llvm

   define i8* @test(i8* %p) {
     %p2 = getelementptr i8, i8* %p, i64 1
     ret i8* %p2
   }

   ; Is automatically converted into the following if -opaque-pointers
   ; is enabled:

   define ptr @test(ptr %p) {
     %p2 = getelementptr i8, ptr %p, i64 1
     ret ptr %p2
   }

Migration Instructions
======================

In order to support opaque pointers, two types of changes tend to be necessary.
The first is the removal of all calls to ``PointerType::getElementType()`` and
``Type::getPointerElementType()``.

In the LLVM middle-end and backend, this is usually accomplished by inspecting
the type of relevant operations instead. For example, memory access related
analyses and optimizations should use the types encoded in the load and store
instructions instead of querying the pointer type.

Here are some common ways to avoid pointer element type accesses:

* For loads, use ``getType()``.
* For stores, use ``getValueOperand()->getType()``.
* Use ``getLoadStoreType()`` to handle both of the above in one call.
* For getelementptr instructions, use ``getSourceElementType()``.
* For calls, use ``getFunctionType()``.
* For allocas, use ``getAllocatedType()``.
* For globals, use ``getValueType()``.
* For consistency assertions, use
  ``PointerType::isOpaqueOrPointeeTypeEquals()``.
* To create a pointer type in a different address space, use
  ``PointerType::getWithSamePointeeType()``.
* To check that two pointers have the same element type, use
  ``PointerType::hasSameElementTypeAs()``.
* While it is preferred to write code in a way that accepts both typed and
  opaque pointers, ``Type::isOpaquePointerTy()`` and
  ``PointerType::isOpaque()`` can be used to handle opaque pointers specially.
  ``PointerType::getNonOpaquePointerElementType()`` can be used as a marker in
  code-paths where opaque pointers have been explicitly excluded.
* To get the type of a byval argument, use ``getParamByValType()``. Similar
  method exists for other ABI-affecting attributes that need to know the
  element type, such as byref, sret, inalloca and preallocated.
* Some intrinsics require an ``elementtype`` attribute, which can be retrieved
  using ``getParamElementType()``. This attribute is required in cases where
  the intrinsic does not naturally encode a needed element type. This is also
  used for inline assembly.

Note that some of the methods mentioned above only exist to support both typed
and opaque pointers at the same time, and will be dropped once the migration
has completed. For example, ``isOpaqueOrPointeeTypeEquals()`` becomes
meaningless once all pointers are opaque.

While direct usage of pointer element types is immediately apparent in code,
there is a more subtle issue that opaque pointers need to contend with: A lot
of code assumes that pointer equality also implies that the used load/store
type or GEP source element type is the same. Consider the following examples
with typed an opaque pointers:

.. code-block:: llvm

    define i32 @test(i32* %p) {
      store i32 0, i32* %p
      %bc = bitcast i32* %p to i64*
      %v = load i64, i64* %bc
      ret i64 %v
    }

    define i32 @test(ptr %p) {
      store i32 0, ptr %p
      %v = load i64, ptr %p
      ret i64 %v
    }

Without opaque pointers, a check that the pointer operand of the load and
store are the same also ensures that the accessed type is the same. Using a
different type requires a bitcast, which will result in distinct pointer
operands.

With opaque pointers, the bitcast is not present, and this check is no longer
sufficient. In the above example, it could result in store to load forwarding
of an incorrect type. Code making such assumptions needs to be adjusted to
check the accessed type explicitly:
``LI->getType() == SI->getValueOperand()->getType()``.

Frontends
---------

Frontends need to be adjusted to track pointee types independently of LLVM,
insofar as they are necessary for lowering. For example, clang now tracks the
pointee type in the ``Address`` structure.

Frontends using the C API through an FFI interface should be aware that a
number of C API functions are deprecated and will be removed as part of the
opaque pointer transition::

    LLVMBuildLoad -> LLVMBuildLoad2
    LLVMBuildCall -> LLVMBuildCall2
    LLVMBuildInvoke -> LLVMBuildInvoke2
    LLVMBuildGEP -> LLVMBuildGEP2
    LLVMBuildInBoundsGEP -> LLVMBuildInBoundsGEP2
    LLVMBuildStructGEP -> LLVMBuildStructGEP2
    LLVMBuildPtrDiff -> LLVMBuildPtrDiff2
    LLVMConstGEP -> LLVMConstGEP2
    LLVMConstInBoundsGEP -> LLVMConstInBoundsGEP2
    LLVMAddAlias -> LLVMAddAlias2

Additionally, it will no longer be possible to call ``LLVMGetElementType()``
on a pointer type.

Transition State
================

As of April 2022 both LLVM and Clang have complete support for opaque pointers,
and opaque pointers are enabled by default in Clang. It is possible to
temporarily restore the old default using the
``-DCLANG_ENABLE_OPAQUE_POINTERS=OFF`` cmake option. Opaque pointers can be
disabled for a single Clang invocation using ``-Xclang -no-opaque-pointers``.

The MLIR and Polly monorepo projects are not fully compatible with opaque
pointers yet.
