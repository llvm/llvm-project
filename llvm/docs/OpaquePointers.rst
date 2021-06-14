===============
Opaque Pointers
===============

The Opaque Pointer Type
=======================

Traditionally, LLVM IR pointer types have contained a pointee type. For example,
``i32 *`` is a pointer that points to an ``i32`` somewhere in memory. However,
due to a lack of pointee type semantics and various issues with having pointee
types, there is a desire to remove pointee types from pointers.

The opaque pointer type project aims to replace all pointer types containing
pointee types in LLVM with an opaque pointer type. The new pointer type is
tentatively represented textually as ``ptr``.

Address spaces are still used to distinguish between different kinds of pointers
where the distinction is relevant for lowering (e.g. data vs function pointers
have different sizes on some architectures). Opaque pointers are not changing
anything related to address spaces and lowering. For more information, see
`DataLayout <LangRef.html#langref-datalayout>`_.

Issues with explicit pointee types
==================================

LLVM IR pointers can be cast back and forth between pointers with different
pointee types. The pointee type does not necessarily actually represent the
actual underlying type in memory. In other words, the pointee type contains no
real semantics.

Lots of operations do not actually care about the underlying type. These
operations, typically intrinsics, usually end up taking an ``i8 *``. This causes
lots of redundant no-op bitcasts in the IR to and from a pointer with a
different pointee type. The extra bitcasts take up space and require extra work
to look through in optimizations. And more bitcasts increases the chances of
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

I Still Need Pointee Types!
===========================

The frontend should already know what type each operation operates on based on
the input source code. However, some frontends like Clang may end up relying on
LLVM pointer pointee types to keep track of pointee types. The frontend needs to
keep track of frontend pointee types on its own.

For optimizations around frontend types, pointee types are not useful due their
lack of semantics. Rather, since LLVM IR works on untyped memory, for a frontend
to tell LLVM about frontend types for the purposes of alias analysis, extra
metadata is added to the IR. For more information, see `TBAA
<LangRef.html#tbaa-metadata>`_.

Some specific operations still need to know what type a pointer types to. For
the most part, this is codegen and ABI specific. For example, `byval
<LangRef.html#parameter-attributes>`_ arguments are pointers, but backends need
to know the underlying type of the argument to properly lower it. In cases like
these, the attributes contain a type argument. For example,

.. code-block:: llvm

  call void @f(ptr byval(i32) %p)

signifies that ``%p`` as an argument should be lowered as an ``i32`` passed
indirectly.

If you have use cases that this sort of fix doesn't cover, please email
llvm-dev.

Transition Plan
===============

LLVM currently has many places that depend on pointee types. Each dependency on
pointee types needs to be resolved in some way or another. This essentially
translates to figuring out how to remove all calls to
``PointerType::getElementType`` and ``Type::getPointerElementType()``.

Making everything use opaque pointers in one huge commit is infeasible. This
needs to be done incrementally. The following steps need to be done, in no
particular order:

* Introduce the opaque pointer type

  * Already done

* Remove remaining in-tree users of pointee types

  * There are many miscellaneous uses that should be cleaned up individually

  * Some of the larger use cases are mentioned below

* Various ABI attributes and instructions that rely on pointee types need to be
  modified to specify the type separately

  * This has already happened for all instructions like loads, stores, GEPs,
    and various attributes like ``byval``

  * More cases may be found as work continues

* Remove calls to and deprecate ``IRBuilder`` methods that rely on pointee types

  * For example, some of the ``IRBuilder::CreateGEP()`` methods use the pointer
    operand's pointee type to determine the GEP operand type

  * Some methods are already deprecated with ``LLVM_ATTRIBUTE_DEPRECATED``, such
    as some overloads of ``IRBuilder::CreateLoad()``

* Allow bitcode auto-upgrade of legacy pointer type to the new opaque pointer
  type (not to be turned on until ready)

  * To support legacy bitcode, such as legacy stores/loads, we need to track
    pointee types for all values since legacy instructions may infer the types
    from a pointer operand's pointee type

* Migrate frontends to not keep track of frontend pointee types via LLVM pointer
  pointee types

  * This is mostly Clang, see ``clang::CodeGen::Address::getElementType()``

* Figure out how to name overloaded intrinsics with pointer parameters

  * See ``getMangledTypeStr()``

* Add option to internally treat all pointer types opaque pointers and see what
  breaks, starting with LLVM tests, then run Clang over large codebases

  * We don't want to start mass-updating tests until we're fairly confident that opaque pointers won't cause major issues

* Replace legacy pointer types in LLVM tests with opaque pointer types

Frontend Migration Steps
========================

If you have your own frontend, there are a couple of things to do after opaque
pointer types fully work.

* Don't rely on LLVM pointee types to keep track of frontend pointee types

* Migrate away from LLVM IR instruction builders that rely on pointee types

  * For example, ``IRBuilder::CreateGEP()`` has multiple overloads; make sure to
    use one where the source element type is explicitly passed in, not inferred
    from the pointer operand pointee type
