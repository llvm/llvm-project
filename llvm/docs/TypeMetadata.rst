=============
Type Metadata
=============

Type metadata is a mechanism that allows IR modules to co-operatively build
pointer sets corresponding to addresses within a given set of globals. LLVM's
`control flow integrity`_ implementation uses this metadata to efficiently
check (at each call site) that a given address corresponds to either a
valid vtable or function pointer for a given class or function type, and its
whole-program devirtualization pass uses the metadata to identify potential
callees for a given virtual call.

To use the mechanism, a client creates metadata nodes with two elements:

1. a byte offset into the global (generally zero for functions)
2. a metadata object representing an identifier for the type

These metadata nodes are associated with globals by using global object
metadata attachments with the ``!type`` metadata kind.

Each type identifier must exclusively identify either global variables
or functions.

.. admonition:: Limitation

  The current implementation only supports attaching metadata to functions on
  the x86-32 and x86-64 architectures.

An intrinsic, :ref:`llvm.type.test <type.test>`, is used to test whether a
given pointer is associated with a type identifier.

.. _control flow integrity: http://clang.llvm.org/docs/ControlFlowIntegrity.html

Representing Type Information using Type Metadata
=================================================

This section describes how Clang represents C++ type information associated with
virtual tables using type metadata.

Consider the following inheritance hierarchy:

.. code-block:: c++

  struct A {
    virtual void f();
  };

  struct B : A {
    virtual void f();
    virtual void g();
  };

  struct C {
    virtual void h();
  };

  struct D : A, C {
    virtual void f();
    virtual void h();
  };

The virtual table objects for A, B, C and D look like this (under the Itanium ABI):

.. csv-table:: Virtual Table Layout for A, B, C, D
  :header: Class, 0, 1, 2, 3, 4, 5, 6

  A, A::offset-to-top, &A::rtti, &A::f
  B, B::offset-to-top, &B::rtti, &B::f, &B::g
  C, C::offset-to-top, &C::rtti, &C::h
  D, D::offset-to-top, &D::rtti, &D::f, &D::h, D::offset-to-top, &D::rtti, thunk for &D::h

When an object of type A is constructed, the address of ``&A::f`` in A's
virtual table object is stored in the object's vtable pointer.  In ABI parlance
this address is known as an `address point`_. Similarly, when an object of type
B is constructed, the address of ``&B::f`` is stored in the vtable pointer. In
this way, the vtable in B's virtual table object is compatible with A's vtable.

D is a little more complicated, due to the use of multiple inheritance. Its
virtual table object contains two vtables, one compatible with A's vtable and
the other compatible with C's vtable. Objects of type D contain two virtual
pointers, one belonging to the A subobject and containing the address of
the vtable compatible with A's vtable, and the other belonging to the C
subobject and containing the address of the vtable compatible with C's vtable.

The full set of compatibility information for the above class hierarchy is
shown below. The following table shows the name of a class, the offset of an
address point within that class's vtable and the name of one of the classes
with which that address point is compatible.

.. csv-table:: Type Offsets for A, B, C, D
  :header: VTable for, Offset, Compatible Class

  A, 16, A
  B, 16, A
   ,   , B
  C, 16, C
  D, 16, A
   ,   , D
   , 48, C

The next step is to encode this compatibility information into the IR. The way
this is done is to create type metadata named after each of the compatible
classes, with which we associate each of the compatible address points in
each vtable. For example, these type metadata entries encode the compatibility
information for the above hierarchy:

::

  @_ZTV1A = constant [...], !type !0
  @_ZTV1B = constant [...], !type !0, !type !1
  @_ZTV1C = constant [...], !type !2
  @_ZTV1D = constant [...], !type !0, !type !3, !type !4

  !0 = !{i64 16, !"_ZTS1A"}
  !1 = !{i64 16, !"_ZTS1B"}
  !2 = !{i64 16, !"_ZTS1C"}
  !3 = !{i64 16, !"_ZTS1D"}
  !4 = !{i64 48, !"_ZTS1C"}

With this type metadata, we can now use the ``llvm.type.test`` intrinsic to
test whether a given pointer is compatible with a type identifier. Working
backwards, if ``llvm.type.test`` returns true for a particular pointer,
we can also statically determine the identities of the virtual functions
that a particular virtual call may call. For example, if a program assumes
a pointer to be a member of ``!"_ZST1A"``, we know that the address can
be only be one of ``_ZTV1A+16``, ``_ZTV1B+16`` or ``_ZTV1D+16`` (i.e. the
address points of the vtables of A, B and D respectively). If we then load
an address from that pointer, we know that the address can only be one of
``&A::f``, ``&B::f`` or ``&D::f``.

.. _address point: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#vtable-general

Testing Addresses For Type Membership
=====================================

If a program tests an address using ``llvm.type.test``, this will cause
a link-time optimization pass, ``LowerTypeTests``, to replace calls to this
intrinsic with efficient code to perform type member tests. At a high level,
the pass will lay out referenced globals in a consecutive memory region in
the object file, construct bit vectors that map onto that memory region,
and generate code at each of the ``llvm.type.test`` call sites to test
pointers against those bit vectors. Because of the layout manipulation, the
globals' definitions must be available at LTO time. For more information,
see the `control flow integrity design document`_.

A type identifier that identifies functions is transformed into a jump table,
which is a block of code consisting of one branch instruction for each
of the functions associated with the type identifier that branches to the
target function. The pass will redirect any taken function addresses to the
corresponding jump table entry. In the object file's symbol table, the jump
table entries take the identities of the original functions, so that addresses
taken outside the module will pass any verification done inside the module.

Jump tables may call external functions, so their definitions need not
be available at LTO time. Note that if an externally defined function is
associated with a type identifier, there is no guarantee that its identity
within the module will be the same as its identity outside of the module,
as the former will be the jump table entry if a jump table is necessary.

The `GlobalLayoutBuilder`_ class is responsible for laying out the globals
efficiently to minimize the sizes of the underlying bitsets.

.. _control flow integrity design document: http://clang.llvm.org/docs/ControlFlowIntegrityDesign.html

:Example:

::

    target datalayout = "e-p:32:32"

    @a = internal global i32 0, !type !0
    @b = internal global i32 0, !type !0, !type !1
    @c = internal global i32 0, !type !1
    @d = internal global [2 x i32] [i32 0, i32 0], !type !2

    define void @e() !type !3 {
      ret void
    }

    define void @f() {
      ret void
    }

    declare void @g() !type !3

    !0 = !{i32 0, !"typeid1"}
    !1 = !{i32 0, !"typeid2"}
    !2 = !{i32 4, !"typeid2"}
    !3 = !{i32 0, !"typeid3"}

    declare i1 @llvm.type.test(i8* %ptr, metadata %typeid) nounwind readnone

    define i1 @foo(i32* %p) {
      %pi8 = bitcast i32* %p to i8*
      %x = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid1")
      ret i1 %x
    }

    define i1 @bar(i32* %p) {
      %pi8 = bitcast i32* %p to i8*
      %x = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid2")
      ret i1 %x
    }

    define i1 @baz(void ()* %p) {
      %pi8 = bitcast void ()* %p to i8*
      %x = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid3")
      ret i1 %x
    }

    define void @main() {
      %a1 = call i1 @foo(i32* @a) ; returns 1
      %b1 = call i1 @foo(i32* @b) ; returns 1
      %c1 = call i1 @foo(i32* @c) ; returns 0
      %a2 = call i1 @bar(i32* @a) ; returns 0
      %b2 = call i1 @bar(i32* @b) ; returns 1
      %c2 = call i1 @bar(i32* @c) ; returns 1
      %d02 = call i1 @bar(i32* getelementptr ([2 x i32]* @d, i32 0, i32 0)) ; returns 0
      %d12 = call i1 @bar(i32* getelementptr ([2 x i32]* @d, i32 0, i32 1)) ; returns 1
      %e = call i1 @baz(void ()* @e) ; returns 1
      %f = call i1 @baz(void ()* @f) ; returns 0
      %g = call i1 @baz(void ()* @g) ; returns 1
      ret void
    }

.. _GlobalLayoutBuilder: https://github.com/llvm/llvm-project/blob/master/llvm/include/llvm/Transforms/IPO/LowerTypeTests.h
