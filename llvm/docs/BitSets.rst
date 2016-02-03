=======
Bitsets
=======

This is a mechanism that allows IR modules to co-operatively build pointer
sets corresponding to addresses within a given set of globals. One example
of a use case for this is to allow a C++ program to efficiently verify (at
each call site) that a vtable pointer is in the set of valid vtable pointers
for the type of the class or its derived classes.

To use the mechanism, a client creates a global metadata node named
``llvm.bitsets``.  Each element is a metadata node with three elements:

1. a metadata object representing an identifier for the bitset
2. either a global variable or a function
3. a byte offset into the global (generally zero for functions)

Each bitset must exclusively contain either global variables or functions.

.. admonition:: Limitation

  The current implementation only supports functions as members of bitsets on
  the x86-32 and x86-64 architectures.

An intrinsic, :ref:`llvm.bitset.test <bitset.test>`, is used to test
whether a given pointer is a member of a bitset.

Representing Type Information using Bitsets
===========================================

This section describes how Clang represents C++ type information associated with
virtual tables using bitsets.

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

.. csv-table:: Bitsets for A, B, C, D
  :header: VTable for, Offset, Compatible Class

  A, 16, A
  B, 16, A
   ,   , B
  C, 16, C
  D, 16, A
   ,   , D
   , 48, C

The next step is to encode this compatibility information into the IR. The way
this is done is to create bitsets named after each of the compatible classes,
into which we add each of the compatible address points in each vtable.
For example, these bitset entries encode the compatibility information for
the above hierarchy:

::

  !0 = !{!"_ZTS1A", [3 x i8*]* @_ZTV1A, i64 16}
  !1 = !{!"_ZTS1A", [4 x i8*]* @_ZTV1B, i64 16}
  !2 = !{!"_ZTS1B", [4 x i8*]* @_ZTV1B, i64 16}
  !3 = !{!"_ZTS1C", [3 x i8*]* @_ZTV1C, i64 16}
  !4 = !{!"_ZTS1A", [7 x i8*]* @_ZTV1D, i64 16}
  !5 = !{!"_ZTS1D", [7 x i8*]* @_ZTV1D, i64 16}
  !6 = !{!"_ZTS1C", [7 x i8*]* @_ZTV1D, i64 48}

With these bitsets, we can now use the ``llvm.bitset.test`` intrinsic to test
whether a given pointer is compatible with a bitset. Working backwards,
if ``llvm.bitset.test`` returns true for a particular pointer, we can also
statically determine the identities of the virtual functions that a particular
virtual call may call. For example, if a program assumes a pointer to be in the
``!"_ZST1A"`` bitset, we know that the address can be only be one of ``_ZTV1A+16``,
``_ZTV1B+16`` or ``_ZTV1D+16`` (i.e. the address points of the vtables of A,
B and D respectively). If we then load an address from that pointer, we know
that the address can only be one of ``&A::f``, ``&B::f`` or ``&D::f``.

.. _address point: https://mentorembedded.github.io/cxx-abi/abi.html#vtable-general

Testing Bitset Addresses
========================

If a program tests an address using ``llvm.bitset.test``, this will cause
a link-time optimization pass, ``LowerBitSets``, to replace calls to this
intrinsic with efficient code to perform bitset tests. At a high level,
the pass will lay out referenced globals in a consecutive memory region in
the object file, construct bit vectors that map onto that memory region,
and generate code at each of the ``llvm.bitset.test`` call sites to test
pointers against those bit vectors. Because of the layout manipulation, the
globals' definitions must be available at LTO time. For more information,
see the `control flow integrity design document`_.

A bit set containing functions is transformed into a jump table, which is a
block of code consisting of one branch instruction for each of the functions
in the bit set that branches to the target function. The pass will redirect
any taken function addresses to the corresponding jump table entry. In the
object file's symbol table, the jump table entries take the identities of
the original functions, so that addresses taken outside the module will pass
any verification done inside the module.

Jump tables may call external functions, so their definitions need not
be available at LTO time. Note that if an externally defined function is a
member of a bitset, there is no guarantee that its identity within the module
will be the same as its identity outside of the module, as the former will
be the jump table entry if a jump table is necessary.

The `GlobalLayoutBuilder`_ class is responsible for laying out the globals
efficiently to minimize the sizes of the underlying bitsets.

.. _control flow integrity design document: http://clang.llvm.org/docs/ControlFlowIntegrityDesign.html

:Example:

::

    target datalayout = "e-p:32:32"

    @a = internal global i32 0
    @b = internal global i32 0
    @c = internal global i32 0
    @d = internal global [2 x i32] [i32 0, i32 0]

    define void @e() {
      ret void
    }

    define void @f() {
      ret void
    }

    declare void @g()

    !llvm.bitsets = !{!0, !1, !2, !3, !4, !5, !6}

    !0 = !{!"bitset1", i32* @a, i32 0}
    !1 = !{!"bitset1", i32* @b, i32 0}
    !2 = !{!"bitset2", i32* @b, i32 0}
    !3 = !{!"bitset2", i32* @c, i32 0}
    !4 = !{!"bitset2", i32* @d, i32 4}
    !5 = !{!"bitset3", void ()* @e, i32 0}
    !6 = !{!"bitset3", void ()* @g, i32 0}

    declare i1 @llvm.bitset.test(i8* %ptr, metadata %bitset) nounwind readnone

    define i1 @foo(i32* %p) {
      %pi8 = bitcast i32* %p to i8*
      %x = call i1 @llvm.bitset.test(i8* %pi8, metadata !"bitset1")
      ret i1 %x
    }

    define i1 @bar(i32* %p) {
      %pi8 = bitcast i32* %p to i8*
      %x = call i1 @llvm.bitset.test(i8* %pi8, metadata !"bitset2")
      ret i1 %x
    }

    define i1 @baz(void ()* %p) {
      %pi8 = bitcast void ()* %p to i8*
      %x = call i1 @llvm.bitset.test(i8* %pi8, metadata !"bitset3")
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

.. _GlobalLayoutBuilder: http://llvm.org/klaus/llvm/blob/master/include/llvm/Transforms/IPO/LowerBitSets.h
