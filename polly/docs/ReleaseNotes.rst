========================
Release Notes (upcoming)
========================

In Polly 3.9 the following important changes have been incorporated.

Polly directly available in clang/opt/bugpoint
----------------------------------------------

Polly supported since a long time to be directly linked into tools such as
opt/clang/bugpoint. Since this release, the default for a source checkout that
contains Polly is to provide Polly directly through these tools, rather than as
an additional module. This makes using Polly significantly easier.

Instead of

.. code-block:: bash

    opt -load lib/LLVMPolly.so -O3 -polly file.ll
    clang -Xclang -load -Xclang lib/LLVMPolly.so -O3 -mllvm -polly file.ll

one can now use

.. code-block:: bash

    opt -O3 -polly file.ll
    clang -O3 -mllvm -polly file.c


Increased analysis coverage
---------------------------

Polly's modeling has been improved to increase the applicability of Polly. The
following code pieces are newly supported:

Arrays accessed through different types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is not uncommon that one array stores elements of different types. Polly now
can model and optimize such code.

.. code-block:: c

    void multiple_types(char *Short, char *Float, char *Double) {
      for (long i = 0; i < 100; i++) {
        Short[i] = *(short *)&Short[2 * i];
        Float[i] = *(float *)&Float[4 * i];
        Double[i] = *(double *)&Double[8 * i];
      }
    }


If the accesses are not aligned with the size of the access type we model them
as multiple accesses to an array of smaller elements. This is especially
useful for structs containing different typed elements as accesses to them are
represented using only one base pointer, namely the ``struct`` itself.  In the
example below the accesses to ``s`` are all modeled as if ``s`` was a single
char array because the accesses to ``s->A`` and ``s->B`` are not aligned with
their respective type size (both are off-by-one due to the ``char`` field in
the ``struct``).

.. code-block:: c

    struct S {
      char Offset;
      int A[100];
      double B[100];
    };

    void struct_accesses(struct S *s) {
      for (long i = 0; i < 100; i++)
        s->B[i] += s->A[i];
    }



Function calls with known side effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Function calls that have only known memory effects can be represented as
accesses in the polyhedral model. While calls without side effects were
supported before, we now allow and model two other kinds. The first are
intrinsic calls to ``memcpy``, ``memmove`` and ``memset``. These calls can be
represented precisely if the pointers involved are known and the given length
is affine. Additionally, we allow to over-approximate function calls that are
known only to read memory, read memory accessible through pointer arguments or
access only memory accessible through pointer arguments. See also the function
attributes ``readonly`` and ``argmemonly`` for more information.

Fine-grain dependences analysis
-------------------------------

In addition of the ScopStmt wise dependences analysis, now the "polly-dependence"
pass provides dependences analysis at memory reference wise and memory access wise.
The memory reference wise analysis distinguishes the accessed references in the
same statement, and generates dependences relationships between (statement, reference)
pairs. The memory access wise analysis distinguishes accesses in the same statement,
and generates dependences relationships between (statement, access) pairs. These
fine-grain dependences are enabled by "-polly-dependences-analysis-level=reference-wise"
and "-polly-dependences-analysis-level=access-wise", respectively.

Update of the isl math library
------------------------------

We imported the latest version of the isl math library into Polly.

