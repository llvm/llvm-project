=============
Release Notes
=============

In Polly 3.9 the following important changes have been incorporated.

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



Update of the isl math library
------------------------------

We imported the latest version of the isl math library into Polly.

