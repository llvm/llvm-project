OpenMP Extensions for OpenACC
=============================

OpenACC provides some functionality that OpenMP does not.  In some
cases, Clang supports OpenMP extensions to provide similar
functionality, taking advantage of the runtime implementation already
required for OpenACC.  This section documents those extensions.

By default, Clang recognizes these extensions.  The command-line
option ``-fno-openmp-extensions`` can be specified to disable all
OpenMP extensions, including those described in this section.

.. _ompx-motivation:

Motivation
----------

There are multiple benefits to exposing OpenACC functionality as LLVM
OpenMP extensions:

* OpenMP applications can take advantage of the additional
  functionality.
* As LLVM's implementation of these extensions matures, it can serve
  as a basis for including these extensions in the OpenMP standard.
* Source-to-source translation from certain OpenACC features to OpenMP
  is otherwise impossible.
* Runtime tests can be written in terms of OpenMP instead of OpenACC
  or low-level runtime calls.
* More generally, there is a clean separation of concerns between
  OpenACC and OpenMP development in LLVM.  That is, LLVM's OpenMP
  developers can discuss, modify, and debug LLVM's extended OpenMP
  implementation and test suite without directly considering OpenACC's
  language and execution model, which are handled by LLVM's OpenACC
  developers.

.. _ompx-hold:

``ompx_hold`` Map Type Modifier
-------------------------------

.. _ompx-holdExample:

Example
^^^^^^^

.. code-block:: c++

  #pragma omp target data map(ompx_hold, tofrom: x) // holds onto mapping of x throughout region
  {
    foo(); // might have map(delete: x)
    #pragma omp target map(present, alloc: x) // x is guaranteed to be present
    printf("%d\n", x);
  }

The ``ompx_hold`` map type modifier above specifies that the ``target
data`` directive holds onto the mapping for ``x`` throughout the
associated region regardless of any ``target exit data`` directives
executed during the call to ``foo``.  Thus, the presence assertion for
``x`` at the enclosed ``target`` construct cannot fail.

.. _ompx-holdBehavior:

Behavior
^^^^^^^^

* Stated more generally, the ``ompx_hold`` map type modifier specifies
  that the associated data is not unmapped until the end of the
  construct.  As usual, the standard OpenMP reference count for the
  data must also reach zero before the data is unmapped.
* If ``ompx_hold`` is specified for the same data on lexically or
  dynamically enclosed constructs, there is no additional effect as
  the data mapping is already held throughout their regions.
* The ``ompx_hold`` map type modifier is permitted to appear only on
  ``target`` constructs (and associated combined constructs) and
  ``target data`` constructs.  It is not permitted to appear on
  ``target enter data`` or ``target exit data`` directives because
  there is no associated statement, so it is not meaningful to hold
  onto a mapping until the end of the directive.
* The runtime reports an error if ``omp_target_disassociate_ptr`` is
  called for a mapping for which the ``ompx_hold`` map type modifier
  is in effect.
* Like the ``present`` map type modifier, the ``ompx_hold`` map type
  modifier applies to an entire struct if it's specified for any
  member of that struct even if other ``map`` clauses on the same
  directive specify other members without the ``ompx_hold`` map type
  modifier.
* ``ompx_hold`` support is not yet provided for ``defaultmap``.

Implementation
^^^^^^^^^^^^^^

* LLVM uses the term *dynamic reference count* for the standard OpenMP
  reference count for host/device data mappings.
* The ``ompx_hold`` map type modifier selects an alternate reference
  count, called the *hold reference count*.
* A mapping is removed only once both its reference counts reach zero.
* Because ``ompx_hold`` can appear only constructs, increments and
  decrements of the hold reference count are guaranteed to be
  balanced, so it is impossible to decrement it below zero.
* The dynamic reference count is used wherever ``ompx_hold`` is not
  specified (and possibly cannot be specified).  Decrementing the
  dynamic reference count has no effect if it is already zero.
* The runtime determines that the ``ompx_hold`` map type modifier is
  *in effect* (see :ref:`Behavior <ompx-holdBehavior>` above) when the
  hold reference count is greater than zero.

Relationship with OpenACC
^^^^^^^^^^^^^^^^^^^^^^^^^

OpenACC specifies two reference counts for tracking host/device data
mappings.  Which reference count is used to implement an OpenACC
directive is determined by the nature of that directive, either
dynamic or structured:

* The *dynamic reference count* is always used for ``enter data`` and
  ``exit data`` directives and corresponding OpenACC routines.
* The *structured reference count* is always used for ``data`` and
  compute constructs, which are similar to OpenMP's ``target data``
  and ``target`` constructs.

Contrast with OpenMP, where the dynamic reference count is always used
unless the application developer specifies an alternate behavior via
our map type modifier extension.  We chose the name *hold* for that
map type modifier because, as demonstrated in the above :ref:`example
<ompx-holdExample>`, *hold* concisely identifies the desired behavior
from the application developer's perspective without referencing the
implementation of that behavior.

The hold reference count is otherwise modeled after OpenACC's
structured reference count.  For example, calling ``acc_unmap_data``,
which is similar to ``omp_target_disassociate_ptr``, is an error when
the structured reference count is not zero.

While Flang and Clang obviously must implement the syntax and
semantics for selecting OpenACC reference counts differently than for
selecting OpenMP reference counts, the implementation is the same at
the runtime level.  That is, OpenACC's dynamic reference count is
OpenMP's dynamic reference count, and OpenACC's structured reference
count is our OpenMP hold reference count extension.

.. _atomicWithinTeams:

``atomic`` Strictly Nested Within ``teams``
-------------------------------------------

Example
^^^^^^^

OpenMP 5.2, sec. 10.2 "teams Construct", p. 232, L9-12 restricts what
regions can be strictly nested within a ``teams`` region.  As an
extension, Clang relaxes that restriction in the case of the
``atomic`` construct so that, for example, the following case is
permitted:

.. code-block:: c++

  #pragma omp target teams map(tofrom:x)
  #pragma omp atomic update
  x++;

Relationship with OpenACC
^^^^^^^^^^^^^^^^^^^^^^^^^

This extension is important when translating OpenACC to OpenMP because
OpenACC does not have the same restriction for its corresponding
constructs.  For example, the following is conforming OpenACC:

.. code-block:: c++

  #pragma acc parallel copy(x)
  #pragma acc atomic update
  x++;
