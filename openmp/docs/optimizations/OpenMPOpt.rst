==========================
OpenMP-Aware Optimizations
==========================

LLVM, since `version 11 <https://releases.llvm.org/download.html#11.0.0>`_ (12
Oct 2020), supports an :ref:`OpenMP-Aware optimization pass <OpenMPOpt>`. This
optimization pass will attempt to optimize the module with OpenMP-specific
domain-knowledge. This pass is enabled by default at high optimization levels
(O2 / O3) if compiling with OpenMP support enabled.

.. _OpenMPOpt:

OpenMPOpt
=========

.. contents::
   :local:
   :depth: 1

OpenMPOpt contains several OpenMP-Aware optimizations. This pass is run early on
the entire Module, and later on the entire call graph. Most optimizations done
by OpenMPOpt support remarks. Optimization remarks can be enabled by compiling
with the following flags.

.. code-block:: console

  $ clang -Rpass=openmp-opt -Rpass-missed=openmp-opt -Rpass-analysis=openmp-opt

OpenMP Runtime Call Deduplication
---------------------------------

The OpenMP runtime library contains several functions used to implement features
of the OpenMP standard. Several of the runtime calls are constant within a
parallel region. A common optimization is to replace invariant code with a
single reference, but in this case the compiler will only see an opaque call
into the runtime library. To get around this, OpenMPOpt maintains a list of
OpenMP runtime functions that are constant and will manually deduplicate them.

Globalization
-------------

The OpenMP standard requires that data can be shared between different threads.
This requirement poses a unique challenge when offloading to GPU accelerators.
Data cannot be shared between the threads in a GPU by default, in order to do
this it must either be placed in global or shared memory. This needs to be done
every time a variable may potentially be shared in order to create correct
OpenMP programs. Unfortunately, this has significant performance implications
and is not needed in the majority of cases. For example, when Clang is
generating code for this offloading region, it will see that the variable `x`
escapes and is potentially shared. This will require globalizing the variable,
which means it cannot reside in the registers on the device.

.. code-block:: c++

  void use(void *) { }

  void foo() {
    int x;
    use(&x);
  }

  int main() {
  #pragma omp target parallel
    foo();
  }

In many cases, this transformation is not actually necessary but still carries a
significant performance penalty. Because of this, OpenMPOpt can perform and
inter-procedural optimization and scan each known usage of the globalized
variable and determine if it is potentially captured and shared by another
thread. If it is not actually captured, it can safely be moved back to fast
register memory.

Another case is memory that is intentionally shared between the threads, but is
shared from one thread to all the others. Such variables can be moved to shared
memory when compiled without needing to go through the runtime library.  This
allows for users to confidently declare shared memory on the device without
needing to use custom OpenMP allocators or rely on the runtime.


.. code-block:: c++

  static void share(void *);

  static void foo() {
    int x[64];
  #pragma omp parallel
    share(x);
  }

  int main() {
    #pragma omp target
    foo();
  }

These optimizations can have very large performance implications. Both of these
optimizations rely heavily on inter-procedural analysis. Because of this,
offloading applications should ideally be contained in a single translation unit
and functions should not be externally visible unless needed. OpenMPOpt will
inform the user if any globalization calls remain if remarks are enabled. This
should be treated as a defect in the program.

Resources
=========

- 2021 OpenMP Webinar: "A Compiler's View of OpenMP" https://youtu.be/eIMpgez61r4
- 2020 LLVM Developers’ Meeting: "(OpenMP) Parallelism-Aware Optimizations" https://youtu.be/gtxWkeLCxmU
- 2019 EuroLLVM Developers’ Meeting: "Compiler Optimizations for (OpenMP) Target Offloading to GPUs" https://youtu.be/3AbS82C3X30
