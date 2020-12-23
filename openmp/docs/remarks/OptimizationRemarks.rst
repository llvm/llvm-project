OpenMP Optimization Remarks
===========================

The :doc:`OpenMP-Aware optimization pass </optimizations/OpenMPOpt>` is able to
generate compiler remarks for performed and missed optimisations. To emit them,
pass ``-Rpass=openmp-opt``, ``-Rpass-analysis=openmp-opt``, and
``-Rpass-missed=openmp-opt`` to the Clang invocation.  For more information and
features of the remark system the clang documentation should be consulted:

+ `Clang options to emit optimization reports <https://clang.llvm.org/docs/UsersManual.html#options-to-emit-optimization-reports>`_
+ `Clang diagnostic and remark flags <https://clang.llvm.org/docs/ClangCommandLineReference.html#diagnostic-flags>`_
+ The `-foptimization-record-file flag
  <https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-foptimization-record-file>`_
  and the `-fsave-optimization-record flag
  <https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang1-fsave-optimization-record>`_


.. _ompXXX:

Some OpenMP remarks start with a "tag", like `[OMP100]`, which indicates that
there is further information about them on this page. To directly jump to the
respective entry, navigate to
`https://openmp.llvm.org/docs/remarks/OptimizationRemarks.html#ompXXX <https://openmp.llvm.org/docs/remarks/OptimizationRemarks.html#ompXXX>`_ where `XXX` is
the three digit code shown in the tag.


----


.. _omp100:
.. _omp_no_external_caller_in_target_region:

`[OMP100]` Potentially unknown OpenMP target region caller
----------------------------------------------------------

A function remark that indicates the function, when compiled for a GPU, is
potentially called from outside the translation unit. Note that a remark is
only issued if we tried to perform an optimization which would require us to
know all callers on the GPU.

To facilitate OpenMP semantics on GPUs we provide a runtime mechanism through
which the code that makes up the body of a parallel region is shared with the
threads in the team. Generally we use the address of the outlined parallel
region to identify the code that needs to be executed. If we know all target
regions that reach the parallel region we can avoid this function pointer
passing scheme and often improve the register usage on the GPU. However, If a
parallel region on the GPU is in a function with external linkage we may not
know all callers statically. If there are outside callers within target
regions, this remark is to be ignored. If there are no such callers, users can
modify the linkage and thereby help optimization with a `static` or
`__attribute__((internal))` function annotation. If changing the linkage is
impossible, e.g., because there are outside callers on the host, one can split
the function into an external visible interface which is not compiled for
the target and an internal implementation which is compiled for the target
and should be called from within the target region.
