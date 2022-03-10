OpenMP Optimization Remarks
===========================

The :doc:`OpenMP-Aware optimization pass </optimizations/OpenMPOpt>` is able to
generate compiler remarks for performed and missed optimisations. To emit them,
pass these options to the Clang invocation: ``-Rpass=openmp-opt
-Rpass-analysis=openmp-opt -Rpass-missed=openmp-opt``.  For more information and
features of the remark system, consult the clang documentation:

+ `Clang options to emit optimization reports <https://clang.llvm.org/docs/UsersManual.html#options-to-emit-optimization-reports>`_
+ `Clang diagnostic and remark flags <https://clang.llvm.org/docs/ClangCommandLineReference.html#diagnostic-flags>`_
+ The `-foptimization-record-file flag
  <https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-foptimization-record-file>`_
  and the `-fsave-optimization-record flag
  <https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang1-fsave-optimization-record>`_


OpenMP Remarks
--------------

.. toctree::
   :hidden:
   :maxdepth: 1

   OMP100
   OMP101
   OMP102
   OMP110
   OMP111
   OMP112
   OMP113
   OMP120
   OMP121
   OMP130
   OMP131
   OMP132
   OMP133
   OMP140
   OMP150
   OMP160
   OMP170
   OMP180

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Diagnostics Number
     - Diagnostics Kind
     - Diagnostics Description
   * - :ref:`OMP100 <omp100>`
     - Analysis
     - Potentially unknown OpenMP target region caller.
   * - :ref:`OMP101 <omp101>`
     - Analysis
     - Parallel region is used in unknown / unexpected ways. Will not attempt to
       rewrite the state machine.
   * - :ref:`OMP102 <omp102>`
     - Analysis
     - Parallel region is not called from a unique kernel. Will not attempt to
       rewrite the state machine.
   * - :ref:`OMP110 <omp110>`
     - Optimization
     - Moving globalized variable to the stack.
   * - :ref:`OMP111 <omp111>`
     - Optimization
     - Replaced globalized variable with X bytes of shared memory.
   * - :ref:`OMP112 <omp112>`
     - Missed
     - Found thread data sharing on the GPU. Expect degraded performance due to
       data globalization.
   * - :ref:`OMP113 <omp113>`
     - Missed
     - Could not move globalized variable to the stack. Variable is potentially
       captured in call. Mark parameter as `__attribute__((noescape))` to
       override.
   * - :ref:`OMP120 <omp120>`
     - Optimization
     - Transformed generic-mode kernel to SPMD-mode.
   * - :ref:`OMP121 <omp121>`
     - Analysis
     - Value has potential side effects preventing SPMD-mode execution. Add
       `__attribute__((assume(\"ompx_spmd_amenable\")))` to the called function
       to override.
   * - :ref:`OMP130 <omp130>`
     - Optimization
     - Removing unused state machine from generic-mode kernel.
   * - :ref:`OMP131 <omp131>`
     - Optimization
     - Rewriting generic-mode kernel with a customized state machine.
   * - :ref:`OMP132 <omp132>`
     - Analysis
     - Generic-mode kernel is executed with a customized state machine that
       requires a fallback.
   * - :ref:`OMP133 <omp133>`
     - Analysis
     - Call may contain unknown parallel regions. Use
       `__attribute__((assume("omp_no_parallelism")))` to override.
   * - :ref:`OMP140 <omp140>`
     - Analysis
     - Could not internalize function. Some optimizations may not be possible.
   * - :ref:`OMP150 <omp150>`
     - Optimization
     - Parallel region merged with parallel region at <location>.
   * - :ref:`OMP160 <omp160>`
     - Optimization
     - Removing parallel region with no side-effects.
   * - :ref:`OMP170 <omp170>`
     - Optimization
     - OpenMP runtime call <call> deduplicated.
   * - :ref:`OMP180 <omp180>`
     - Optimization
     - Replacing OpenMP runtime call <call> with <value>.
