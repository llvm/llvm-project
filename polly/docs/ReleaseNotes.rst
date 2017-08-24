=================
Release Notes 5.0
=================

In Polly 5 the following important changes have been incorporated.

-----------------------------------
Robustness testing: AOSP and FFMPEG
-----------------------------------

Polly can now compile all of Android. While most of Android is not the primary
target of polyhedral data locality optimizations, Android provides us with a
large and diverse set of robustness tests.  Our new `nightly build bot
<http://lab.llvm.org:8011/builders/aosp-O3-polly-before-vectorizer-unprofitable>`_
ensures we do not regress.

Polly also successfully compiles `FFMPEG <http://fate.ffmpeg.org/>`_ and
obviously the `LLVM test suite
<http://lab.llvm.org:8011/console?category=polly>`_.

---------------------------------------------------------
C++ bindings for isl math library improve maintainability
---------------------------------------------------------

In the context of `Polly Labs <pollylabs.org>`_, a new set of C++ bindings was
developed for the isl math library. Thanks to the new isl C++ interface there
is no need for manual memory management any more and programming with integer
sets became easier in general.

Today::

    void isDiffEmptyOrUnionTheUniverse(isl::set S1, isl::set S2) {
      isl::set Difference = S1.subtract(S2);
      isl::set Union = S1.unite(S2);

      if (Difference.is_empty())
        return true;

      if (Union.is_universe())
        return true;

      return false;
    }

Before::

    void isDiffEmptyOrUnionTheUniverse(__isl_take isl_set S1,
                                       __isl_take isl_set S2) {
      isl_set *Difference = isl_set_subtract(isl_set_copy(S1),
                                             isl_set_copy(S2));

      isl_set *Union = isl_set_union(S1, S2);

      isl_bool IsEmpty = isl_set_is_empty(Difference);
      isl_set_free(Difference);

      if (IsEmpty == isl_bool_error)
        llvm_unreachable();

      if (IsEmpty)
        return true;

      isl_bool IsUniverse = isl_set_is_Universe(Union);
      isl_set_free(Union);

      if (IsUniverse == isl_bool_error)
        llvm_unreachable();

      if (IsUniverse)
        return true;

      return false;
    }

--------------------------
Improved Polly Diagnostics
--------------------------

Polly now uses the LLVM OptimizationDiagnosticInfo API for emitting diagnostic remarks.
This allows Polly remarks to appear in the yaml optimization record when compiling
with the flag -fsave-optimization-record. This also allows Polly remarks to appear in the opt-viewer
tool, allowing for remarks to be viewed next to the source code, and sorted by hotness.

--------------------------
Polly-ACC works with Julia
--------------------------

Polly can now offload Julia to GPUs. This feature is enabled by setting the
USE_POLLY_ACC variable to 1, i.e. USE_POLLY_ACC:=1.
