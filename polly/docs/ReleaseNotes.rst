============================
Release Notes 5.0 (upcoming)
============================

In Polly 5 the following important changes have been incorporated.

.. warning::

  These releaes notes are for the next release of Polly and describe
  the new features that have recently been committed to our development
  branch.

- Change ...

--------
Internal
--------

C++ bindings
------------

Polly uses now C++ bindings when using isl which remove the need for manual
memory management.

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
