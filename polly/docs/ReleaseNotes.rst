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

Update of the isl math library
------------------------------

We imported the latest version of the isl math library into Polly.

