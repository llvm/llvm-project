
    README for the OpenMP Tooling Interface Testing Library (libomptest)
    ====================================================================

Introduction
============
TBD

Limitations
===========
Currently, there are some peculiarities which have to be kept in mind when using
this library:

## Callbacks
  * It is not possible to e.g. test non-EMI -AND- EMI callbacks within the same
    test file. Reason: all testsuites will share the initialization and
    therefore the registered callbacks.
  * It is not possible to check for device initialization and/or load callbacks
    more than once per test file. The first testcase being run, triggering these
    will be the only testcase that is able to check for these callbacks. This is
    because after that, the device remains initialized.
  * It is not possible to check for device finalization callbacks, as libomptest
    is un-loaded before this callback occurs. Same holds true for the final
    ThreadEnd event(s).
