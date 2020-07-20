==================
Release Notes 10.0
==================

In Polly 10 the following important changes have been incorporated.

- The mechanism that Polly uses to link itself statically into the opt,
  bugpoint and clang executables has been generalized such that it can
  be used by other pass plugins. An example plugin "Bye" has been added
  to LLVM to illustate the mechanism.

- Some ScopInfo methods that are only relevant during SCoP construction
  have been moved into the ScopBuilder class. In addition to making it
  clearer which methods can only be used during the construction phase,
  the refactoring helps shrinking the size of the public header
  ScopInfo.h.
