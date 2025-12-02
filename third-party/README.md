# Third-party LLVM dependencies

This directory contains third-party dependencies used in various components of LLVM.
Integrating a new third-party dependency generally requires it to be licensed under
the Apache-with-LLVM-exception license. For integrating code under other licenses,
please follow the process explained in the [LLVM Developer Policy](https://llvm.org/docs/DeveloperPolicy.html#copyright-license-and-patents).

In particular, due to its non-LLVM license, the Boost.Math third-party dependency
can exclusively be used within the libc++ compiled library as discussed in [this RFC](https://discourse.llvm.org/t/rfc-libc-taking-a-dependency-on-boost-math-for-the-c-17-math-special-functions).
Do not use it in other parts of LLVM without prior discussion with the LLVM Board
(and update this documentation).
