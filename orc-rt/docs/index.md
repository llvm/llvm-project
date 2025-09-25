# LLVM ORC Runtime

## Overview

The ORC runtime provides executor-side support code for the LLVM ORC APIs.

```{eval-rst}
.. toctree::
   :titlesonly:

   Building-orc-rt
```

### Current Status

The ORC Runtime is a new, experimental project. It is being actively developed,
and neither the ABI nor API are stable. LLVM ORC API clients should be careful
to use an ORC Runtime from the same build as their LLVM ORC libraries.

### Platform and Compiler Support

* TODO

The following minimum compiler versions are strongly recommended.

* Clang 16 and above

Anything older *may* work.

### Notes and Known Issues

* TODO

## Getting Involved

First please review our
[Developer's Policy](https://llvm.org/docs/DeveloperPolicy.html) and
[Getting started with LLVM](https://llvm.org/docs/GettingStarted.html).

**Bug Reports**

If you think you've found a bug in the ORC Runtime, please report it using
the [LLVM bug tracker](https://github.com/llvm/llvm-project/labels/orc-rt/).
Please use the tag "orc-rt" for new threads.

**Patches**

If you want to contribute a patch to th ORC runtime, please start by reading
the LLVM
[documentation about contributing](https://www.llvm.org/docs/Contributing.html).

**Discussion and Questions**

* TODO
