=====================================================
LLVM parallel-libs Subproject Charter
=====================================================

----------------------------------------------
Description
----------------------------------------------
The LLVM open source project will contain a subproject named `parallel-libs`
which will host the development of libraries which are aimed at enabling
parallelism in code and which are also closely tied to compiler technology.
Examples of libraries suitable for hosting within the `parallel-libs`
subproject are runtime libraries and parallel math libraries. The initial
candidates for inclusion in this subproject are **StreamExecutor** and
**libomptarget** which would live in the `streamexecutor` and `libomptarget`
subdirectories of `parallel-libs`, respectively.

The `parallel-libs` project will host a collection of libraries where each
library may be dependent on other libraries from the project or may be
completely independent of any other libraries in the project. The rationale for
hosting independent libraries within the same subproject is that all libraries
in the project are providing related functionality that lives at the
intersection of parallelism and compiler technology. It is expected that some
libraries which initially began as independent will develop dependencies over
time either between existing libraries or by extracting common code that can be
used by each. One of the purposes of this subproject is to provide a working
space where such refactoring and code sharing can take place.

Libraries in the `parallel-libs` subproject may also depend on the LLVM core
libraries. This will be useful for avoiding duplication of code within the LLVM
project for common utilities such as those found in the LLVM support library.


----------------------------------------------
Requirements
----------------------------------------------
Libraries included in the `parallel-libs` subproject must strive to achieve the
following requirements:

1. Adhere to the LLVM coding standards.
2. Use the LLVM build and test infrastructure.
3. Be released under LLVM's license.


Coding standards
----------------
Libraries in `parallel-libs` will match the LLVM coding standards. For existing
projects being checked into the subproject as-is, an exception will be made
during the initial check-in, with the understanding that the code will be
promptly updated to follow the standards. Therefore, a three month grace period
will be allowed for new libraries to meet the LLVM coding standards.

Additional exceptions to strict adherence to the LLVM coding standards may be
allowed in certain other cases, but the reasons for such exceptions must be
discussed and documented on a case-by-case basis.


LLVM build and test infrastructure
----------------------------------
Using the LLVM build and test infrastructure currently means using `cmake` for
building, `lit` for testing, and `buildbot` for automating build and testing.
This project will follow the main LLVM project conventions here and track them
as they evolve.

Each subproject library will be able to build separately without a single,
unified cmake file, but each subproject libraries will also be integrated into
the LLVM build so they can be built directly from the top level of the LLVM
cmake infrastructure.


LLVM license
------------
For simplicity, the `parallel-libs` project will use the normal LLVM license.
While some runtime libraries use a dual license scheme in LLVM, we anticipate
the project removing the need for this eventually and in the interim follow the
simpler but still permissive license. Among other things, this makes it
straightforward for these libraries to re-use core LLVM libraries where
appropriate.


----------------------------------------------
Mailing List and Bugs
----------------------------------------------
Two mailing lists will be set up for the project:

1. parallel_libs-dev@lists.llvm.org for discussions among project developers, and
2. parallel_libs-commits@lists.llvm.org for patches and commits to the project.

Each subproject library will manage its own components in Bugzilla. So, for
example, there can be several Bugzilla components for different parts of
StreamExecutor, etc.
