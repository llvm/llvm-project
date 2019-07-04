llvm-build - LLVM Project Build Utility
=======================================

.. program:: llvm-build

SYNOPSIS
--------

**llvm-build** [*options*]

DESCRIPTION
-----------

**llvm-build** is a tool for working with LLVM projects that use the LLVMBuild
system for describing their components.

At heart, **llvm-build** is responsible for loading, verifying, and manipulating
the project's component data. The tool is primarily designed for use in
implementing build systems and tools which need access to the project structure
information.

OPTIONS
-------

**-h**, **--help**

 Print the builtin program help.

**--source-root**\ =\ *PATH*

 If given, load the project at the given source root path. If this option is not
 given, the location of the project sources will be inferred from the location of
 the **llvm-build** script itself.

**--print-tree**

 Print the component tree for the project.

**--write-library-table**

 Write out the C++ fragment which defines the components, library names, and
 required libraries. This C++ fragment is built into llvm-config|llvm-config
 in order to provide clients with the list of required libraries for arbitrary
 component combinations.

**--write-llvmbuild**

 Write out new *LLVMBuild.txt* files based on the loaded components. This is
 useful for auto-upgrading the schema of the files. **llvm-build** will try to a
 limited extent to preserve the comments which were written in the original
 source file, although at this time it only preserves block comments that precede
 the section names in the *LLVMBuild* files.

**--write-cmake-fragment**

 Write out the LLVMBuild in the form of a CMake fragment, so it can easily be
 consumed by the CMake based build system. The exact contents and format of this
 file are closely tied to how LLVMBuild is integrated with CMake, see LLVM's
 top-level CMakeLists.txt.

**--write-make-fragment**

 Write out the LLVMBuild in the form of a Makefile fragment, so it can easily be
 consumed by a Make based build system. The exact contents and format of this
 file are closely tied to how LLVMBuild is integrated with the Makefiles, see
 LLVM's Makefile.rules.

**--llvmbuild-source-root**\ =\ *PATH*

 If given, expect the *LLVMBuild* files for the project to be rooted at the
 given path, instead of inside the source tree itself. This option is primarily
 designed for use in conjunction with **--write-llvmbuild** to test changes to
 *LLVMBuild* schema.

EXIT STATUS
-----------

**llvm-build** exits with 0 if operation was successful. Otherwise, it will exist
with a non-zero value.
