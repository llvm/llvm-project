llgo
====

llgo is a Go (http://golang.org) frontend for LLVM, written in Go.

llgo is under active development. It compiles and passes most of the
standard library test suite and a substantial portion of the gc test suite,
but there are some corner cases that are known not to be handled correctly
yet. Nevertheless it can compile modestly substantial programs (including
itself; it is self hosting on x86-64 Linux).

Mailing list: https://groups.google.com/d/forum/llgo-dev

Supported platforms
-------------------

llgo is currently only supported on the x86-64 Linux platform. Contributions
that add support for other platforms are welcome.

There are two components which would need to be ported to new platforms: the
compiler and the runtime library. The compiler has little platform-specific
code; the most significant is in irgen/cabi.go. The main limiting factor
for new platforms is the runtime library in third_party/gofrontend/libgo,
which inherits some support for other platforms from the gc compiler's
runtime library, but this support tends to be incomplete.

Installation
------------

llgo requires:
* Go 1.3 or later.
* CMake 2.8.8 or later (to build LLVM).
* A modern C++ toolchain (to build LLVM).
  http://llvm.org/docs/GettingStarted.html#getting-a-modern-host-c-toolchain

Note that Ubuntu Precise is one Linux distribution which does not package
a sufficiently new CMake or C++ toolchain.

To build and install llgo:

    # Checkout LLVM:
    svn co http://llvm.org/svn/llvm-project/llvm/trunk /path/to/llvm

    # Checkout Clang:
    cd /path/to/llvm/tools
    svn co http://llvm.org/svn/llvm-project/cfe/trunk clang

    # Checkout llgo:
    svn co http://llvm.org/svn/llvm-project/llgo/trunk llgo

    # Build LLVM, Clang and llgo: (see also http://llvm.org/docs/CMake.html)
    mkdir /path/to/llvm-build
    cd /path/to/llvm-build
    cmake /path/to/llvm -DCMAKE_INSTALL_PREFIX=/path/to/llvm-inst
    make install

Running
-------

llgo-go is llgo's version of the "go" command. It has the same command line
interface as go, and works the same way, but it uses llgo to compile.

llgoi is an interactive REPL for Go. It supports expressions, statements, most
declarations and imports, including binary imports from the standard library
and source imports from $GOPATH. See docs/llgoi.rst for more information.

llgo is the compiler binary. It has a command line interface that is intended
to be compatible to a large extent with gccgo.

Contributing
------------

Changes to code outside the third_party directory should be contributed in
the normal way by sending patches to <llvm-commits@lists.llvm.org>.

Changes to code in the third_party directory must first be made in the
respective upstream project, from which they will be mirrored into the llgo
repository. See the script update_third_party.sh for the locations of the
upstream projects and details of how the mirroring works.

