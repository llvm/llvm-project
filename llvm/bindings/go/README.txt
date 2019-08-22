This directory contains LLVM bindings for the Go programming language
(http://golang.org).

Prerequisites
-------------

* Go 1.2+.
* CMake (to build LLVM).

Using the bindings
------------------

The package path "llvm.org/llvm/bindings/go/llvm" can be used to
import the latest development version of LLVM from SVN. Paths such as
"llvm.org/llvm.v36/bindings/go/llvm" refer to released versions of LLVM.

It is recommended to use the "-d" flag with "go get" to download the
package or a dependency, as an additional step is required to build LLVM
(see "Building LLVM" below).

Building LLVM
-------------

The script "build.sh" in this directory can be used to build LLVM and prepare
it to be used by the bindings. If you receive an error message from "go build"
like this:

    ./analysis.go:4:84: fatal error: llvm-c/Analysis.h: No such file or directory
     #include <llvm-c/Analysis.h> // If you are getting an error here read bindings/go/README.txt

or like this:

    ./llvm_dep.go:5: undefined: run_build_sh

it means that LLVM needs to be built or updated by running the script.

    $ $GOPATH/src/llvm.org/llvm/bindings/go/build.sh

Any command line arguments supplied to the script are passed to LLVM's CMake
build system. A good set of arguments to use during development are:

    $ $GOPATH/src/llvm.org/llvm/bindings/go/build.sh -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=host -DBUILD_SHARED_LIBS=ON

Note that CMake keeps a cache of build settings so once you have built
LLVM there is no need to pass these arguments again after updating.

Alternatively, you can build LLVM yourself, but you must then set the
CGO_CPPFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS environment variables:

    $ export CGO_CPPFLAGS="`/path/to/llvm-build/bin/llvm-config --cppflags`"
    $ export CGO_CXXFLAGS=-std=c++14
    $ export CGO_LDFLAGS="`/path/to/llvm-build/bin/llvm-config --ldflags --libs --system-libs all`"
    $ go build -tags byollvm

If you see a compilation error while compiling your code with Go 1.9.4 or later as follows,

    go build llvm.org/llvm/bindings/go/llvm: invalid flag in #cgo LDFLAGS: -Wl,-headerpad_max_install_names

you need to setup $CGO_LDFLAGS_ALLOW to allow a compiler to specify some linker options:

    $ export CGO_LDFLAGS_ALLOW='-Wl,(-search_paths_first|-headerpad_max_install_names)'
