# ORC Runtime Regression Tests

End-to-end tests of the ORC runtime. Run them with `ninja check-orc-rt`. For
tests of individual APIs, write unit tests under `test/unit` instead.

## Where does my test go?

Place a test by the *question it asks*, not by the kind of input file it uses:

* **`jit-free-foundations/`**: Tests that don't compile, link, or run any
  JIT'd code, e.g. tests of ogre's command line, process info, logging, and
  connection lifecycle.
* **`languages/<language>/`** (e.g. `languages/c/`): Tests that a
  source-language construct behaves correctly when JIT'd.
* **`object-formats/<format>/<arch>/`** (e.g. `object-formats/mach-o/arm64/`):
  Tests that a specific object format feature (e.g. a relocation, section, or
  directive) is handled correctly. These are written in assembly.

The `jit-free-foundations/` rule is enforced: the `%{cc}`, `%{cxx}`, `%{mc}`,
and `%{obj-jit}` substitutions are unavailable there.

## Writing tests that run JIT'd code

A typical test:

```c
// Check that JIT'd code can load from static data defined in the same object.
//
// Stresses: fixups for code addressing data in the same object.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

static int Data = 42;

int main(void) {
  // Volatile load, to prevent constant propagation of the initializer.
  if (*(volatile int *)&Data != 42)
    return 1;
  return 0;
}
```

Conventions:

* **Test one construct per test.** Linker and loader bugs usually crash the
  JIT'd program rather than return a wrong value, and a crash identifies only
  the test, not the check that crashed. Keep each test to a single source
  construct (in language tests) or object format feature (in object format
  tests), so that the name of a failing test says what broke. Tests of how
  constructs interact are the exception, and should say so.
* **Say what the test checks, and what it stresses.** Start with a one-line
  "Check that ..." summary of the behavior under test, followed by a
  "Stresses:" line naming the linker or loader behavior that it depends on.
  This tells someone triaging a failure where to look. Be specific: name the
  fixups, sections, or optimizations involved.
* **Return 0 on success.** If a test makes more than one check of its
  construct (e.g. that data reads correctly, then that it can be written),
  return a distinct non-zero value from each, so that a failure reports which
  check failed (`JIT result: 2`).
* **Check behavior, not object contents.** Language tests should keep passing
  if the compiler changes how a construct is lowered, and only fail if the
  new lowering doesn't work under the runtime.
* **Write freestanding sources.** Don't include system headers; declare any
  library functions you need yourself. This keeps tests usable when
  cross-compiling for targets without a sysroot.

For tests compiled from source:

* **Test at `-O0` and `-O2`,** with RUN lines for each. Most JIT'd code is
  optimized, and optimization changes how constructs are lowered. If a
  construct can only be tested at one level, say which, and why.
* **Keep constructs alive under optimization, but no more.** An optimizing
  compiler may fold away a construct (e.g. replace a load of static data with
  its initializer), leaving nothing for the JIT linker to handle. Prefer
  making the construct opaque to the compiler (e.g. defining data in another
  object, or with external linkage) over constraining it with `volatile` or
  `noinline`, which may hide the optimized lowering we want to test. When a
  test needs a barrier, apply the weakest one to the construct under test
  only, and name the optimization it suppresses (e.g. `// Volatile load, to
  prevent store-to-load forwarding.`).
* **Don't assume a particular compiler.** `%{cc}` and `%{cxx}` may be
  user-supplied (see below). Say what a compiler *may* do, not what it will
  do, and use generic optimization names rather than those of a particular
  compiler's passes.

### Tests with more than one source file

Tests that need more than one source file (e.g. to link several objects) are
`.test` files that hold their sources, split out with `split-file`:

```
# Check that JIT'd code can load from global data defined in another object.
#
# Stresses: fixups for code addressing data defined in another object, which
# the compiler usually addresses through the GOT.

REQUIRES: split-file
RUN: rm -rf %t && split-file %s %t
RUN: %{cc} -O0 -c -o %t/def.O0.o %t/def.c
RUN: %{cc} -O0 -c -o %t/main.O0.o %t/main.c
RUN: %{obj-jit} -show-jit-result %t/main.O0.o %t/def.O0.o | FileCheck %s
RUN: %{cc} -O2 -c -o %t/def.O2.o %t/def.c
RUN: %{cc} -O2 -c -o %t/main.O2.o %t/main.c
RUN: %{obj-jit} -show-jit-result %t/main.O2.o %t/def.O2.o | FileCheck %s

CHECK: JIT result: 0

#--- def.c
int Data = 42;

#--- main.c
extern int Data;

int main(void) {
  if (Data != 42)
    return 1;
  return 0;
}
```

### Substitutions

* **`%{cc}`**: Compiles C for the runtime's target.
* **`%{cxx}`**: Compiles C++ for the runtime's target.
* **`%{mc}`**: Assembles its input into an object file for the runtime's
  target, using `llvm-mc`. Unlike `%{cc}` and `%{cxx}`, this can't be
  overridden, so object format tests always check the runtime against the
  same assembler.
* **`%{obj-jit}`**: Links and runs its inputs under ogre, using `llvm-jitlink`
  as the controller. Pass `-show-jit-result` to print `JIT result: <value>`.
* **`%{ogre}`**: Path to the ogre executable.

Don't spell out a target triple or connection method in a test unless that's
what the test is about: the substitutions provide these, so that the same
tests can be run with different compilers and targets. Flags that select a
feature under test (e.g. `-mattr=`, `-mcmodel=`, `-fPIC`) are fine, but gate
the test on the target architecture and CPU features they require.

`%{cc}` and `%{cxx}` accept clang-style options, and tests must restrict
themselves to this subset: `-c`, `-o`, `-I`, `-D`, `-O<n>`, `-g`, `-std=`,
`-x`, `-fexceptions`, and `-fno-exceptions`. If a test needs another option,
add it to this list, so that anyone writing a wrapper for another compiler
knows which options to support.

### Features

Directories and tests gate on lit features describing the build, rather than
the host:

* **`orc-rt-cc`**: `%{cc}` is usable.
* **`orc-rt-cxx`**: `%{cxx}` is usable.
* **`llvm-mc`**: `%{mc}` is usable.
* **`llvm-jitlink`**: `%{obj-jit}` is usable.
* **`split-file`**: `split-file` is usable.
* **`target-arch=<arch>`**: The runtime's target architecture (`arm64` and
  `aarch64` are aliases).
* **`target-object-format=<coff|elf|mach-o>`**: The runtime's target object
  format.
* **`orc-rt-log-backend-<backend>`**, **`orc-rt-log-level-<level>`**: The
  runtime was built with the given logging backend / level.

`languages/c/` requires `orc-rt-cc` and `llvm-jitlink`, so tests there don't
need to repeat those requirements. Likewise, `object-formats/` requires
`llvm-mc` and `llvm-jitlink`, and each `<format>/<arch>/` subdirectory
requires the matching `target-object-format=` and `target-arch=` features.

## Running with a different compiler

By default `%{cc}` and `%{cxx}` are the `clang` and `clang++` in the LLVM
tools directory, i.e. `bin/` in the LLVM build given by `-DLLVM_BINARY_DIR`
(or `-DORC_RT_LLVM_TOOLS_DIR`). If no clang is found, the C tests are reported
as unsupported.

To test the runtime against another compiler's output, pass:

```
llvm-lit --param orc-rt-cc=<cc> --param orc-rt-cxx=<cxx> \
  <path-to-build>/orc-rt/test/regression
```

The compiler must accept the clang-style options above and must target the
runtime's target without being told to. Write a wrapper script if your
compiler needs different options. Treat results from other compilers as
reports rather than expectations: don't add `XFAIL`s for them in-tree.

## Tips

To print each test's result instead of a progress bar:

```
LIT_OPTS="--no-progress-bar --print-result-after=all" ninja check-orc-rt
```
