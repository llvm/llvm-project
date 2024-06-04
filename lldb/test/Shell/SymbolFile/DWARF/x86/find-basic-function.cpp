// REQUIRES: lld

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -gno-pubnames
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=BASE %s
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck --check-prefix=METHOD %s
// RUN: lldb-test symbols --name=foo --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL %s
// RUN: lldb-test symbols --name=_Z3fooi --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED %s
// RUN: lldb-test symbols --name=_ZN3bar3baz3fooEv --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED-NAMESPACE %s
// RUN: lldb-test symbols --name=_ZN4sbar3fooEi --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED-METHOD %s
// RUN: lldb-test symbols --name=foo --context=context --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=function %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s
//
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=foo --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=BASE %s
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck --check-prefix=METHOD %s
// RUN: lldb-test symbols --name=foo --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL %s
// RUN: lldb-test symbols --name=_Z3fooi --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED %s
// RUN: lldb-test symbols --name=_ZN3bar3baz3fooEv --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED-NAMESPACE %s
// RUN: lldb-test symbols --name=_ZN4sbar3fooEi --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED-METHOD %s
// RUN: lldb-test symbols --name=foo --context=context --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --compiler-context=Struct:sbar,Function:foo -language=c++ -find=function -function-flags=method %t | \
// RUN:   FileCheck --check-prefix=COMPILER-CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=function %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

// RUN: %clang %s -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --name=foo --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=BASE %s
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck --check-prefix=METHOD %s
// RUN: lldb-test symbols --name=foo --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL %s
// RUN: lldb-test symbols --name=_Z3fooi --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED %s
// RUN: lldb-test symbols --name=_ZN3bar3baz3fooEv --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED-NAMESPACE %s
// RUN: lldb-test symbols --name=_ZN4sbar3fooEi --find=function --function-flags=full %t | \
// RUN:   FileCheck --check-prefix=FULL-MANGLED-METHOD %s
// RUN: lldb-test symbols --name=foo --context=context --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=function %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

/// Test a per-module index built by lld.
// RUN: ld.lld --debug-names %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=function --function-flags=base %t | \
// RUN:   FileCheck --check-prefix=BASE %s

// NAMES: Name: .debug_names

// BASE: Found 4 functions:
// BASE-DAG: name = "foo()", mangled = "_Z3foov"
// BASE-DAG: name = "foo(int)", mangled = "_Z3fooi"
// BASE-DAG: name = "bar::foo()", mangled = "_ZN3bar3fooEv"
// BASE-DAG: name = "bar::baz::foo()", mangled = "_ZN3bar3baz3fooEv"

// METHOD: Found 3 functions:
// METHOD-DAG: name = "sbar::foo()", mangled = "_ZN4sbar3fooEv"
// METHOD-DAG: name = "sbar::foo(int)", mangled = "_ZN4sbar3fooEi"
// METHOD-DAG: name = "ffbar()::sbaz::foo()", mangled = "_ZZ5ffbarvEN4sbaz3fooEv"

// FULL: Found 0 functions:

// FULL-MANGLED: Found 1 functions:
// FULL-MANGLED-DAG: name = "foo(int)", mangled = "_Z3fooi"

// FULL-MANGLED-NAMESPACE: Found 1 functions:
// FULL-MANGLED-NAMESPACE-DAG: name = "bar::baz::foo()", mangled = "_ZN3bar3baz3fooEv"

// FULL-MANGLED-METHOD: Found 1 functions:
// FULL-MANGLED-METHOD-DAG: name = "sbar::foo(int)", mangled = "_ZN4sbar3fooEi"

// CONTEXT: Found 1 functions:
// CONTEXT-DAG: name = "bar::foo()", mangled = "_ZN3bar3fooEv", decl_context = {Namespace(bar)}

// COMPILER-CONTEXT: Found 2 functions:
// COMPILER-CONTEXT-DAG: name = "sbar::foo()", mangled = "_ZN4sbar3fooEv"
// COMPILER-CONTEXT-DAG: name = "sbar::foo(int)", mangled = "_ZN4sbar3fooEi"

// EMPTY: Found 0 functions:

void foo() {}
void foo(int) {}

namespace bar {
int context;
void foo() {}
namespace baz {
void foo() {}
} // namespace baz
} // namespace bar

struct foo {};
void fbar(struct foo) {}

void Foo() {}

struct sbar {
  void foo();
  static void foo(int);
};
void sbar::foo() {}
void sbar::foo(int) {}

void ffbar() {
  struct sbaz {
    void foo() {}
  };
  sbaz a;
  a.foo();
}

extern "C" void _start() {}
