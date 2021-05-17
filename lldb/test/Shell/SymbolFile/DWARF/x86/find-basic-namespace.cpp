// REQUIRES: lld

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -gno-pubnames
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=namespace %t | \
// RUN:   FileCheck --check-prefix=FOO %s
// RUN: lldb-test symbols --name=foo --find=namespace --context=context %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=namespace %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s
//
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=foo --find=namespace %t | \
// RUN:   FileCheck --check-prefix=FOO %s
// RUN: lldb-test symbols --name=foo --find=namespace --context=context %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=namespace %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

// RUN: %clang %s -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --name=foo --find=namespace %t | \
// RUN:   FileCheck --check-prefix=FOO %s
// RUN: lldb-test symbols --name=foo --find=namespace --context=context %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=namespace %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

// NAMES: Name: .debug_names

// FOO: Found namespace: foo

// CONTEXT: Found namespace: bar::foo

// EMPTY: Namespace not found.

namespace foo {
int X;
}

namespace bar {
int context;
namespace foo {
int X;
}
} // namespace bar

extern "C" void _start() {}
