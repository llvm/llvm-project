// REQUIRES: lld

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -gno-pubnames
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=type %t | \
// RUN:   FileCheck --check-prefix=NAME %s
// RUN: lldb-test symbols --name=::foo --find=type %t | \
// RUN:   FileCheck --check-prefix=EXACT %s
// RUN: lldb-test symbols --name=foo --context=context --find=type %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=type %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s
//
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=foo --find=type %t | \
// RUN:   FileCheck --check-prefix=NAME %s
// RUN: lldb-test symbols --name=::foo --find=type %t | \
// RUN:   FileCheck --check-prefix=EXACT %s
// RUN: lldb-test symbols --name=foo --context=context --find=type %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=type %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

// RUN: %clang %s -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --name=foo --find=type %t | \
// RUN:   FileCheck --check-prefix=NAME %s
// RUN: lldb-test symbols --name=::foo --find=type %t | \
// RUN:   FileCheck --check-prefix=EXACT %s
// RUN: lldb-test symbols --name=foo --context=context --find=type %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=not_there --find=type %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

// NAMES: Name: .debug_names

// EMPTY: Found 0 types:
// NAME: Found 4 types:
// EXACT: Found 1 types:
// CONTEXT: Found 1 types:
struct foo { };
// NAME-DAG: name = "foo", {{.*}} decl = find-basic-type.cpp:[[@LINE-1]]
// EXACT-DAG: name = "foo", {{.*}} decl = find-basic-type.cpp:[[@LINE-2]]

namespace bar {
int context;
struct foo {};
// NAME-DAG: name = "foo", {{.*}} decl = find-basic-type.cpp:[[@LINE-1]]
// CONTEXT-DAG: name = "foo", {{.*}} decl = find-basic-type.cpp:[[@LINE-2]]
namespace baz {
struct foo {};
// NAME-DAG: name = "foo", {{.*}} decl = find-basic-type.cpp:[[@LINE-1]]
}
}

struct sbar {
  struct foo {};
// NAME-DAG: name = "foo", {{.*}} decl = find-basic-type.cpp:[[@LINE-1]]
};

struct foobar {};

struct Foo {};

extern "C" void _start(foo, bar::foo, bar::baz::foo, sbar::foo, foobar, Foo) {}
