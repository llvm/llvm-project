// REQUIRES: lld

// RUN: %clang -g -c -o %t-1.o --target=x86_64-pc-linux -gno-pubnames %s
// RUN: %clang -g -c -o %t-2.o --target=x86_64-pc-linux -gno-pubnames %S/Inputs/find-variable-file-2.cpp
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s

// Run the same test with split-dwarf. This is interesting because the two
// split compile units will have the same offset (0).
// RUN: %clang -g -c -o %t-1.o --target=x86_64-pc-linux -gsplit-dwarf %s
// RUN: %clang -g -c -o %t-2.o --target=x86_64-pc-linux -gsplit-dwarf %S/Inputs/find-variable-file-2.cpp
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s

// RUN: %clang -c -o %t-1.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames %s
// RUN: %clang -c -o %t-2.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames %S/Inputs/find-variable-file-2.cpp
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s

// Run the same test with split dwarf and pubnames to check whether we can find
// the compile unit using the name index if it is split.
// RUN: %clang -c -o %t-1.o --target=x86_64-pc-linux -gdwarf-5 -gsplit-dwarf -gpubnames %s
// RUN: %clang -c -o %t-2.o --target=x86_64-pc-linux -gdwarf-5 -gsplit-dwarf -gpubnames %S/Inputs/find-variable-file-2.cpp
// RUN: %clang -c -o %t-3.o --target=x86_64-pc-linux -gdwarf-5 -gsplit-dwarf -gpubnames %S/Inputs/find-variable-file-3.cpp
// RUN: ld.lld %t-1.o %t-2.o %t-3.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s
// RUN: lldb-test symbols --file=find-variable-file-3.cpp --find=variable \
// RUN:   --name=notexists %t

// NAMES: Name: .debug_names

// ONE: Found 1 variables:
namespace one {
int foo;
// ONE-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-variable-file.cpp:[[@LINE-1]]
} // namespace one

extern "C" void _start() {}

// TWO: Found 1 variables:
// TWO-DAG: name = "foo", {{.*}} decl = find-variable-file-2.cpp:2
