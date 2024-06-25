// UNSUPPORTED: system-windows

// Test that LLDB can follow DWO links produced by -gmodules debug
// info to find a type in a precompiled header.
//
// RUN: %clangxx_host -g -gmodules -fmodules -std=c99 -x c-header %S/Inputs/pch.h -g -c -o %t.pch
// RUN: %clangxx_host -g -gmodules -fmodules -std=c99 -x c -include-pch %t.pch %s -c -o %t.o
// RUN: %clangxx_host %t.o -o %t.exe
// RUN: lldb-test symbols -dump-clang-ast -find type --language=C99 \
// RUN:   -compiler-context 'AnyModule:*,ClassOrStruct:TypeFromPCH' %t.exe | FileCheck %s

anchor_t anchor;

int main(int argc, char **argv) { return 0; }

// CHECK: Found 1 type
// CHECK: "TypeFromPCH"
// CHECK: FieldDecl {{.*}} field
