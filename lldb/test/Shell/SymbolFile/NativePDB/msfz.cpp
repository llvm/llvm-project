//===-- msfz.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
// REQUIRES: lld, x86, zstd

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe \
// RUN:     -pdb:%t.pdb /pdbmsfz:1
// RUN: llvm-pdbutil dump -summary %t.pdb | FileCheck --check-prefix=SUMMARY %s
// RUN: lldb-test symbols --find=function --name=main --function-flags=full \
// RUN:     %t.exe | FileCheck --check-prefix=SYMBOLS %s
// RUN: lldb-test symbols --verify %t.exe | FileCheck --check-prefix=LINES %s

int main() {
  return 0;
}

// SUMMARY: Container: MSFZ
// SYMBOLS: Found 1 functions:
// SYMBOLS: Function{{.*}}, demangled = main
// SYMBOLS: LineEntry: {{.*}}msfz.cpp:
// LINES: The line table contains {{[1-9][0-9]*}} entries.
// LINES: The symbol information is verified.
