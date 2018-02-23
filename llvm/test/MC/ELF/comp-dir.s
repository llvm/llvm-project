// REQUIRES: shell
// RUN: llvm-mc -triple=x86_64-linux-unknown -g -fdebug-compilation-dir=/test/comp/dir %s -filetype=obj -o %t.o
// RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s

// CHECK: DW_AT_comp_dir [DW_FORM_string] ("{{([A-Za-z]:.*)?}}/test/comp/dir")

// RUN: mkdir -p %t.foo
// RUN: ln -sf %t.foo %t.bar
// RUN: cd %t.foo
// RUN: env PWD=%t.bar llvm-mc -triple=x86_64-linux-unknown -g %s -filetype=obj -o %t.o
// RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck --check-prefix=PWD %s
// PWD: DW_AT_comp_dir [DW_FORM_string] ("{{.*}}.bar")


f:
  nop
