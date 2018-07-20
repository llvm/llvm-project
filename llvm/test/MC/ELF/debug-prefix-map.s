// RUN: mkdir -p %t.foo
// RUN: cp %s %t.foo/src.s
// RUN: cd %t.foo

// RUN: llvm-mc -triple=x86_64-linux-unknown -g src.s -filetype=obj -o out.o
// RUN: llvm-dwarfdump -v -debug-info out.o | FileCheck --check-prefix=NO_MAP %s

// RUN: llvm-mc -triple=x86_64-linux-unknown -g src.s -filetype=obj -o out.o -fdebug-prefix-map=%t.foo=src_root
// RUN: llvm-dwarfdump -v -debug-info out.o | FileCheck --check-prefix=MAP --implicit-check-not ".foo" %s

// RUN: llvm-mc -triple=x86_64-linux-unknown -g %t.foo/src.s -filetype=obj -o out.o -fdebug-prefix-map=%t.foo=/src_root
// RUN: llvm-dwarfdump -v -debug-info out.o | FileCheck --check-prefix=MAP_ABS --implicit-check-not ".foo" %s

f:
  nop

// NO_MAP: DW_AT_comp_dir [DW_FORM_string] ("{{.*}}.foo")

// MAP: DW_AT_name [DW_FORM_string] ("src.s")
// MAP: DW_AT_comp_dir [DW_FORM_string] ("src_root")
// MAP: DW_AT_decl_file [DW_FORM_data4] ("src_root{{(/|\\)+}}src.s")

// MAP_ABS: DW_AT_name [DW_FORM_string] ("{{(/|\\)+}}src_root{{(/|\\)+}}src.s")
// MAP_ABS: DW_AT_comp_dir [DW_FORM_string] ("{{(/|\\)+}}src_root")
// MAP_ABS: DW_AT_decl_file [DW_FORM_data4] ("{{(/|\\)+}}src_root{{(/|\\)+}}src.s")
