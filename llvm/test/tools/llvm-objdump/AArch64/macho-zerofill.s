// RUN: llvm-mc < %s -triple aarch64-macho -filetype=obj | llvm-objdump -triple=aarch64-- -D - | FileCheck %s


// Check that we don't print garbage when we dump zerofill sections.

.zerofill __DATA,__common,_data64unsigned,472,3
// CHECK: Disassembly of section __DATA,__common:
// CHECK: <ltmp1>:
// CHECK-NEXT: ...
