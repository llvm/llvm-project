// RUN: llvm-mc -arch=hexagon -filetype=obj -o - %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

{ r7 = #-1
  r6 = #-1 }
// CHECK: { r7 = #-1; r6 = #-1 }

{ p0 = r0
  if (p0.new) r7 = #0
  if (!p0.new) r7 = #0 }
// CHECK: if (p0.new) r7 = #0; if (!p0.new) r7 = #0
