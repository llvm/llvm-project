; RUN: llc -march=hexagon -filetype=obj -o - < %s | llvm-readobj -r - | FileCheck %s

declare void @bar(i32);

define void @foo(i32 %a) {
  %b = mul i32 %a, 3
  call void @bar(i32 %b)
  ret void
}
; CHECK:     0x4 R_HEX_B22_PCREL bar 0x4
