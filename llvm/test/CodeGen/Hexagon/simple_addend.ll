; RUN: llc -mtriple=hexagon -filetype=obj -o - < %s | llvm-readobj -r - | FileCheck %s

declare void @bar(i32);

define void @foo(i32 %a) {
  %b = mul i32 %a, 3
  call void @bar(i32 %b)
  ret void
}
; The scalar multiply on SLOT2/SLOT3 is a multi-cycle (TC3x) producer whose
; write reaches the register file too late to be observed by a co-packetized
; call, so the multiply and the call must live in separate packets.
; CHECK:     0x8 R_HEX_B22_PCREL bar 0x0
