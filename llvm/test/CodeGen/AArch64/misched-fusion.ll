; RUN: llc -o - %s -mattr=+macroop-fusion,+use-postra-scheduler | FileCheck %s
; RUN: llc -o - %s -mcpu=cyclone | FileCheck %s

target triple = "arm64-apple-ios"

declare void @foobar(i32 %v0, i32 %v1)

; Make sure sub is scheduled in front of cbnz
; CHECK-LABEL: test_sub_cbz:
; CHECK: add w[[ADDRES:[0-9]+]], w1, #7
; CHECK: sub w[[SUBRES:[0-9]+]], w0, #13
; CHECK-NEXT: cbnz w[[SUBRES]], [[SKIPBLOCK:LBB[0-9_]+]]
; CHECK: mov [[REGTY:[x,w]]]0, [[REGTY]][[ADDRES]]
; CHECK: mov [[REGTY]]1, [[REGTY]][[SUBRES]]
; CHECK: bl _foobar
; CHECK: [[SKIPBLOCK]]:
; CHECK: mov [[REGTY]]0, [[REGTY]][[SUBRES]]
; CHECK: mov [[REGTY]]1, [[REGTY]][[ADDRES]]
; CHECK: bl _foobar
define void @test_sub_cbz(i32 %a0, i32 %a1) {
entry:
  ; except for the fusion opportunity the sub/add should be equal so the
  ; scheduler would leave them in source order if it weren't for the scheduling
  %v0 = sub i32 %a0, 13
  %cond = icmp eq i32 %v0, 0
  %v1 = add i32 %a1, 7
  br i1 %cond, label %if, label %exit

if:
  call void @foobar(i32 %v1, i32 %v0)
  br label %exit

exit:
  call void @foobar(i32 %v0, i32 %v1)
  ret void
}
