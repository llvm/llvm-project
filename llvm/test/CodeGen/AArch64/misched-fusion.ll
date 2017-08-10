; RUN: llc -o - %s -mattr=+arith-bcc-fusion | FileCheck --check-prefix=FUSEBCC %s
; RUN: llc -o - %s -mattr=+arith-cbz-fusion | FileCheck --check-prefix=FUSECBZ %s
; RUN: llc -o - %s -mcpu=cyclone            | FileCheck --check-prefix=FUSEBCC --check-prefix=FUSECBZ %s

target triple = "aarch64-unknown"

declare void @foobar(i32 %v0, i32 %v1)

; Make sure cmp is scheduled in front of bcc
; FUSEBCC-LABEL: test_cmp_bcc:
; FUSEBCC: cmp {{w[0-9]+}}, #13
; FUSEBCC-NEXT: b.ne {{.?LBB[0-9_]+}}
define void @test_cmp_bcc(i32 %a0, i32 %a1) {
entry:
  %cond = icmp eq i32 %a0, 13
  %v1 = add i32 %a1, 7
  br i1 %cond, label %if, label %exit

if:
  call void @foobar(i32 %v1, i32 %a0)
  br label %exit

exit:
  call void @foobar(i32 %a0, i32 %v1)
  ret void
}

; Make sure sub is scheduled in front of cbnz
; FUSECBZ-LABEL: test_sub_cbz:
; FUSECBZ: subs [[SUBRES:w[0-9]+]], w0, #13
; FUSECBZ: b.ne {{.?LBB[0-9_]+}}
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
