; RUN: opt < %s -slsr -nary-reassociate -S | FileCheck %s
; RUN: opt < %s -slsr -S | opt -passes='nary-reassociate' -S | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

; foo((a + b) + c);
; foo((a + b * 2) + c);
; foo((a + b * 3) + c);
;   =>
; abc = (a + b) + c;
; foo(abc);
; ab2c = abc + b;
; foo(ab2c);
; ab3c = ab2c + b;
; foo(ab3c);
define void @nary_reassociate_after_slsr(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @nary_reassociate_after_slsr(
; PTX-LABEL: .visible .func nary_reassociate_after_slsr(
; PTX: ld.param.u32 [[b:%r[0-9]+]], [nary_reassociate_after_slsr_param_1];
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  call void @foo(i32 %abc)
; CHECK: call void @foo(i32 %abc)
; PTX: st.param.b32 [param0+0], [[abc:%r[0-9]+]];

  %b2 = shl i32 %b, 1
  %ab2 = add i32 %a, %b2
  %ab2c = add i32 %ab2, %c
; CHECK-NEXT: %ab2c = add i32 %abc, %b
; PTX: add.s32 [[ab2c:%r[0-9]+]], [[abc]], [[b]]
  call void @foo(i32 %ab2c)
; CHECK-NEXT: call void @foo(i32 %ab2c)
; PTX: st.param.b32 [param0+0], [[ab2c]];

  %b3 = mul i32 %b, 3
  %ab3 = add i32 %a, %b3
  %ab3c = add i32 %ab3, %c
; CHECK-NEXT: %ab3c = add i32 %ab2c, %b
; PTX: add.s32 [[ab3c:%r[0-9]+]], [[ab2c]], [[b]]
  call void @foo(i32 %ab3c)
; CHECK-NEXT: call void @foo(i32 %ab3c)
; PTX: st.param.b32 [param0+0], [[ab3c]];

  ret void
}

declare void @foo(i32)
