; RUN: opt < %s -reassociate -S | FileCheck %s

; These tests make sure that before processing insts
; any previous instructions are already canonicalized.
define i32 @foo(i32 %in) {
; CHECK-LABEL: @foo
; CHECK-NEXT: %factor = mul i32 %in, -4
; CHECK-NEXT: %factor1 = mul i32 %in, 2
; CHECK-NEXT: %_3 = add i32 %factor, 1
; CHECK-NEXT: %_5 = add i32 %_3, %factor1
; CHECK-NEXT: ret i32 %_5
  %_0 = add i32 %in, 1
  %_1 = mul i32 %in, -2
  %_2 = add i32 %_0, %_1
  %_3 = add i32 %_1, %_2
  %_4 = add i32 %_3, 1
  %_5 = add i32 %in, %_3
  ret i32 %_5
}

; CHECK-LABEL: @foo1
define void @foo1(float %in, i1 %cmp) {
wrapper_entry:
  br label %foo1

for.body:
  %0 = fadd float %in1, %in1
  br label %foo1

foo1:
  %_0 = fmul fast float %in, -3.000000e+00
  %_1 = fmul fast float %_0, 3.000000e+00
  %in1 = fadd fast float -3.000000e+00, %_1
  %in1use = fadd fast float %in1, %in1
  br label %for.body


}

; CHECK-LABEL: @foo2
define void @foo2(float %in, i1 %cmp) {
wrapper_entry:
  br label %for.body

for.body:
; If the operands of the phi are sheduled for processing before
; foo1 is processed, the invariant of reassociate are not preserved
  %unused = phi float [%in1, %foo1], [undef, %wrapper_entry]
  br label %foo1

foo1:
  %_0 = fmul fast float %in, -3.000000e+00
  %_1 = fmul fast float %_0, 3.000000e+00
  %in1 = fadd fast float -3.000000e+00, %_1
  %in1use = fadd fast float %in1, %in1
  br label %for.body
}
