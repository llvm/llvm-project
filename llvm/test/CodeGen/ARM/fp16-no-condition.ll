; RUN: llc -O3 -mtriple=armv8a-none-eabi -mattr=+fullfp16 -arm-restrict-it -o - %s | FileCheck %s
; RUN: llc -O3 -mtriple=thumbv8a-none-eabi -mattr=+fullfp16 -o - %s | FileCheck %s

; Require the vmul.f16 not to be predicated, because it's illegal to
; do so with fp16 instructions
define half @conditional_fmul_f16(ptr %p) {
; CHECK-LABEL: conditional_fmul_f16:
; CHECK: vmul.f16
entry:
  %p1 = getelementptr half, ptr %p, i32 1
  %a = load half, ptr %p, align 2
  %threshold = load half, ptr %p1, align 2
  %flag = fcmp ogt half %a, %threshold
  br i1 %flag, label %mul, label %out

mul:
  %p2 = getelementptr half, ptr %p, i32 2
  %mult = load half, ptr %p2, align 2
  %b = fmul half %a, %mult
  br label %out

out:
  %sel = phi half [ %a, %entry ], [ %b, %mul ]
  ret half %sel
}

; Expect that the corresponding vmul.f32 _will_ be predicated (to make
; sure the previous test is really testing something)
define float @conditional_fmul_f32(ptr %p) {
; CHECK-LABEL: conditional_fmul_f32:
; CHECK: vmulgt.f32
entry:
  %p1 = getelementptr float, ptr %p, i32 1
  %a = load float, ptr %p, align 2
  %threshold = load float, ptr %p1, align 2
  %flag = fcmp ogt float %a, %threshold
  br i1 %flag, label %mul, label %out

mul:
  %p2 = getelementptr float, ptr %p, i32 2
  %mult = load float, ptr %p2, align 2
  %b = fmul float %a, %mult
  br label %out

out:
  %sel = phi float [ %a, %entry ], [ %b, %mul ]
  ret float %sel
}

; Require the two comparisons to be done with unpredicated vcmp.f16
; instructions (again, it is illegal to predicate them)
define void @chained_comparisons_f16(ptr %p) {
; CHECK-LABEL: chained_comparisons_f16:
; CHECK: vcmp.f16
; CHECK: vcmp.f16
entry:
  %p1 = getelementptr half, ptr %p, i32 1

  %a = load half, ptr %p, align 2
  %b = load half, ptr %p1, align 2

  %aflag = fcmp oeq half %a, 0xH0000
  %bflag = fcmp oeq half %b, 0xH0000
  %flag = or i1 %aflag, %bflag
  br i1 %flag, label %call, label %out

call:
  call void @external_function()
  br label %out

out:
  ret void
}

; Again, do the corresponding test with 32-bit floats and check that
; the second comparison _is_ predicated on the result of the first.
define void @chained_comparisons_f32(ptr %p) {
; CHECK-LABEL: chained_comparisons_f32:
; CHECK: vcmp.f32
; CHECK: vcmpne.f32
entry:
  %p1 = getelementptr float, ptr %p, i32 1

  %a = load float, ptr %p, align 2
  %b = load float, ptr %p1, align 2

  %aflag = fcmp oeq float %a, 0x00000000
  %bflag = fcmp oeq float %b, 0x00000000
  %flag = or i1 %aflag, %bflag
  br i1 %flag, label %call, label %out

call:
  call void @external_function()
  br label %out

out:
  ret void
}

declare void @external_function()
