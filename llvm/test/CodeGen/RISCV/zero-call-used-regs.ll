; RUN: llc < %s -verify-machineinstrs -mtriple=riscv64-unknown-unknown | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=riscv32-unknown-unknown | FileCheck %s

target triple = "riscv64-unknown-linux-gnu"

define dso_local i32 @skip(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 "zero-call-used-regs"="skip" {
; CHECK-LABEL: skip:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local i32 @used_gpr_arg(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="used-gpr-arg" {
; CHECK-LABEL: used_gpr_arg:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local i32 @used_gpr(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="used-gpr" {
; CHECK-LABEL: used_gpr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local i32 @used_arg(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="used-arg" {
; CHECK-LABEL: used_arg:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local i32 @used(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="used" {
; CHECK-LABEL: used:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local i32 @all_gpr_arg(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 "zero-call-used-regs"="all-gpr-arg" {
; CHECK-LABEL: all_gpr_arg:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    li a3, 0
; CHECK-NEXT:    li a4, 0
; CHECK-NEXT:    li a5, 0
; CHECK-NEXT:    li a6, 0
; CHECK-NEXT:    li a7, 0
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local i32 @all_gpr(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 "zero-call-used-regs"="all-gpr" {
; CHECK-LABEL: all_gpr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mul{{w?}} a0, a1, a0
; CHECK-NEXT:    or a0, a0, a2
; CHECK-NEXT:    li t0, 0
; CHECK-NEXT:    li t1, 0
; CHECK-NEXT:    li t2, 0
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    li a3, 0
; CHECK-NEXT:    li a4, 0
; CHECK-NEXT:    li a5, 0
; CHECK-NEXT:    li a6, 0
; CHECK-NEXT:    li a7, 0
; CHECK-NEXT:    li t3, 0
; CHECK-NEXT:    li t4, 0
; CHECK-NEXT:    li t5, 0
; CHECK-NEXT:    li t6, 0
; CHECK-NEXT:    ret

entry:
  %mul = mul nsw i32 %b, %a
  %or = or i32 %mul, %c
  ret i32 %or
}

define dso_local double @skip_float(double noundef %a, float noundef %b) local_unnamed_addr #0 "zero-call-used-regs"="skip" {
; CHECK-LABEL: skip_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fcvt.d.s fa5, fa1
; CHECK-NEXT:    fmul.d	fa0, fa5, fa0
; CHECK-NEXT:    ret

entry:
  %conv = fpext float %b to double
  %mul = fmul double %conv, %a
  ret double %mul
}

define dso_local double @used_gpr_arg_float(double noundef %a, float noundef %b) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="used-gpr-arg" {
; CHECK-LABEL: used_gpr_arg_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fcvt.d.s fa5, fa1
; CHECK-NEXT:    fmul.d	fa0, fa5, fa0
; CHECK-NEXT:    ret

entry:
  %conv = fpext float %b to double
  %mul = fmul double %conv, %a
  ret double %mul
}

define dso_local double @used_gpr_float(double noundef %a, float noundef %b) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="used-gpr" {
; CHECK-LABEL: used_gpr_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fcvt.d.s fa5, fa1
; CHECK-NEXT:    fmul.d	fa0, fa5, fa0
; CHECK-NEXT:    ret

entry:
  %conv = fpext float %b to double
  %mul = fmul double %conv, %a
  ret double %mul
}

define dso_local double @all_gpr_arg_float(double noundef %a, float noundef %b) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="all-gpr-arg" {
; CHECK-LABEL: all_gpr_arg_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fcvt.d.s fa5, fa1
; CHECK-NEXT:    fmul.d	fa0, fa5, fa0
; CHECK-NEXT:    li a0, 0
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    li a3, 0
; CHECK-NEXT:    li a4, 0
; CHECK-NEXT:    li a5, 0
; CHECK-NEXT:    li a6, 0
; CHECK-NEXT:    li a7, 0
; CHECK-NEXT:    ret

entry:
  %conv = fpext float %b to double
  %mul = fmul double %conv, %a
  ret double %mul
}

define dso_local double @all_gpr_float(double noundef %a, float noundef %b) local_unnamed_addr #0 noinline optnone "zero-call-used-regs"="all-gpr" {
; CHECK-LABEL: all_gpr_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fcvt.d.s fa5, fa1
; CHECK-NEXT:    fmul.d	fa0, fa5, fa0
; CHECK-NEXT:    li t0, 0
; CHECK-NEXT:    li t1, 0
; CHECK-NEXT:    li t2, 0
; CHECK-NEXT:    li a0, 0
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    li a3, 0
; CHECK-NEXT:    li a4, 0
; CHECK-NEXT:    li a5, 0
; CHECK-NEXT:    li a6, 0
; CHECK-NEXT:    li a7, 0
; CHECK-NEXT:    li t3, 0
; CHECK-NEXT:    li t4, 0
; CHECK-NEXT:    li t5, 0
; CHECK-NEXT:    li t6, 0
; CHECK-NEXT:    ret

entry:
  %conv = fpext float %b to double
  %mul = fmul double %conv, %a
  ret double %mul
}

; Don't emit zeroing registers in "main" function.
define dso_local i32 @main() local_unnamed_addr #0 {
; CHECK-LABEL: main:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    li a0, 0
; CHECK-NEXT:    ret

entry:
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+m,+f,+d" }
