; RUN: llc -march=x86-64 -mcpu=generic -mtriple=x86_64-unknown-linux-gnu -relocation-model=static -asm-verbose=false < %s | FileCheck %s
; RUN: llc -march=x86-64 -mcpu=atom -mtriple=x86_64-unknown-linux-gnu -relocation-model=static -asm-verbose=false < %s | FileCheck -check-prefix=ATOM %s

; CHECK: xorl  %eax, %eax
; CHECK: movsd .LCPI0_0(%rip), %xmm0
; CHECK: align
; CHECK-NEXT: BB0_2:
; CHECK-NEXT: movsd A(,%rax,8)
; CHECK-NEXT: mulsd
; CHECK-NEXT: movsd
; CHECK-NEXT: incq %rax


; ATOM: xorl  %eax, %eax
; ATOM: movsd .LCPI0_0(%rip), %xmm0
; ATOM: align
; ATOM-NEXT: BB0_2:
; ATOM-NEXT: movsd A(,%rax,8)
; ATOM-NEXT: mulsd
; ATOM-NEXT: movsd
; ATOM-NEXT: incq %rax

@A = external global [0 x double]

define void @foo(i64 %n) nounwind {
entry:
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %for.body, label %for.end

for.body:
  %i.06 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr [0 x double], [0 x double]* @A, i64 0, i64 %i.06
  %tmp3 = load double, double* %arrayidx, align 8
  %mul = fmul double %tmp3, 2.300000e+00
  store double %mul, double* %arrayidx, align 8
  %inc = add nsw i64 %i.06, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
