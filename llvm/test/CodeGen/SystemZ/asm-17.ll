; Test explicit register names.
;
; RUN: llc < %s  -verify-machineinstrs -mtriple=s390x-linux-gnu -no-integrated-as \
; RUN:   | FileCheck %s

; Test i32 GPRs.
define i32 @f1() {
; CHECK-LABEL: f1:
; CHECK: lhi %r4, 1
; CHECK: blah %r4
; CHECK: lr %r2, %r4
; CHECK: br %r14
  %ret = call i32 asm "blah $0", "={r4},0" (i32 1)
  ret i32 %ret
}

; Test i64 GPRs.
define i64 @f2() {
; CHECK-LABEL: f2:
; CHECK: lghi %r4, 1
; CHECK: blah %r4
; CHECK: lgr %r2, %r4
; CHECK: br %r14
  %ret = call i64 asm "blah $0", "={r4},0" (i64 1)
  ret i64 %ret
}

; Test i32 FPRs.
define float @f3() {
; CHECK-LABEL: f3:
; CHECK: lzer %f4
; CHECK: blah %f4
; CHECK: ler %f0, %f4
; CHECK: br %r14
  %ret = call float asm "blah $0", "={f4},0" (float 0.0)
  ret float %ret
}

; Test i64 FPRs.
define double @f4() {
; CHECK-LABEL: f4:
; CHECK: lzdr %f4
; CHECK: blah %f4
; CHECK: ldr %f0, %f4
; CHECK: br %r14
  %ret = call double asm "blah $0", "={f4},0" (double 0.0)
  ret double %ret
}

; Test i128 FPRs.
define void @f5(ptr %dest) {
; CHECK-LABEL: f5:
; CHECK: lzxr %f4
; CHECK: blah %f4
; CHECK-DAG: std %f4, 0(%r2)
; CHECK-DAG: std %f6, 8(%r2)
; CHECK: br %r14
  %ret = call fp128 asm "blah $0", "={f4},0" (fp128 0xL00000000000000000000000000000000)
  store fp128 %ret, ptr %dest
  ret void
}

; Test clobbers of GPRs and CC.
define i32 @f6(i32 %in) {
; CHECK-LABEL: f6:
; CHECK: lr [[REG:%r[01345]]], %r2
; CHECK: blah
; CHECK: lr %r2, [[REG]]
; CHECK: br %r14
  call void asm sideeffect "blah", "~{r2},~{cc}"()
  ret i32 %in
}

; Test clobbers of FPRs and CC.
define float @f7(float %in) {
; CHECK-LABEL: f7:
; CHECK: ler [[REG:%f[1-7]]], %f0
; CHECK: blah
; CHECK: ler %f0, [[REG]]
; CHECK: br %r14
  call void asm sideeffect "blah", "~{f0},~{cc}"()
  ret float %in
}

; Test that both registers in a GR128 pair get hoisted.
define void @f8(i32 %count) {
; CHECK-LABEL: f8
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lhi %r1, 1
; CHECK: %loop
; CHECK-NOT: %r
; CHECK: blah %r0, %r1
; CHECK: br %r14
entry:
  br label %loop

loop:
  %this = phi i32 [ %count, %entry ], [ %next, %loop ]
  call void asm sideeffect "blah $0, $1", "{r0},{r1}" (i32 0, i32 1)
  %next = sub i32 %this, 1
  %cmp = icmp ne i32 %next, 0
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
