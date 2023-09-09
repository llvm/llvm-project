; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-linux-gnu < %s \
; RUN:   2>&1 | FileCheck --check-prefix=CHECK-LE %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s \
; RUN:   2>&1 | FileCheck --check-prefix=CHECK-BE %s

; CHECK-NOT: warning: {{.*}} stack frame size ({{.*}}) exceeds limit (4294967295) in function 'foo'
; CHECK-NOT: warning: {{.*}} stack frame size ({{.*}}) exceeds limit (4294967295) in function 'large_stack'
; CHECK: warning: {{.*}} stack frame size ({{.*}}) exceeds limit (4294967295) in function 'warn_on_large_stack'

declare void @bar(ptr)

define void @foo(i8 %x) {
; CHECK-LE-LABEL: foo:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    lis 0, -1
; CHECK-LE-NEXT:    ori 0, 0, 65534
; CHECK-LE-NEXT:    sldi 0, 0, 32
; CHECK-LE-NEXT:    oris 0, 0, 65535
; CHECK-LE-NEXT:    ori 0, 0, 65504
; CHECK-LE-NEXT:    stdux 1, 1, 0
; CHECK-LE-NEXT:    .cfi_def_cfa_offset 32
; CHECK-LE-NEXT:    li 4, 1
; CHECK-LE-NEXT:    li 5, -1
; CHECK-LE-NEXT:    addi 6, 1, 32
; CHECK-LE-NEXT:    stb 3, 32(1)
; CHECK-LE-NEXT:    rldic 4, 4, 31, 32
; CHECK-LE-NEXT:    rldic 5, 5, 0, 32
; CHECK-LE-NEXT:    stbx 3, 6, 4
; CHECK-LE-NEXT:    stbx 3, 6, 5
; CHECK-LE-NEXT:    ld 1, 0(1)
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: foo:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    lis 0, -1
; CHECK-BE-NEXT:    ori 0, 0, 65534
; CHECK-BE-NEXT:    sldi 0, 0, 32
; CHECK-BE-NEXT:    oris 0, 0, 65535
; CHECK-BE-NEXT:    ori 0, 0, 65488
; CHECK-BE-NEXT:    stdux 1, 1, 0
; CHECK-BE-NEXT:    li 4, 1
; CHECK-BE-NEXT:    addi 5, 1, 48
; CHECK-BE-NEXT:    rldic 4, 4, 31, 32
; CHECK-BE-NEXT:    stb 3, 48(1)
; CHECK-BE-NEXT:    stbx 3, 5, 4
; CHECK-BE-NEXT:    li 4, -1
; CHECK-BE-NEXT:    rldic 4, 4, 0, 32
; CHECK-BE-NEXT:    stbx 3, 5, 4
; CHECK-BE-NEXT:    ld 1, 0(1)
; CHECK-BE-NEXT:    blr
entry:
  %a = alloca i8, i64 4294967296, align 16
  %c = getelementptr i8, ptr %a, i64 2147483648
  %d = getelementptr i8, ptr %a, i64 4294967295
  store volatile i8 %x, ptr %a
  store volatile i8 %x, ptr %c
  store volatile i8 %x, ptr %d
  ret void
}

define ptr @large_stack() {
  %s = alloca [281474976710656 x i8], align 1
  %e = getelementptr i8, ptr %s, i64 0
  ret ptr %e
}

define ptr @warn_on_large_stack() "warn-stack-size"="4294967295" {
  %s = alloca [281474976710656 x i8], align 1
  %e = getelementptr i8, ptr %s, i64 0
  ret ptr %e
}
