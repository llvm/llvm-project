; RUN: llc -mtriple=arm64-apple-macosx -verify-machineinstrs < %s | FileCheck %s
; RUN: not --crash llc -mtriple=aarch64-unknown-linux < %s 2>&1 | FileCheck %s --check-prefix=ERROR

; Verify that the stack-probing attribute emitted by Apple Swift is accepted
; on Darwin and lowered through the Darwin stack-check helper ABI.

define void @below_probe_threshold(ptr %out) #0 {
; CHECK-LABEL: below_probe_threshold:
; CHECK-NOT:   ___chkstk_darwin
; CHECK:       sub sp, sp, #1040
; CHECK-NOT:   ___chkstk_darwin
; CHECK:       ret
entry:
  %frame = alloca i8, i64 1040, align 1
  store ptr %frame, ptr %out, align 8
  ret void
}

define void @just_below_probe_threshold(ptr %out) #0 {
; CHECK-LABEL: just_below_probe_threshold:
; CHECK-NOT:   ___chkstk_darwin
; CHECK:       sub sp, sp, #4080
; CHECK-NOT:   ___chkstk_darwin
; CHECK:       ret
entry:
  %frame = alloca i8, i64 4080, align 1
  store ptr %frame, ptr %out, align 8
  ret void
}

define void @fixed_probe(ptr %out) #0 {
; CHECK-LABEL: fixed_probe:
; CHECK:       str x30, [sp, #-16]!
; CHECK-DAG:   mov x9, #4096
; CHECK-DAG:   adrp x16, ___chkstk_darwin@GOTPAGE
; CHECK:       ldr x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; CHECK-NEXT:  blr x16
; CHECK-NEXT:  ldr x30, [sp], #16
; CHECK-NEXT:  sub sp, sp, #1, lsl #12
; CHECK-NOT:   ldr xzr, [sp]
; CHECK:       ret
entry:
  %frame = alloca i8, i64 4096, align 1
  store ptr %frame, ptr %out, align 8
  ret void
}

define void @large_fixed_probe(ptr %out) #0 {
; CHECK-LABEL: large_fixed_probe:
; CHECK:       str x30, [sp, #-16]!
; CHECK-DAG:   mov x9, #20480
; CHECK-DAG:   adrp x16, ___chkstk_darwin@GOTPAGE
; CHECK:       ldr x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; CHECK-NEXT:  blr x16
; CHECK-NEXT:  ldr x30, [sp], #16
; CHECK-NEXT:  sub sp, sp, #5, lsl #12
; CHECK-NOT:   ldr xzr, [sp]
; CHECK:       ret
entry:
  %frame = alloca i8, i64 20480, align 1
  store ptr %frame, ptr %out, align 8
  ret void
}

define void @dynamic_probe(ptr %out, i64 %size) #0 {
; CHECK-LABEL: dynamic_probe:
; CHECK-DAG:   mov x9, [[SIZE:x[0-9]+]]
; Keep the old SP in a register preserved by both the current helper and the
; compatibility implementation, which clobbers x9, x10, and x16.
; CHECK-DAG:   mov [[OLDSP:x([0-8]|1[1-5]|1[7-9]|2[0-8])]], sp
; CHECK-DAG:   adrp [[CALLEE:x[0-9]+]], ___chkstk_darwin@GOTPAGE
; CHECK-DAG:   ldr [[CALLEE]], {{\[}}[[CALLEE]], ___chkstk_darwin@GOTPAGEOFF]
; CHECK:       blr [[CALLEE]]
; CHECK:       sub [[TARGETSP:x[0-9]+]], [[OLDSP]], [[SIZE]]
; CHECK-NEXT:  mov sp, [[TARGETSP]]
; CHECK:       ret
entry:
  %frame = alloca i8, i64 %size, align 16
  store ptr %frame, ptr %out, align 8
  ret void
}

attributes #0 = {
  noinline
  "frame-pointer"="none"
  "probe-stack"="__chkstk_darwin"
  "stack-probe-size"="4096"
  uwtable(async)
}

; ERROR: LLVM ERROR: Unsupported stack probing method
