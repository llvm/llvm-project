; RUN: llc -mtriple=arm64_32-apple-watchos -stop-after=finalize-isel %s -o - | FileCheck %s

define ptr @issue94075(ptr %p) {
entry:
  %rdar125263567 = call ptr @llvm.ptrmask.p0.i32(ptr %p, i32 4294967288)
  ret ptr %rdar125263567
}

; CHECK-LABEL: name: issue94075
; CHECK:         %0:gpr64 = COPY $x0
; CHECK-NEXT:    %1:gpr64sp = ANDXri %0, 8028
; CHECK-NEXT:    $x0 = COPY %1
; CHECK-NEXT:    RET_ReallyLR implicit $x0