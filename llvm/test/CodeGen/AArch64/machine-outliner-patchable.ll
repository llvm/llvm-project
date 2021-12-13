; RUN: llc < %s -verify-machineinstrs -enable-machine-outliner | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

declare void @foo(i32, i32, i32, i32) minsize

;; TargetOpcode::FENTRY_CALL at the start of the function expands to a __fentry__
;; call which must be present. Don't outline it.
define void @fentry0(i1 %a) nounwind "fentry-call"="true" {
; CHECK-LABEL: fentry0:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    # FEntry call
; CHECK:       // %bb.1:
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

define void @fentry1(i1 %a) nounwind "fentry-call"="true" {
; CHECK-LABEL: fentry1:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    # FEntry call
; CHECK:       // %bb.1:
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

;; TargetOpcode::PATCHABLE_FUNCTION_ENTER at the start of the function expands to
;; NOPs which must be present. Don't outline them.
define void @patchable0(i1 %a) nounwind "patchable-function-entry"="2" {
; CHECK-LABEL: patchable0:
; CHECK-NEXT:  .Lfunc_begin0:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK:       // %bb.1:
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

define void @patchable1(i1 %a) nounwind "patchable-function-entry"="2" {
; CHECK-LABEL: patchable1:
; CHECK-NEXT:  .Lfunc_begin1:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK:       // %bb.1:
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

;; Similar to "patchable-function-entry".
define void @xray0(i1 %a) nounwind "function-instrument"="xray-always" {
; CHECK-LABEL: xray0:
; CHECK-NEXT:  .Lfunc_begin2:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:  .p2align 2
; CHECK-NEXT:  .Lxray_sled_0:
; CHECK:       // %bb.1:
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

define void @xray1(i1 %a) nounwind "function-instrument"="xray-always" {
; CHECK-LABEL: xray1:
; CHECK-NEXT:  .Lfunc_begin3:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:  .p2align 2
; CHECK-NEXT:  .Lxray_sled_2:
; CHECK:       // %bb.1:
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}
