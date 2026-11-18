; RUN: mlir-translate -import-llvm -split-input-file -verify-diagnostics %s | FileCheck %s

; CHECK-LABEL: llvm.func @inline_asm(
; CHECK-SAME: %[[ARG:[^:]+]]: i32,
; CHECK: %[[RESULT:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "", "=r,0" %[[ARG]] : (i32) -> i32
; CHECK-NEXT: llvm.return %[[RESULT]] : i32
define i32 @inline_asm(i32 %arg, i32 %state) {
  ; expected-warning @unknown {{dropping operand bundles on inline assembly}}
  %result = call i32 asm sideeffect "", "=r,0"(i32 %arg) ["tag"(i32 %state)]
  ret i32 %result
}

; // -----
; expected-warning-re @-2 {{unsupported metadata: !{{[0-9]+}} = distinct !{!"payload"}}}
declare i32 @callee(i32)
declare float @llvm.fptrunc.round.f32.f64(double, metadata)

; CHECK-LABEL: llvm.func @unsupported_metadata(
; CHECK-SAME: %[[ARG:.*]]: i32, %[[STATE:.*]]: i32)
; CHECK: %[[RESULT:.*]] = llvm.call @callee(%[[ARG]]) ["before"(%[[STATE]] : i32), "empty"(), "after"(%[[ARG]] : i32)] : (i32) -> i32
; CHECK-NEXT: llvm.return %[[RESULT]] : i32
define i32 @unsupported_metadata(i32 %arg, i32 %state) {
  ; expected-warning @unknown {{dropping operand bundle 'bad'}}
  %result = call i32 @callee(i32 %arg) ["before"(i32 %state), "bad"(i32 %arg, metadata !0, metadata !0), "empty"(), "after"(i32 %arg)]
  ret i32 %result
}

; CHECK-LABEL: llvm.func @intrinsic_without_remaining_bundles(
; CHECK-SAME: %[[ARG:.*]]: f64)
; CHECK: %[[ROUNDING:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[RESULT:.*]] = llvm.call_intrinsic "llvm.fptrunc.round.f32.f64"(%[[ARG]], %[[ROUNDING]]) : (f64, !llvm.metadata) -> f32
; CHECK: llvm.return %[[RESULT]] : f32
define float @intrinsic_without_remaining_bundles(double %arg) {
  ; expected-warning @unknown {{unhandled constant: ptr no_cfi @callee since no_cfi is unsupported}}
  ; expected-warning @unknown {{dropping operand bundle 'state'}}
  %result = call float @llvm.fptrunc.round.f32.f64(double %arg, metadata !"round.tonearest") ["state"(ptr no_cfi @callee)]
  call i32 @callee(i32 ptrtoint (ptr @callee to i32))
  ret float %result
}

!0 = distinct !{!"payload"}

; // -----
; expected-warning-re @-2 {{unsupported metadata: !{{[0-9]+}} = distinct !{}}}
declare void @callee()

; CHECK-LABEL: llvm.func @shared_metadata(
; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"shared">
; CHECK: llvm.cond_br
; CHECK: llvm.call @callee() : () -> ()
; CHECK: llvm.call @callee() ["keep"(%[[MD]] : !llvm.metadata)] : () -> ()
define void @shared_metadata(i1 %cond) {
  br i1 %cond, label %keep, label %drop
drop:
  ; expected-warning @unknown {{dropping operand bundle 'drop'}}
  call void @callee() ["drop"(metadata !"shared", metadata !0)]
  ret void
keep:
  call void @callee() ["keep"(metadata !"shared")]
  ret void
}
!0 = distinct !{}

; // -----
declare void @callee(ptr)
define void @unsupported_call_argument() {
  ; expected-error @unknown {{unhandled constant: ptr no_cfi @callee since no_cfi is unsupported}}
  call void @callee(ptr no_cfi @callee)
  ret void
}
