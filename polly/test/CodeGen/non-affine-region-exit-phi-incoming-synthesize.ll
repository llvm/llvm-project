; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; This caused the code generation to generate invalid code as the same BBMap was
; used for the whole non-affine region. When %add is synthesized for the
; incoming value of subregion_if first, the code for it was generated into
; subregion_if, but reused for the incoming value of subregion_exit, although it
; is not dominated by subregion_if.
;
; CHECK-LABEL: polly.stmt.subregion_entry:
; CHECK:         %[[R0:[0-9]*]] = add i32 %n, -2
;
; CHECK-LABEL: polly.stmt.subregion_if:
; CHECK:         %[[R1:[0-9]*]] = add i32 %n, -2
;
; CHECK-LABEL: polly.stmt.subregion_exit.region_exiting:
; CHECK:         %polly.retval = phi i32 [ %[[R1]], %polly.stmt.subregion_if ], [ %[[R0]], %polly.stmt.subregion_entry ]

define i32 @func(i32 %n){
entry:
  br label %subregion_entry

subregion_entry:
  %add = add nsw i32 %n, -2
  %cmp = fcmp ogt float undef, undef
  br i1 %cmp, label %subregion_if, label %subregion_exit

subregion_if:
  br label %subregion_exit

subregion_exit:
  %retval = phi i32 [ %add, %subregion_if ], [ %add, %subregion_entry ]
  ret i32 %retval
}
