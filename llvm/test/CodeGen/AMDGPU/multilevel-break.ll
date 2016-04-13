; RUN: opt -S -mtriple=amdgcn-- -structurizecfg -si-annotate-control-flow < %s | FileCheck %s

; CHECK-LABEL: {{^}}define amdgpu_vs void @main
; CHECK: main_body:
; CHECK: LOOP.outer:
; CHECK: LOOP:
; CHECK:     [[if:%[0-9]+]] = call { i1, i64 } @llvm.amdgcn.if(
; CHECK:     [[if_exec:%[0-9]+]] = extractvalue { i1, i64 } [[if]], 1
;
; CHECK: Flow:
;
; Ensure two else.break calls, for both the inner and outer loops
;
; CHECK:        call i64 @llvm.amdgcn.else.break(i64 [[if_exec]],
; CHECK-NEXT:   call i64 @llvm.amdgcn.else.break(i64 [[if_exec]],
; CHECK-NEXT:   call void @llvm.amdgcn.end.cf
;
; CHECK: Flow1:
define amdgpu_vs void @main(<4 x float> %vec, i32 %ub, i32 %cont) {
main_body:
  br label %LOOP.outer

LOOP.outer:                                       ; preds = %ENDIF, %main_body
  %tmp43 = phi i32 [ 0, %main_body ], [ %tmp47, %ENDIF ]
  br label %LOOP

LOOP:                                             ; preds = %ENDIF, %LOOP.outer
  %tmp45 = phi i32 [ %tmp43, %LOOP.outer ], [ %tmp47, %ENDIF ]
  %tmp47 = add i32 %tmp45, 1
  %tmp48 = icmp slt i32 %tmp45, %ub
  br i1 %tmp48, label %ENDIF, label %IF

IF:                                               ; preds = %LOOP
  ret void

ENDIF:                                            ; preds = %LOOP
  %tmp51 = icmp eq i32 %tmp47, %cont
  br i1 %tmp51, label %LOOP, label %LOOP.outer
}

attributes #0 = { nounwind readnone }
