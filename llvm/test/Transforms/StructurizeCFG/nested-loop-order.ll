; RUN: opt -S -structurizecfg %s -o - | FileCheck %s

define void @main(float addrspace(1)* %out) {

; CHECK: main_body:
; CHECK: br label %LOOP.outer
main_body:
  br label %LOOP.outer

; CHECK: LOOP.outer:
; CHECK: br label %LOOP
LOOP.outer:                                       ; preds = %ENDIF28, %main_body
  %temp8.0.ph = phi float [ 0.000000e+00, %main_body ], [ %tmp35, %ENDIF28 ]
  %temp4.0.ph = phi i32 [ 0, %main_body ], [ %tmp20, %ENDIF28 ]
  br label %LOOP

; CHECK: LOOP:
; br i1 %{{[0-9]+}}, label %ENDIF, label %Flow
LOOP:                                             ; preds = %IF29, %LOOP.outer
  %temp4.0 = phi i32 [ %temp4.0.ph, %LOOP.outer ], [ %tmp20, %IF29 ]
  %tmp20 = add i32 %temp4.0, 1
  %tmp22 = icmp sgt i32 %tmp20, 3
  br i1 %tmp22, label %ENDLOOP, label %ENDIF

; CHECK: Flow3
; CHECK: br i1 %{{[0-9]+}}, label %ENDLOOP, label %LOOP.outer

; CHECK: ENDLOOP:
; CHECK: ret void
ENDLOOP:                                          ; preds = %ENDIF28, %IF29, %LOOP
  %temp8.1 = phi float [ %temp8.0.ph, %LOOP ], [ %temp8.0.ph, %IF29 ], [ %tmp35, %ENDIF28 ]
  %tmp23 = icmp eq i32 %tmp20, 3
  %.45 = select i1 %tmp23, float 0.000000e+00, float 1.000000e+00
  store float %.45, float addrspace(1)* %out
  ret void

; CHECK: ENDIF:
; CHECK: br i1 %tmp31, label %IF29, label %Flow1
ENDIF:                                            ; preds = %LOOP
  %tmp31 = icmp sgt i32 %tmp20, 1
  br i1 %tmp31, label %IF29, label %ENDIF28

; CHECK: Flow:
; CHECK: br i1 %{{[0-9]+}}, label %Flow2, label %LOOP

; CHECK: IF29:
; CHECK: br label %Flow1
IF29:                                             ; preds = %ENDIF
  %tmp32 = icmp sgt i32 %tmp20, 2
  br i1 %tmp32, label %ENDLOOP, label %LOOP

; CHECK: Flow1:
; CHECK: br label %Flow

; CHECK: Flow2:
; CHECK: br i1 %{{[0-9]+}}, label %ENDIF28, label %Flow3

; CHECK: ENDIF28:
; CHECK: br label %Flow3
ENDIF28:                                          ; preds = %ENDIF
  %tmp35 = fadd float %temp8.0.ph, 1.0
  %tmp36 = icmp sgt i32 %tmp20, 2
  br i1 %tmp36, label %ENDLOOP, label %LOOP.outer
}

attributes #0 = { "enable-no-nans-fp-math"="true" "unsafe-fp-math"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { readnone }
