; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; The branch instruction in LOOP49 has a uniform condition, but PHI instructions
; introduced by the structurizecfg pass previously caused a false divergence
; which ended up in an assertion (or incorrect code) because
; SIAnnotateControlFlow and structurizecfg had different ideas about which
; branches are uniform.
;
; CHECK-LABEL: {{^}}main:
; CHECK: ; %LOOP49
; CHECK: s_cmp_{{lg|eq}}_u32 s{{[0-9]+}}, 0
; CHECK: s_cbranch_scc{{[0-1]}}
; CHECK: ; %ENDIF53
define amdgpu_vs float @main(i32 %in) {
main_body:
  %cmp = mul i32 %in, 2
  br label %LOOP

LOOP:                                             ; preds = %ENDLOOP48, %main_body
  %counter = phi i32 [ 0, %main_body ], [ %counter.next, %ENDLOOP48 ]
  %v.LOOP = phi i32 [ 0, %main_body ], [ %v.ENDLOOP48, %ENDLOOP48 ]
  %tmp7 = icmp slt i32 %cmp, %counter
  br i1 %tmp7, label %IF, label %LOOP49

IF:                                               ; preds = %LOOP
  %r = bitcast i32 %v.LOOP to float
  ret float %r

LOOP49:                                           ; preds = %LOOP
  %tmp8 = icmp ne i32 %counter, 0
  br i1 %tmp8, label %ENDLOOP48, label %ENDIF53

ENDLOOP48:                                        ; preds = %ENDIF53, %LOOP49
  %v.ENDLOOP48 = phi i32 [ %v.LOOP, %LOOP49 ], [ %v.ENDIF53, %ENDIF53 ]
  %counter.next = add i32 %counter, 1
  br label %LOOP

ENDIF53:                                          ; preds = %LOOP49
  %v.ENDIF53 = add i32 %v.LOOP, %counter
  br label %ENDLOOP48
}
