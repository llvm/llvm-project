; REQUIRES: have_tflite
; REQUIRES: default_triple
;
; Check that the basic block profile dump outputs data and in the correct
; format.
;
; RUN: llc -o /dev/null -mbb-profile-dump=%t %s
; RUN: FileCheck --input-file %t %s

define i64 @f2(i64 %a, i64 %b) {
    %sum = add i64 %a, %b
    ret i64 %sum
}

define i64 @f1() {
    %sum = call i64 @f2(i64 2, i64 2)
    ret i64 %sum
}

; CHECK: f2,0,1.000000e+00
; CHECK-NEXT: f1,0,1.000000e+00
