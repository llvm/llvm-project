; RUN: opt < %s -passes=sccp -S | FileCheck %s

define i1 @float.1(i1 %cmp) {
; CHECK-LABEL: define i1 @float.1(i1 %cmp) {

; CHECK-LABEL: end:
; CHECK-NEXT:    ret i1 true
;
entry:
  br i1 %cmp, label %if.true, label %end

if.true:
  br label %end

end:
  %p = phi float [ 1.0, %entry ], [ 1.0, %if.true]
  %c = fcmp ueq float %p, 1.0
  ret i1 %c
}

define i1 @float.2(i1 %cmp) {
; CHECK-LABEL: define i1 @float.2(i1 %cmp) {

; CHECK-LABEL: end:
; CHECK-NEXT:    %p = phi float [ 1.000000e+00, %entry ], [ 2.000000e+00, %if.true ]
; CHECK-NEXT:    %c = fcmp ueq float %p, 1.000000e+00
; CHECK-NEXT:    ret i1 %c
;
entry:
  br i1 %cmp, label %if.true, label %end

if.true:
  br label %end

end:
  %p = phi float [ 1.0, %entry ], [ 2.0, %if.true]
  %c = fcmp ueq float %p, 1.0
  ret i1 %c
}

define i1 @float.3(float %f, i1 %cmp) {
; CHECK-LABEL: define i1 @float.3(float %f, i1 %cmp)

; CHECK-LABEL: end:
; CHECK-NEXT:    %p = phi float [ 1.000000e+00, %entry ], [ %f, %if.true ]
; CHECK-NEXT:    %c = fcmp ueq float %p, 1.000000e+00
; CHECK-NEXT:    ret i1 %c
;
entry:
  br i1 %cmp, label %if.true, label %end

if.true:
  br label %end

end:
  %p = phi float [ 1.0, %entry ], [ %f, %if.true]
  %c = fcmp ueq float %p, 1.0
  ret i1 %c
}


define i1 @float.4_unreachable(float %f, i1 %cmp) {
; CHECK-LABEL: define i1 @float.4_unreachable(float %f, i1 %cmp)

; CHECK-LABEL: end:
; CHECK-NEXT:    ret i1 false
;
entry:
  br i1 %cmp, label %if.true, label %end

if.true:
  br label %end

dead:
  br label %end

end:
  %p = phi float [ 1.0, %entry ], [ 1.0, %if.true], [ %f, %dead ]
  %c = fcmp une float %p, 1.0
  ret i1 %c
}
