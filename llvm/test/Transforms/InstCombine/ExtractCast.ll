; RUN: opt < %s -instcombine -S -o - | FileCheck %s

; CHECK: @a
define i32 @a(<4 x i64> %I) {
entry:
; CHECK-NOT: trunc <4 x i64>
        %J = trunc <4 x i64> %I to <4 x i32>
        %K = extractelement <4 x i32> %J, i32 3
; CHECK: extractelement <4 x i64>
; CHECK: trunc i64
; CHECK: ret
        ret i32 %K
}


; CHECK: @b
define i32 @b(<4 x float> %I) {
entry:
; CHECK-NOT: fptosi <4 x float>
        %J = fptosi <4 x float> %I to <4 x i32>
        %K = extractelement <4 x i32> %J, i32 3
; CHECK: extractelement <4 x float>
; CHECK: fptosi float
; CHECK: ret
        ret i32 %K
}

