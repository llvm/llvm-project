; RUN: opt < %s -S -instcombine | FileCheck %s
; rdar://problem/9267970
; ideally this test will run on a 32-bit host
; must not discard GEPs that might overflow at runtime (aren't inbounds)

define i32 @main(i32 %argc) {
entry:
    %tmp1 = add i32 %argc, -2
    %tmp2 = add i32 %argc, 1879048192
    %p = alloca i8
; CHECK: getelementptr
    %p1 = getelementptr i8* %p, i32 %tmp1
; CHECK: getelementptr
    %p2 = getelementptr i8* %p, i32 %tmp2
    %cmp = icmp ult i8* %p1, %p2
    br i1 %cmp, label %bbtrue, label %bbfalse
bbtrue:          ; preds = %entry
    ret i32 -1
bbfalse:         ; preds = %entry
    ret i32 0
}
