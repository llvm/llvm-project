; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s
; PR1600

declare i16 @llvm.cttz.i16(i16, i1)

define i32 @test(ptr %P, ptr %Q) {
; CHECK: ret i32 0
        %A = load i16, ptr %Q               ; <i16> [#uses=1]
        %x = load i32, ptr %P               ; <i32> [#uses=1]
        %B = call i16 @llvm.cttz.i16( i16 %A, i1 true )          ; <i16> [#uses=1]
        %y = load i32, ptr %P               ; <i32> [#uses=1]
        store i16 %B, ptr %Q
        %z = sub i32 %x, %y             ; <i32> [#uses=1]
        ret i32 %z
}

