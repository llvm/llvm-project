; RUN: opt < %s -passes=gvn -enable-pre -S | FileCheck %s

@H = common global i32 0
@G = common global i32 0

define i32 @test1(i1 %cond, i32 %v) nounwind {
; CHECK-LABEL: @test1
entry:
    br i1 %cond, label %bb, label %bb1

bb:
    %add.1 = add nuw nsw i32 %v, 42
; CHECK: %add.1 = add i32 %v, 42
    store i32 %add.1, ptr @G, align 4
    br label %return

bb1:
; CHECK: %.pre = add i32 %v, 42
    br label %return

return:
; CHECK: %add.2.pre-phi = phi i32 [ %.pre, %bb1 ], [ %add.1, %bb ]
; CHECK-NEXT: store i32 %add.2.pre-phi, ptr @H, align 4
; CHECK-NEXT: ret i32 0
    %add.2 = add i32 %v, 42
    store i32 %add.2, ptr @H, align 4
    ret i32 0
}

define i32 @test2(i1 %cond, i32 %v) nounwind {
; CHECK-LABEL: @test2
entry:
    br i1 %cond, label %bb, label %bb1

bb:
    %add.1 = add i32 %v, 42
; CHECK: %add.1 = add i32 %v, 42
    store i32 %add.1, ptr @G, align 4
    br label %return

bb1:
; CHECK: %.pre = add nuw nsw i32 %v, 42
    br label %return

return:
; CHECK: %add.2.pre-phi = phi i32 [ %.pre, %bb1 ], [ %add.1, %bb ]
; CHECK-NEXT: store i32 %add.2.pre-phi, ptr @H, align 4
; CHECK-NEXT: ret i32 0
    %add.2 = add nuw nsw i32 %v, 42
    store i32 %add.2, ptr @H, align 4
    ret i32 0
}
