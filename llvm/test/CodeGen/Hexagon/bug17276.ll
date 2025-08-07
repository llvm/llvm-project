; RUN: llc -mtriple=hexagon -function-sections < %s | FileCheck %s
; CHECK: if (!p0)
; CHECK-NOT: if (p0.new)
; CHECK: {

target triple = "hexagon-unknown--elf"

%s.0 = type { ptr, ptr }
%s.1 = type { i8, [2 x ptr] }
%s.2 = type { i32, i32 }

@g0 = internal constant %s.0 zeroinitializer, align 4

; Function Attrs: minsize nounwind
define i32 @f0(ptr %a0) #0 {
b0:
  %v0 = tail call i32 @f1(ptr %a0, i32 0)
  ret i32 %v0
}

; Function Attrs: minsize nounwind
define internal i32 @f1(ptr %a0, i32 %a1) #0 {
b0:
  %v0 = icmp eq ptr %a0, null
  br i1 %v0, label %b4, label %b1

b1:                                               ; preds = %b0
  %v1 = icmp eq i32 %a1, 1
  br i1 %v1, label %b3, label %b2

b2:                                               ; preds = %b1
  tail call void @f2(ptr null) #3
  unreachable

b3:                                               ; preds = %b1
  tail call void @f2(ptr @g0) #3
  unreachable

b4:                                               ; preds = %b0
  %v2 = load ptr, ptr inttoptr (i32 4 to ptr), align 4, !tbaa !0
  %v3 = icmp eq ptr %v2, null
  br i1 %v3, label %b5, label %b6

b5:                                               ; preds = %b4
  tail call void @f3(i32 0) #4
  br label %b10

b6:                                               ; preds = %b4
  %v4 = tail call zeroext i8 @f4(ptr null) #4
  %v5 = icmp eq i8 %v4, 0
  br i1 %v5, label %b7, label %b8

b7:                                               ; preds = %b6
  tail call void @f3(i32 0) #4
  br label %b9

b8:                                               ; preds = %b6
  %v6 = load ptr, ptr inttoptr (i32 4 to ptr), align 4, !tbaa !0
  %v7 = icmp eq i32 %a1, 1
  %v8 = getelementptr inbounds %s.2, ptr %v6, i32 0, i32 1
  %v10 = select i1 %v7, ptr %v8, ptr %v6
  %v11 = tail call i32 @f5(ptr %v10) #4
  br label %b9

b9:                                               ; preds = %b8, %b7
  %v12 = phi i32 [ 0, %b7 ], [ %v11, %b8 ]
  tail call void @f3(i32 %v12) #4
  br label %b10

b10:                                              ; preds = %b9, %b5
  %v13 = phi i32 [ 0, %b5 ], [ %v12, %b9 ]
  ret i32 %v13
}

; Function Attrs: noreturn optsize
declare void @f2(ptr) #1

; Function Attrs: optsize
declare void @f3(i32) #2

; Function Attrs: optsize
declare zeroext i8 @f4(ptr) #2

; Function Attrs: optsize
declare i32 @f5(ptr) #2

; Function Attrs: minsize nounwind
define i32 @f6(ptr %a0) #0 {
b0:
  %v0 = tail call i32 @f1(ptr %a0, i32 1)
  ret i32 %v0
}

attributes #0 = { minsize nounwind }
attributes #1 = { noreturn optsize }
attributes #2 = { optsize }
attributes #3 = { noreturn nounwind optsize }
attributes #4 = { nounwind optsize }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
