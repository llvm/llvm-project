; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts
; Check for successful compilation.

target triple = "hexagon-unknown--elf"

%s.0 = type { ptr }
%s.1 = type { i32 }
%s.2 = type { %s.1 }

@g0 = global { i32, i32 } { i32 ptrtoint (ptr @f0 to i32), i32 0 }, align 4
@g1 = global i32 0, align 4
@g2 = global %s.0 zeroinitializer, align 4
@g3 = global { i32, i32 } { i32 1, i32 0 }, align 4
@g4 = global i32 0, align 4
@g5 = global i32 0, align 4
@g6 = global i32 0, align 4
@g7 = private unnamed_addr constant [53 x i8] c"REF: ISO/IEC 14882:1998, 8.2.3 Pointers to members.\0A\00", align 1
@g8 = private unnamed_addr constant [6 x i8] c"%s\0A%s\00", align 1
@g9 = private unnamed_addr constant [43 x i8] c"Can we assign a pointer to member function\00", align 1
@g10 = private unnamed_addr constant [49 x i8] c" to a function member of the second base class?\0A\00", align 1
@g11 = external global i32
@g12 = private unnamed_addr constant [46 x i8] c"Can we assign a pointer to member to a member\00", align 1
@g13 = private unnamed_addr constant [29 x i8] c"  of the second base class?\0A\00", align 1
@g14 = private unnamed_addr constant [7 x i8] c"%s\0A%s\0A\00", align 1
@g15 = private unnamed_addr constant [51 x i8] c"Testing dereferencing a pointer to member function\00", align 1
@g16 = private unnamed_addr constant [24 x i8] c"in a complex expression\00", align 1
@g17 = linkonce_odr unnamed_addr constant [3 x ptr] [ptr null, ptr @g20, ptr @f9]
@g18 = external global ptr
@g19 = linkonce_odr constant [3 x i8] c"1S\00"
@g20 = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @g18, i32 2), ptr @g19 }

; Function Attrs: nounwind readnone
define linkonce_odr i32 @f0(ptr nocapture readnone %a0) #0 align 2 {
b0:
  ret i32 11
}

; Function Attrs: nounwind readnone
define ptr @f1() #0 {
b0:
  ret ptr @g2
}

define internal fastcc void @f2() {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !0
  %v1 = add nsw i32 %v0, 5
  store i32 %v1, ptr @g5, align 4, !tbaa !0
  %v2 = load { i32, i32 }, ptr @g3, align 4, !tbaa !4
  %v3 = extractvalue { i32, i32 } %v2, 1
  %v4 = getelementptr inbounds i8, ptr @g2, i32 %v3
  %v6 = extractvalue { i32, i32 } %v2, 0
  %v7 = and i32 %v6, 1
  %v8 = icmp eq i32 %v7, 0
  br i1 %v8, label %b2, label %b1

b1:                                               ; preds = %b0
  %v10 = load ptr, ptr %v4, align 4, !tbaa !5
  %v11 = add i32 %v6, -1
  %v12 = getelementptr i8, ptr %v10, i32 %v11
  %v14 = load ptr, ptr %v12, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v15 = inttoptr i32 %v6 to ptr
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v16 = phi ptr [ %v14, %b1 ], [ %v15, %b2 ]
  %v17 = tail call i32 %v16(ptr %v4)
  store i32 %v17, ptr @g6, align 4, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
define i32 @f3() #0 {
b0:
  %v0 = alloca %s.2, align 4
  %v1 = alloca %s.2, align 4
  tail call void @f4()
  tail call void @f5()
  tail call void (ptr, ...) @f6(ptr @g7)
  tail call void (ptr, ...) @f6(ptr @g8, ptr @g9, ptr @g10)
  %v2 = load { i32, i32 }, ptr @g0, align 4, !tbaa !4
  %v3 = extractvalue { i32, i32 } %v2, 1
  %v5 = getelementptr inbounds i8, ptr %v0, i32 %v3
  %v7 = extractvalue { i32, i32 } %v2, 0
  %v8 = and i32 %v7, 1
  %v9 = icmp eq i32 %v8, 0
  br i1 %v9, label %b1, label %b2

b1:                                               ; preds = %b0
  %v10 = inttoptr i32 %v7 to ptr
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v11 = phi ptr [ %v10, %b1 ], [ undef, %b0 ]
  %v12 = call i32 %v11(ptr %v5)
  %v13 = icmp eq i32 %v12, 11
  br i1 %v13, label %b4, label %b3

b3:                                               ; preds = %b2
  store i32 1, ptr @g11, align 4, !tbaa !0
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v14 = call i32 @f7()
  call void @f5()
  call void (ptr, ...) @f6(ptr @g8, ptr @g12, ptr @g13)
  store i32 11, ptr %v1, align 4, !tbaa !7
  %v16 = load i32, ptr @g1, align 4, !tbaa !4
  %v18 = getelementptr inbounds i8, ptr %v1, i32 %v16
  %v20 = load i32, ptr %v18, align 4, !tbaa !0
  %v21 = icmp eq i32 %v20, 11
  br i1 %v21, label %b6, label %b5

b5:                                               ; preds = %b4
  store i32 1, ptr @g11, align 4, !tbaa !0
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v22 = call i32 @f7()
  call void @f5()
  call void (ptr, ...) @f6(ptr @g14, ptr @g15, ptr @g16)
  %v23 = load i32, ptr @g4, align 4, !tbaa !0
  %v24 = icmp eq i32 %v23, 11
  br i1 %v24, label %b8, label %b7

b7:                                               ; preds = %b6
  store i32 1, ptr @g11, align 4, !tbaa !0
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v25 = call i32 @f7()
  call void @f5()
  call void (ptr, ...) @f6(ptr @g14, ptr @g15, ptr @g16)
  %v26 = load i32, ptr @g6, align 4, !tbaa !0
  %v27 = icmp eq i32 %v26, 11
  br i1 %v27, label %b10, label %b9

b9:                                               ; preds = %b8
  store i32 1, ptr @g11, align 4, !tbaa !0
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v28 = call i32 @f7()
  %v29 = call i32 @f8(i32 4)
  ret i32 %v29
}

; Function Attrs: nounwind readnone
declare void @f4() #0

; Function Attrs: nounwind readnone
declare void @f5() #0

; Function Attrs: nounwind readnone
declare void @f6(ptr, ...) #0

; Function Attrs: nounwind readnone
declare i32 @f7() #0

; Function Attrs: nounwind readnone
declare i32 @f8(i32) #0

; Function Attrs: nounwind readnone
define linkonce_odr i32 @f9(ptr nocapture readnone %a0) unnamed_addr #0 align 2 {
b0:
  ret i32 11
}

define internal void @f10() {
b0:
  store ptr getelementptr inbounds ([3 x ptr], ptr @g17, i32 0, i32 2), ptr @g2, align 4, !tbaa !5
  %v0 = load { i32, i32 }, ptr @g3, align 4, !tbaa !4
  %v1 = extractvalue { i32, i32 } %v0, 1
  %v2 = getelementptr inbounds i8, ptr @g2, i32 %v1
  %v4 = extractvalue { i32, i32 } %v0, 0
  %v5 = and i32 %v4, 1
  %v6 = icmp eq i32 %v5, 0
  br i1 %v6, label %b2, label %b1

b1:                                               ; preds = %b0
  %v8 = load ptr, ptr %v2, align 4, !tbaa !5
  %v9 = add i32 %v4, -1
  %v10 = getelementptr i8, ptr %v8, i32 %v9
  %v12 = load ptr, ptr %v10, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v13 = inttoptr i32 %v4 to ptr
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v14 = phi ptr [ %v12, %b1 ], [ %v13, %b2 ]
  %v15 = tail call i32 %v14(ptr %v2)
  store i32 %v15, ptr @g4, align 4, !tbaa !0
  tail call fastcc void @f2()
  ret void
}

attributes #0 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
!5 = !{!6, !6, i64 0}
!6 = !{!"vtable pointer", !3, i64 0}
!7 = !{!8, !1, i64 0}
!8 = !{!"_ZTS2B2", !1, i64 0}
