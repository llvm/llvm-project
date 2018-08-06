; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts
; Check for successful compilation.

target triple = "hexagon-unknown--elf"

%s.0 = type { i32 (...)** }
%s.1 = type { i32 }
%s.2 = type { %s.1 }

@g0 = global { i32, i32 } { i32 ptrtoint (i32 (%s.1*)* @f0 to i32), i32 0 }, align 4
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
@g17 = linkonce_odr unnamed_addr constant [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @g20 to i8*), i8* bitcast (i32 (%s.0*)* @f9 to i8*)]
@g18 = external global i8*
@g19 = linkonce_odr constant [3 x i8] c"1S\00"
@g20 = linkonce_odr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @g18, i32 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g19, i32 0, i32 0) }

; Function Attrs: nounwind readnone
define linkonce_odr i32 @f0(%s.1* nocapture readnone %a0) #0 align 2 {
b0:
  ret i32 11
}

; Function Attrs: nounwind readnone
define %s.0* @f1() #0 {
b0:
  ret %s.0* @g2
}

define internal fastcc void @f2() {
b0:
  %v0 = load i32, i32* @g5, align 4, !tbaa !0
  %v1 = add nsw i32 %v0, 5
  store i32 %v1, i32* @g5, align 4, !tbaa !0
  %v2 = load { i32, i32 }, { i32, i32 }* @g3, align 4, !tbaa !4
  %v3 = extractvalue { i32, i32 } %v2, 1
  %v4 = getelementptr inbounds i8, i8* bitcast (%s.0* @g2 to i8*), i32 %v3
  %v5 = bitcast i8* %v4 to %s.0*
  %v6 = extractvalue { i32, i32 } %v2, 0
  %v7 = and i32 %v6, 1
  %v8 = icmp eq i32 %v7, 0
  br i1 %v8, label %b2, label %b1

b1:                                               ; preds = %b0
  %v9 = bitcast i8* %v4 to i8**
  %v10 = load i8*, i8** %v9, align 4, !tbaa !5
  %v11 = add i32 %v6, -1
  %v12 = getelementptr i8, i8* %v10, i32 %v11
  %v13 = bitcast i8* %v12 to i32 (%s.0*)**
  %v14 = load i32 (%s.0*)*, i32 (%s.0*)** %v13, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v15 = inttoptr i32 %v6 to i32 (%s.0*)*
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v16 = phi i32 (%s.0*)* [ %v14, %b1 ], [ %v15, %b2 ]
  %v17 = tail call i32 %v16(%s.0* %v5)
  store i32 %v17, i32* @g6, align 4, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
define i32 @f3() #0 {
b0:
  %v0 = alloca %s.2, align 4
  %v1 = alloca %s.2, align 4
  tail call void @f4()
  tail call void @f5()
  tail call void (i8*, ...) @f6(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @g7, i32 0, i32 0))
  tail call void (i8*, ...) @f6(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g8, i32 0, i32 0), i8* getelementptr inbounds ([43 x i8], [43 x i8]* @g9, i32 0, i32 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @g10, i32 0, i32 0))
  %v2 = load { i32, i32 }, { i32, i32 }* @g0, align 4, !tbaa !4
  %v3 = extractvalue { i32, i32 } %v2, 1
  %v4 = bitcast %s.2* %v0 to i8*
  %v5 = getelementptr inbounds i8, i8* %v4, i32 %v3
  %v6 = bitcast i8* %v5 to %s.2*
  %v7 = extractvalue { i32, i32 } %v2, 0
  %v8 = and i32 %v7, 1
  %v9 = icmp eq i32 %v8, 0
  br i1 %v9, label %b1, label %b2

b1:                                               ; preds = %b0
  %v10 = inttoptr i32 %v7 to i32 (%s.2*)*
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v11 = phi i32 (%s.2*)* [ %v10, %b1 ], [ undef, %b0 ]
  %v12 = call i32 %v11(%s.2* %v6)
  %v13 = icmp eq i32 %v12, 11
  br i1 %v13, label %b4, label %b3

b3:                                               ; preds = %b2
  store i32 1, i32* @g11, align 4, !tbaa !0
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v14 = call i32 @f7()
  call void @f5()
  call void (i8*, ...) @f6(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g8, i32 0, i32 0), i8* getelementptr inbounds ([46 x i8], [46 x i8]* @g12, i32 0, i32 0), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @g13, i32 0, i32 0))
  %v15 = getelementptr inbounds %s.2, %s.2* %v1, i32 0, i32 0, i32 0
  store i32 11, i32* %v15, align 4, !tbaa !7
  %v16 = load i32, i32* @g1, align 4, !tbaa !4
  %v17 = bitcast %s.2* %v1 to i8*
  %v18 = getelementptr inbounds i8, i8* %v17, i32 %v16
  %v19 = bitcast i8* %v18 to i32*
  %v20 = load i32, i32* %v19, align 4, !tbaa !0
  %v21 = icmp eq i32 %v20, 11
  br i1 %v21, label %b6, label %b5

b5:                                               ; preds = %b4
  store i32 1, i32* @g11, align 4, !tbaa !0
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v22 = call i32 @f7()
  call void @f5()
  call void (i8*, ...) @f6(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g14, i32 0, i32 0), i8* getelementptr inbounds ([51 x i8], [51 x i8]* @g15, i32 0, i32 0), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @g16, i32 0, i32 0))
  %v23 = load i32, i32* @g4, align 4, !tbaa !0
  %v24 = icmp eq i32 %v23, 11
  br i1 %v24, label %b8, label %b7

b7:                                               ; preds = %b6
  store i32 1, i32* @g11, align 4, !tbaa !0
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v25 = call i32 @f7()
  call void @f5()
  call void (i8*, ...) @f6(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g14, i32 0, i32 0), i8* getelementptr inbounds ([51 x i8], [51 x i8]* @g15, i32 0, i32 0), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @g16, i32 0, i32 0))
  %v26 = load i32, i32* @g6, align 4, !tbaa !0
  %v27 = icmp eq i32 %v26, 11
  br i1 %v27, label %b10, label %b9

b9:                                               ; preds = %b8
  store i32 1, i32* @g11, align 4, !tbaa !0
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
declare void @f6(i8*, ...) #0

; Function Attrs: nounwind readnone
declare i32 @f7() #0

; Function Attrs: nounwind readnone
declare i32 @f8(i32) #0

; Function Attrs: nounwind readnone
define linkonce_odr i32 @f9(%s.0* nocapture readnone %a0) unnamed_addr #0 align 2 {
b0:
  ret i32 11
}

define internal void @f10() {
b0:
  store i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @g17, i32 0, i32 2) to i32 (...)**), i32 (...)*** getelementptr inbounds (%s.0, %s.0* @g2, i32 0, i32 0), align 4, !tbaa !5
  %v0 = load { i32, i32 }, { i32, i32 }* @g3, align 4, !tbaa !4
  %v1 = extractvalue { i32, i32 } %v0, 1
  %v2 = getelementptr inbounds i8, i8* bitcast (%s.0* @g2 to i8*), i32 %v1
  %v3 = bitcast i8* %v2 to %s.0*
  %v4 = extractvalue { i32, i32 } %v0, 0
  %v5 = and i32 %v4, 1
  %v6 = icmp eq i32 %v5, 0
  br i1 %v6, label %b2, label %b1

b1:                                               ; preds = %b0
  %v7 = bitcast i8* %v2 to i8**
  %v8 = load i8*, i8** %v7, align 4, !tbaa !5
  %v9 = add i32 %v4, -1
  %v10 = getelementptr i8, i8* %v8, i32 %v9
  %v11 = bitcast i8* %v10 to i32 (%s.0*)**
  %v12 = load i32 (%s.0*)*, i32 (%s.0*)** %v11, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v13 = inttoptr i32 %v4 to i32 (%s.0*)*
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v14 = phi i32 (%s.0*)* [ %v12, %b1 ], [ %v13, %b2 ]
  %v15 = tail call i32 %v14(%s.0* %v3)
  store i32 %v15, i32* @g4, align 4, !tbaa !0
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
