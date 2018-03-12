; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: .cfi_offset r31, -4
; CHECK: .cfi_offset r30, -8
; CHECK: .cfi_offset r17, -12
; CHECK: .cfi_offset r16, -16

@g0 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@g1 = external constant i8*
@g2 = private unnamed_addr constant [15 x i8] c"blah blah blah\00", align 1
@g3 = external constant i8*
@g4 = private unnamed_addr constant [2 x i8] c"{\00"
@g5 = private unnamed_addr constant [2 x i8] c"}\00"
@g6 = private unnamed_addr constant [27 x i8] c"FAIL:Unexpected exception.\00"

; Function Attrs: nounwind
declare i32 @f0(i8* nocapture readonly, ...) #0

; Function Attrs: nounwind
define void @f1(i32 %a0) #0 {
b0:
  %v0 = tail call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i32 %a0)
  ret void
}

define i32 @f2(i32 %a0, i8** nocapture readnone %a1) personality i8* bitcast (i32 (...)* @f5 to i8*) {
b0:
  %v0 = tail call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i32 %a0) #0
  %v1 = tail call i32 @f8(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @g4, i32 0, i32 0)) #0
  %v2 = tail call i8* @f3(i32 4) #0
  %v3 = bitcast i8* %v2 to i8**
  store i8* getelementptr inbounds ([15 x i8], [15 x i8]* @g2, i32 0, i32 0), i8** %v3, align 4, !tbaa !0
  invoke void @f4(i8* %v2, i8* bitcast (i8** @g1 to i8*), i8* null) #2
          to label %b9 unwind label %b1

b1:                                               ; preds = %b0
  %v4 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @g1 to i8*)
          catch i8* null
  %v5 = extractvalue { i8*, i32 } %v4, 0
  %v6 = extractvalue { i8*, i32 } %v4, 1
  %v7 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @g1 to i8*)) #0
  %v8 = icmp eq i32 %v6, %v7
  %v9 = tail call i8* @f6(i8* %v5) #0
  br i1 %v8, label %b2, label %b3

b2:                                               ; preds = %b1
  tail call void @f7() #0
  br label %b4

b3:                                               ; preds = %b1
  %v10 = tail call i32 @f8(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @g6, i32 0, i32 0))
  tail call void @f7()
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v11 = tail call i32 @f8(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @g5, i32 0, i32 0)) #0
  %v12 = tail call i32 @f8(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @g4, i32 0, i32 0)) #0
  %v13 = tail call i8* @f3(i32 4) #0
  %v14 = bitcast i8* %v13 to i32*
  store i32 777, i32* %v14, align 4, !tbaa !4
  invoke void @f4(i8* %v13, i8* bitcast (i8** @g3 to i8*), i8* null) #2
          to label %b9 unwind label %b5

b5:                                               ; preds = %b4
  %v15 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @g3 to i8*)
          catch i8* null
  %v16 = extractvalue { i8*, i32 } %v15, 0
  %v17 = extractvalue { i8*, i32 } %v15, 1
  %v18 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @g3 to i8*)) #0
  %v19 = icmp eq i32 %v17, %v18
  %v20 = tail call i8* @f6(i8* %v16) #0
  br i1 %v19, label %b6, label %b7

b6:                                               ; preds = %b5
  tail call void @f7() #0
  br label %b8

b7:                                               ; preds = %b5
  %v21 = tail call i32 @f8(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @g6, i32 0, i32 0))
  tail call void @f7()
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v22 = tail call i32 @f8(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @g5, i32 0, i32 0)) #0
  ret i32 0

b9:                                               ; preds = %b4, %b0
  unreachable
}

declare i8* @f3(i32)

declare void @f4(i8*, i8*, i8*)

declare i32 @f5(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @f6(i8*)

declare void @f7()

; Function Attrs: nounwind
declare i32 @f8(i8* nocapture readonly) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
