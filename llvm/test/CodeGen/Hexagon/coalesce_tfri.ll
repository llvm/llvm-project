; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

@g0 = external global i32
@g1 = external global i32, align 4
@g2 = external hidden unnamed_addr constant [49 x i8], align 8
@g3 = external hidden unnamed_addr constant [76 x i8], align 8
@g4 = external unnamed_addr constant { i8*, i8* }
@g5 = external hidden unnamed_addr constant [36 x i8], align 8

declare void @f0()

declare i32 @f1()

declare i32 @f2(i32)

declare void @f3()

; Function Attrs: nounwind
declare void ()* @f4(void ()*) #0

; Function Attrs: nounwind
declare void ()* @f5(void ()*) #0

; CHECK: f6:
; CHECK-DAG: call f4
; CHECK-DAG: r0 = ##f3
; CHECK-DAG: call f5
; CHECK-DAG: r0 = ##f0
; CHECK-DAG: call f8
; CHECK-DAG: r0 = ##g2
; CHECK-DAG: call f9
; CHECK-DAG: call f8
; CHECK-DAG: r0 = ##g3
; CHECK-DAG: call f10
; CHECK-DAG: r0 = #4
; CHECK-DAG: r{{[0-9]+}} = ##g1
define i32 @f6() personality i8* bitcast (i32 (...)* @f11 to i8*) {
b0:
  tail call void @f7()
  %v0 = tail call void ()* @f4(void ()* @f3) #0
  %v1 = tail call void ()* @f5(void ()* @f0) #0
  tail call void (i8*, ...) @f8(i8* getelementptr inbounds ([49 x i8], [49 x i8]* @g2, i32 0, i32 0))
  tail call void @f9()
  tail call void (i8*, ...) @f8(i8* getelementptr inbounds ([76 x i8], [76 x i8]* @g3, i32 0, i32 0))
  %v2 = tail call i8* @f10(i32 4) #0
  %v3 = load i32, i32* @g1, align 4, !tbaa !0
  %v4 = add nsw i32 %v3, 1
  store i32 %v4, i32* @g1, align 4, !tbaa !0
  invoke void @f12(i8* %v2, i8* bitcast ({ i8*, i8* }* @g4 to i8*), i8* null) #1
          to label %b7 unwind label %b1

b1:                                               ; preds = %b0
  %v5 = landingpad { i8*, i32 }
          catch i8* null
  %v6 = extractvalue { i8*, i32 } %v5, 0
  %v7 = tail call i8* @f13(i8* %v6) #0
  store i32 0, i32* @g1, align 4, !tbaa !0
  invoke void @f14() #1
          to label %b7 unwind label %b2

b2:                                               ; preds = %b1
  %v8 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @f15()
          to label %b3 unwind label %b6

b3:                                               ; preds = %b2
  %v9 = extractvalue { i8*, i32 } %v8, 0
  %v10 = tail call i8* @f13(i8* %v9) #0
  tail call void @f15()
  %v11 = load i32, i32* @g1, align 4, !tbaa !0
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %b5, label %b4

b4:                                               ; preds = %b3
  tail call void (i8*, ...) @f8(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @g5, i32 0, i32 0))
  store i32 1, i32* @g0, align 4, !tbaa !0
  br label %b5

b5:                                               ; preds = %b4, %b3
  %v13 = tail call i32 @f1()
  %v14 = tail call i32 @f2(i32 1)
  ret i32 %v14

b6:                                               ; preds = %b2
  %v15 = landingpad { i8*, i32 }
          catch i8* null
  tail call void @f16() #2
  unreachable

b7:                                               ; preds = %b1, %b0
  unreachable
}

declare void @f7()

declare void @f8(i8*, ...)

declare void @f9()

declare i8* @f10(i32)

declare i32 @f11(...)

declare void @f12(i8*, i8*, i8*)

declare i8* @f13(i8*)

declare void @f14()

declare void @f15()

declare void @f16()

attributes #0 = { nounwind }
attributes #1 = { noreturn }
attributes #2 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
