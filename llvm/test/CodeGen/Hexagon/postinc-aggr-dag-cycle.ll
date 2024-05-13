; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts
; Check for successful compilation.

target triple = "hexagon"

%s.0 = type { [1 x i32] }
%s.1 = type { %s.2, i8, %s.6 }
%s.2 = type { %s.3 }
%s.3 = type { %s.4 }
%s.4 = type { %s.5 }
%s.5 = type { i32 }
%s.6 = type { ptr, ptr }

@g0 = external constant ptr
@g1 = external global i32
@g2 = internal global %s.1 zeroinitializer, section ".data..percpu", align 4
@g3 = external global [3 x i32]
@g4 = private unnamed_addr constant [29 x i8] c"BUG: failure at %s:%d/%s()!\0A\00", align 1
@g5 = private unnamed_addr constant [22 x i8] c"kernel/stop_machine.c\00", align 1
@g6 = private unnamed_addr constant [14 x i8] c"cpu_stop_init\00", align 1
@g7 = private unnamed_addr constant [5 x i8] c"BUG!\00", align 1

; Function Attrs: nounwind
define internal i32 @f0() #0 section ".init.text" {
b0:
  %v0 = alloca i32, align 4
  %v1 = load ptr, ptr @g0, align 4, !tbaa !0
  %v3 = tail call i32 @f1(ptr %v1, i32 3, i32 0) #0
  %v4 = load i32, ptr @g1, align 4, !tbaa !4
  %v5 = icmp ult i32 %v3, %v4
  br i1 %v5, label %b1, label %b4

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v6 = phi i32 [ %v18, %b2 ], [ %v3, %b1 ]
  %v7 = tail call i32 asm "", "=r,0"(ptr @g2) #0, !srcloc !6
  %v8 = getelementptr inbounds [3 x i32], ptr @g3, i32 0, i32 %v6
  %v9 = load i32, ptr %v8, align 4, !tbaa !7
  %v10 = add i32 %v9, %v7
  %v11 = inttoptr i32 %v10 to ptr
  store volatile i32 0, ptr %v0, align 4
  %v13 = load volatile i32, ptr %v0, align 4
  store volatile i32 %v13, ptr %v11, align 4
  %v14 = getelementptr inbounds %s.1, ptr %v11, i32 0, i32 2
  store ptr %v14, ptr %v14, align 4, !tbaa !9
  %v16 = getelementptr inbounds %s.1, ptr %v11, i32 0, i32 2, i32 1
  store ptr %v14, ptr %v16, align 4, !tbaa !11
  %v17 = add i32 %v6, 1
  %v18 = tail call i32 @f1(ptr %v1, i32 3, i32 %v17) #0
  %v19 = load i32, ptr @g1, align 4, !tbaa !4
  %v20 = icmp ult i32 %v18, %v19
  br i1 %v20, label %b2, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v21 = tail call i32 @f2() #0
  %v22 = icmp eq i32 %v21, 0
  br i1 %v22, label %b6, label %b5, !prof !12

b5:                                               ; preds = %b4
  %v23 = tail call i32 (ptr, ...) @f3(ptr @g4, ptr @g5, i32 354, ptr @g6) #0
  tail call void (ptr, ...) @f4(ptr @g7) #1
  unreachable

b6:                                               ; preds = %b4
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @f1(ptr, i32, i32) #0

; Function Attrs: nounwind
declare i32 @f2() #0

; Function Attrs: nounwind
declare i32 @f3(ptr, ...) #0

; Function Attrs: noreturn
declare void @f4(ptr, ...) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { noreturn }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
!6 = !{i32 521672}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !2, i64 0}
!9 = !{!10, !1, i64 0}
!10 = !{!"list_head", !1, i64 0, !1, i64 4}
!11 = !{!10, !1, i64 4}
!12 = !{!"branch_weights", i32 64, i32 4}
