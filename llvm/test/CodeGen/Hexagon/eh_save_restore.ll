; RUN: llc -O3 -march=hexagon -hexagon-small-data-threshold=0 -disable-packetizer < %s | FileCheck %s

; This test was orignally written to test that we don't save an entire double
; register if only one of the integer registers needs to be saved. The problem
; occurs in exception handling, which only emit information for the registers
; in the callee saved list (and not complete double registers unless both
; parts of the double registers are used).
; Overtime, we evolved in to saving the double register and updating the debug
; information to cover the entire double register.

; Disable the packetizer to avoid complications caused by potentially
; packetizing one of the stores with allocframe, which would change the
; relative order of the stores with the CFI instructions.

; CHECK: cfi_startproc
; CHECK-DAG: cfi_offset r16
; CHECK-DAG: cfi_offset r17
; CHECK-DAG: cfi_offset r18
; CHECK-DAG: cfi_offset r19
; CHECK: memd(r29+{{.*}}) = r17:16
; CHECK: memd(r29+{{.*}}) = r19:18

%s.0 = type { i32 }

@g0 = global i32 0, align 4
@g1 = external constant ptr

; Function Attrs: noreturn
define void @f0(i64 %a0) #0 personality ptr @f2 {
b0:
  %v0 = alloca %s.0, align 4
  %v1 = trunc i64 %a0 to i32
  %v2 = lshr i64 %a0, 32
  %v3 = trunc i64 %v2 to i32
  store i32 0, ptr %v0, align 4, !tbaa !0
  %v5 = load i32, ptr @g0, align 4, !tbaa !5
  %v6 = or i32 %v5, 1
  store i32 %v6, ptr @g0, align 4, !tbaa !5
  %v7 = call ptr @f1(i32 4) #1
  %v10 = getelementptr inbounds i8, ptr %v0, i32 %v3
  %v12 = and i32 %v1, 1
  %v13 = icmp eq i32 %v12, 0
  br i1 %v13, label %b2, label %b1

b1:                                               ; preds = %b0
  %v15 = load ptr, ptr %v10, align 4
  %v16 = add i32 %v1, -1
  %v17 = getelementptr i8, ptr %v15, i32 %v16
  %v19 = load ptr, ptr %v17, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v20 = inttoptr i32 %v1 to ptr
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v21 = phi ptr [ %v19, %b1 ], [ %v20, %b2 ]
  %v22 = invoke i32 %v21(ptr %v10)
          to label %b4 unwind label %b5

b4:                                               ; preds = %b3
  store i32 %v22, ptr %v7, align 4, !tbaa !5
  call void @f4(ptr %v7, ptr @g1, ptr null) #2
  unreachable

b5:                                               ; preds = %b3
  %v23 = landingpad { ptr, i32 }
          cleanup
  call void @f3(ptr %v7) #1
  resume { ptr, i32 } %v23
}

declare ptr @f1(i32)

declare i32 @f2(...)

declare void @f3(ptr)

declare void @f4(ptr, ptr, ptr)

attributes #0 = { noreturn "target-cpu"="hexagonv55" }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!0 = !{!1, !2, i64 0}
!1 = !{!"_ZTS1A", !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!2, !2, i64 0}
