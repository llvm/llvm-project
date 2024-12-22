; RUN: llc -march=hexagon < %s | FileCheck %s

; Look for only one declaration of the const struct.
; CHECK: g0:
; CHECK-NOT: g02:

target triple = "hexagon"

%s.8 = type { %s.9, ptr, ptr }
%s.9 = type { i16, i16, i32 }
%s.0 = type { i32, ptr, ptr, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, [32 x %s.4], i32, i16, i16, i16, i16, [16 x %s.7], i16 }
%s.1 = type { i16, i8, i8, i32, %s.2 }
%s.2 = type { %s.3, ptr }
%s.3 = type { ptr }
%s.4 = type { ptr, ptr, i16, i32 }
%s.5 = type { %s.6, ptr }
%s.6 = type { i16, i8, i8, i32 }
%s.7 = type { ptr, i32 }
%s.11 = type { i32, ptr }
%s.12 = type opaque

@g0 = internal constant %s.8 { %s.9 { i16 531, i16 0, i32 16 }, ptr @g1, ptr @g2 }, align 4
@g1 = private unnamed_addr constant [48 x i8] c"In task 0x%x, Assertion heap_ptr != NULL failed\00", align 8
@g2 = private unnamed_addr constant [10 x i8] c"xxxxxxx.c\00", align 8

; Function Attrs: nounwind
define void @f0(ptr %a0) #0 {
b0:
  %v0 = icmp eq ptr %a0, null
  br i1 %v0, label %b1, label %b4

b1:                                               ; preds = %b0
  %v1 = tail call ptr @f1() #0
  %v2 = icmp eq ptr %v1, null
  br i1 %v2, label %b3, label %b2

b2:                                               ; preds = %b1
  %v3 = ptrtoint ptr %v1 to i32
  tail call void @f2(ptr @g0, i32 %v3, i32 0, i32 0) #0
  br label %b5

b3:                                               ; preds = %b1
  tail call void @f3(ptr @g0) #0
  br label %b5

b4:                                               ; preds = %b0
  %v4 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 9
  store ptr @f4, ptr %v4, align 4, !tbaa !0
  %v5 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 10
  store ptr @f5, ptr %v5, align 4, !tbaa !0
  br label %b5

b5:                                               ; preds = %b4, %b3, %b2
  ret void
}

declare ptr @f1()

declare void @f2(ptr, i32, i32, i32)

declare void @f3(ptr)

declare void @f4(ptr)

declare void @f5(ptr)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
