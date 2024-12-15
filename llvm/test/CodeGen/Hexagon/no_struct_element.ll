; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; CHECK-NOT: 4294967295

target triple = "hexagon"

%s.0 = type {}
%s.1 = type { %s.2, %s.6 }
%s.2 = type { %s.3 }
%s.3 = type { %s.4 }
%s.4 = type { %s.5 }
%s.5 = type { i32 }
%s.6 = type { ptr, ptr }

@g0 = internal global %s.0 zeroinitializer, align 1
@g1 = private unnamed_addr constant [23 x i8] c"......................\00", align 1

; Function Attrs: nounwind
define void @f0(ptr %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = getelementptr inbounds i8, ptr %a0, i32 1028
  store volatile i32 0, ptr %v0, align 4
  %v3 = load volatile i32, ptr %v0, align 4
  store volatile i32 %v3, ptr %v1, align 4
  %v4 = getelementptr inbounds i8, ptr %a0, i32 1032
  call void @f1(ptr %v4, ptr @g1, ptr @g0) #0
  ret void
}

; Function Attrs: nounwind
declare void @f1(ptr, ptr, ptr) #0

attributes #0 = { nounwind }
