; RUN: llc -march=hexagon < %s | FileCheck %s
;
; This used to crash.
; CHECK: call f1

target triple = "hexagon-unknown--elf"

%struct.0 = type { [5 x i32] }
%struct.2 = type { i32, i32, i32, ptr }
%struct.1 = type { ptr, i32, i32, i32 }

@g0 = external hidden unnamed_addr constant [52 x i8], align 1
@g1 = external hidden unnamed_addr constant [3 x i8], align 1

declare extern_weak void @f0(i32, ptr, i32, ptr, ...) #0
declare void @f1(ptr, i32) #0

define void @fred(ptr %a0) #0 {
b1:
  %v2 = alloca %struct.0, align 4
  %v3 = alloca %struct.2, i32 undef, align 8
  br i1 undef, label %b5, label %b4

b4:                                               ; preds = %b1
  br label %b7

b5:                                               ; preds = %b5, %b1
  %v6 = getelementptr inbounds %struct.2, ptr %v3, i32 undef, i32 3
  store ptr undef, ptr %v6, align 4
  br label %b5

b7:                                               ; preds = %b10, %b4
  %v8 = call i32 @llvm.hexagon.V6.extractw(<16 x i32> zeroinitializer, i32 0)
  br i1 icmp eq (ptr @f0, ptr null), label %b11, label %b9

b9:                                               ; preds = %b7
  call void (i32, ptr, i32, ptr, ...) @f0(i32 2, ptr @g0, i32 2346, ptr @g1, i32 %v8)
  unreachable

b10:                                              ; preds = %b11
  call void @f1(ptr nonnull %v2, i32 28)
  br label %b7

b11:                                              ; preds = %b11, %b7
  br i1 undef, label %b10, label %b11
}

declare i32 @llvm.hexagon.V6.extractw(<16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
