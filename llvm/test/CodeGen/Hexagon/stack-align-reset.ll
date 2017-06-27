; RUN: llc -march=hexagon < %s | FileCheck %s
;
; This used to crash.
; CHECK: call f1

target triple = "hexagon-unknown--elf"

%struct.0 = type { [5 x i32] }
%struct.2 = type { i32, i32, i32, %struct.1* }
%struct.1 = type { i16*, i32, i32, i32 }

@g0 = external hidden unnamed_addr constant [52 x i8], align 1
@g1 = external hidden unnamed_addr constant [3 x i8], align 1

declare extern_weak void @f0(i32, i8*, i32, i8*, ...) #0
declare void @f1(%struct.0*, i32) #0

define void @fred(i8* %a0) #0 {
b1:
  %v2 = alloca %struct.0, align 4
  %v3 = alloca %struct.2, i32 undef, align 8
  br i1 undef, label %b5, label %b4

b4:                                               ; preds = %b1
  br label %b7

b5:                                               ; preds = %b5, %b1
  %v6 = getelementptr inbounds %struct.2, %struct.2* %v3, i32 undef, i32 3
  store %struct.1* undef, %struct.1** %v6, align 4
  br label %b5

b7:                                               ; preds = %b10, %b4
  %v8 = call i32 @llvm.hexagon.V6.extractw(<16 x i32> zeroinitializer, i32 0)
  br i1 icmp eq (void (i32, i8*, i32, i8*, ...)* @f0, void (i32, i8*, i32, i8*, ...)* null), label %b11, label %b9

b9:                                               ; preds = %b7
  call void (i32, i8*, i32, i8*, ...) @f0(i32 2, i8* getelementptr inbounds ([52 x i8], [52 x i8]* @g0, i32 0, i32 0), i32 2346, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g1, i32 0, i32 0), i32 %v8)
  unreachable

b10:                                              ; preds = %b11
  call void @f1(%struct.0* nonnull %v2, i32 28)
  br label %b7

b11:                                              ; preds = %b11, %b7
  br i1 undef, label %b10, label %b11
}

declare i32 @llvm.hexagon.V6.extractw(<16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,-hvx-double" }
attributes #1 = { nounwind readnone }
