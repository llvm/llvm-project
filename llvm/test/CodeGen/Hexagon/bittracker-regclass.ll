; RUN: llc -mtriple=hexagon -mcpu=hexagonv75 -mattr=+hvxv75,+hvx-length64b,-small-data < %s | FileCheck %s

; Test that the compiler generates code, and doesn't crash, when the compiler
; creates a DoubleReg value with an IntLow8Reg value. The BitTracker pass
; needs to handle this register class.

; CHECK: [[REG:r[0-9]+:[0-9]+]] = combine(#33,#32)
; CHECK: memd({{.*}}) = [[REG]]

@out = external dso_local global [100 x i32], align 512
@in55 = external dso_local global [55 x i32], align 256
@.str.3 = external dso_local unnamed_addr constant [29 x i8], align 1

define dso_local void @main(i1 %cond) local_unnamed_addr #0 {
entry:
  br label %for.body.i198

for.body.i198:
  br i1 %cond, label %for.body34.preheader, label %for.body.i198

for.body34.preheader:
  %wide.load269.5 = load <16 x i32>, ptr getelementptr inbounds ([100 x i32], ptr @out, i32 0, i32 80), align 64
  %0 = add nsw <16 x i32> %wide.load269.5, zeroinitializer
  %rdx.shuf270 = shufflevector <16 x i32> %0, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx271 = add <16 x i32> %0, %rdx.shuf270
  %bin.rdx273 = add <16 x i32> %bin.rdx271, zeroinitializer
  %bin.rdx275 = add <16 x i32> %bin.rdx273, zeroinitializer
  %bin.rdx277 = add <16 x i32> %bin.rdx275, zeroinitializer
  %1 = extractelement <16 x i32> %bin.rdx277, i32 0
  %add45 = add nsw i32 0, %1
  %add45.1 = add nsw i32 0, %add45
  %add45.2 = add nsw i32 0, %add45.1
  %add45.3 = add nsw i32 0, %add45.2
  call void (ptr, ...) @printf(ptr getelementptr inbounds ([29 x i8], ptr @.str.3, i32 0, i32 0), i32 %add45.3) #2
  store i32 32, ptr getelementptr inbounds ([55 x i32], ptr @in55, i32 0, i32 32), align 128
  store i32 33, ptr getelementptr inbounds ([55 x i32], ptr @in55, i32 0, i32 33), align 4
  ret void
}

declare dso_local void @printf(ptr, ...) local_unnamed_addr #1
