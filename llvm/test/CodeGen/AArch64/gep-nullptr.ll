; RUN: llc -O3 -aarch64-gep-opt=true   < %s |FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"
target triple = "aarch64--linux-gnu"

%structA = type { i8, i8, i8, i8, i8, i8, [4 x i8], i8, i8, [2 x i32], [2 x %unionMV], [4 x [2 x %unionMV]], [4 x [2 x %unionMV]], [4 x i8], i8*, i8*, i32, i8* }
%unionMV = type { i32 }

; Function Attrs: nounwind
define void @test(%structA* %mi_block) {
entry:
  br i1 undef, label %for.body13.us, label %if.else

; Just make sure we don't get a compiler ICE due to dereferncing a nullptr.
; CHECK-LABEL: test
for.body13.us:                                    ; preds = %entry
  %indvars.iv.next40 = or i64 0, 1
  %packed4.i.us.1 = getelementptr inbounds %structA, %structA* %mi_block, i64 0, i32 11, i64 0, i64 %indvars.iv.next40, i32 0
  unreachable

if.else:                                          ; preds = %entry
  ret void
}

