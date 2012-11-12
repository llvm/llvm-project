; RUN: opt < %s -S -scalarrepl | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%union.anon = type { <4 x float> }

; CHECK: @test
; CHECK-NOT: alloca

define void @test() nounwind {
entry:
  %u = alloca %union.anon, align 16
  %u164 = bitcast %union.anon* %u to [4 x i32]*
  %arrayidx165 = getelementptr inbounds [4 x i32]* %u164, i32 0, i32 0
  store i32 undef, i32* %arrayidx165, align 4
  %v186 = bitcast %union.anon* %u to <4 x float>*
  store <4 x float> undef, <4 x float>* %v186, align 16
  ret void
}
