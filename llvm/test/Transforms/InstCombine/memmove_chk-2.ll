; Test that lib call simplification doesn't simplify __memmove_chk calls
; with the wrong prototype.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.T1 = type { [100 x i32], [100 x i32], [1024 x i8] }
%struct.T2 = type { [100 x i32], [100 x i32], [1024 x i8] }

@t1 = common global %struct.T1 zeroinitializer
@t2 = common global %struct.T2 zeroinitializer

define void @test_no_simplify() {
; CHECK: @test_no_simplify
  %dst = bitcast %struct.T1* @t1 to i8*
  %src = bitcast %struct.T2* @t2 to i8*

; CHECK-NEXT: call i8* @__memmove_chk
  call i8* @__memmove_chk(i8* %dst, i8* %src, i64 1824)
  ret void
}

declare i8* @__memmove_chk(i8*, i8*, i64)
