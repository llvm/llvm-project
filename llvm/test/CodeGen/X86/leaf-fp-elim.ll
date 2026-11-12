; RUN: llc < %s -relocation-model=pic -mtriple=x86_64-apple-darwin | FileCheck %s
; <rdar://problem/8170192>
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0"

@msg = internal global ptr null                   ; <ptr> [#uses=1]
@.str = private constant [2 x i8] c"x\00", align 1 ; <ptr> [#uses=1]

define void @test(ptr %p) "frame-pointer"="non-leaf" nounwind optsize ssp {

; No stack frame, please.
; CHECK:     _test
; CHECK-NOT: pushq %rbp
; CHECK-NOT: movq %rsp, %rbp
; CHECK:     InlineAsm Start

entry:
  %0 = icmp eq ptr %p, null                       ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb

bb:                                               ; preds = %entry
  tail call void asm "mov $1, $0", "=*m,{cx},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(ptr) @msg, ptr @.str) nounwind
  tail call void @llvm.trap()
  unreachable

return:                                           ; preds = %entry
  ret void
}

declare void @llvm.trap() nounwind
