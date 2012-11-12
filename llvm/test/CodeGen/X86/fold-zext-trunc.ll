; RUN: llc < %s | FileCheck %s
; PR9055
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i686-pc-linux-gnu"

%struct.S0 = type { i32, [2 x i8], [2 x i8], [4 x i8] }

@g_98 = common global %struct.S0 zeroinitializer, align 4

define void @foo() nounwind {
; CHECK: movzbl
; CHECK-NOT: movzbl
; CHECK: calll
entry:
  %tmp17 = load i8* getelementptr inbounds (%struct.S0* @g_98, i32 0, i32 1, i32 0), align 4
  %tmp54 = zext i8 %tmp17 to i32
  %foo = load i32* bitcast (i8* getelementptr inbounds (%struct.S0* @g_98, i32 0, i32 1, i32 0) to i32*), align 4
  %conv.i = trunc i32 %foo to i8
  tail call void @func_12(i32 %tmp54, i8 zeroext %conv.i) nounwind
  ret void
}

declare void @func_12(i32, i8 zeroext)
