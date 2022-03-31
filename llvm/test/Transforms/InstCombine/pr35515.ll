; RUN: opt -S -passes=instcombine < %s | FileCheck %s

@g_40 = external global i8, align 2
@g_461 = external global [6 x i8], align 2
@g_49 = external local_unnamed_addr global { i8, i8, i8, i8, i8 }, align 2

; CHECK-LABEL: @func_24(
define fastcc void @func_24() {
entry:
  %bf.load81 = load i40, i40* bitcast ({ i8, i8, i8, i8, i8 }* @g_49 to i40*), align 2
  %bf.clear = and i40 %bf.load81, -274869518337
  %bf.set = or i40 %bf.clear, shl (i40 zext (i1 icmp sgt (i32 zext (i1 icmp eq (i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g_461, i64 0, i64 2), i8* @g_40) to i32), i32 0) to i40), i40 23)
  %tmp = lshr i40 %bf.set, 23
  %tmp1 = trunc i40 %tmp to i32
  %tmp2 = and i32 1, %tmp1
  %tmp3 = shl nuw nsw i32 %tmp2, 23
  %bf.shl154 = zext i32 %tmp3 to i40
  %bf.set156 = or i40 %bf.clear, %bf.shl154
  unreachable
}
