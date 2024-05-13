; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; CHECK: rBM_info
; CHECK-NOT: ret

@sES_closure = external global [0 x i64]
declare ghccc void @sEH_info(ptr noalias nocapture, ptr noalias nocapture, ptr noalias nocapture, i64, i64, i64) align 8

define ghccc void @rBM_info(ptr noalias nocapture %Base_Arg, ptr noalias nocapture %Sp_Arg, ptr noalias nocapture %Hp_Arg, i64 %R1_Arg, i64 %R2_Arg, i64 %R3_Arg) nounwind align 8 {
c263:
  %ln265 = getelementptr inbounds i64, ptr %Sp_Arg, i64 -2
  %ln266 = ptrtoint ptr %ln265 to i64
  %ln268 = icmp ult i64 %ln266, %R3_Arg
  br i1 %ln268, label %c26a, label %n26p

n26p:                                             ; preds = %c263
  %and = and i64 ptrtoint (ptr @sES_closure to i64), 7
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %c1ZP.i, label %n1ZQ.i

n1ZQ.i:                                           ; preds = %n26p
  %ln1ZT.i = load i64, ptr @sES_closure, align 8
  %ln1ZU.i = inttoptr i64 %ln1ZT.i to ptr
  tail call ghccc void %ln1ZU.i(ptr %Base_Arg, ptr %Sp_Arg, ptr %Hp_Arg, i64 ptrtoint (ptr @sES_closure to i64), i64 ptrtoint (ptr @sES_closure to i64), i64 %R3_Arg) nounwind
  br label %rBL_info.exit

c1ZP.i:                                           ; preds = %n26p
  tail call ghccc void @sEH_info(ptr %Base_Arg, ptr %Sp_Arg, ptr %Hp_Arg, i64 ptrtoint (ptr @sES_closure to i64), i64 ptrtoint (ptr @sES_closure to i64), i64 %R3_Arg) nounwind
  br label %rBL_info.exit

rBL_info.exit:                                    ; preds = %c1ZP.i, %n1ZQ.i
  ret void

c26a:                                             ; preds = %c263
  %ln27h = getelementptr inbounds i64, ptr %Base_Arg, i64 -2
  %ln27j = load i64, ptr %ln27h, align 8
  %ln27k = inttoptr i64 %ln27j to ptr
  tail call ghccc void %ln27k(ptr %Base_Arg, ptr %Sp_Arg, ptr %Hp_Arg, i64 %R1_Arg, i64 %R2_Arg, i64 %R3_Arg) nounwind
  ret void
}
