; RUN: llc < %s -march=x86-64 | FileCheck %s
; CHECK: rBM_info
; CHECK-NOT: ret

@sES_closure = external global [0 x i64]
declare cc10 void @sEH_info(i64* noalias nocapture, i64* noalias nocapture, i64* noalias nocapture, i64, i64, i64) align 8

define cc10 void @rBM_info(i64* noalias nocapture %Base_Arg, i64* noalias nocapture %Sp_Arg, i64* noalias nocapture %Hp_Arg, i64 %R1_Arg, i64 %R2_Arg, i64 %R3_Arg) nounwind align 8 {
c263:
  %ln265 = getelementptr inbounds i64* %Sp_Arg, i64 -2
  %ln266 = ptrtoint i64* %ln265 to i64
  %ln268 = icmp ult i64 %ln266, %R3_Arg
  br i1 %ln268, label %c26a, label %n26p

n26p:                                             ; preds = %c263
  br i1 icmp ne (i64 and (i64 ptrtoint ([0 x i64]* @sES_closure to i64), i64 7), i64 0), label %c1ZP.i, label %n1ZQ.i

n1ZQ.i:                                           ; preds = %n26p
  %ln1ZT.i = load i64* getelementptr inbounds ([0 x i64]* @sES_closure, i64 0, i64 0), align 8
  %ln1ZU.i = inttoptr i64 %ln1ZT.i to void (i64*, i64*, i64*, i64, i64, i64)*
  tail call cc10 void %ln1ZU.i(i64* %Base_Arg, i64* %Sp_Arg, i64* %Hp_Arg, i64 ptrtoint ([0 x i64]* @sES_closure to i64), i64 ptrtoint ([0 x i64]* @sES_closure to i64), i64 %R3_Arg) nounwind
  br label %rBL_info.exit

c1ZP.i:                                           ; preds = %n26p
  tail call cc10 void @sEH_info(i64* %Base_Arg, i64* %Sp_Arg, i64* %Hp_Arg, i64 ptrtoint ([0 x i64]* @sES_closure to i64), i64 ptrtoint ([0 x i64]* @sES_closure to i64), i64 %R3_Arg) nounwind
  br label %rBL_info.exit

rBL_info.exit:                                    ; preds = %c1ZP.i, %n1ZQ.i
  ret void

c26a:                                             ; preds = %c263
  %ln27h = getelementptr inbounds i64* %Base_Arg, i64 -2
  %ln27j = load i64* %ln27h, align 8
  %ln27k = inttoptr i64 %ln27j to void (i64*, i64*, i64*, i64, i64, i64)*
  tail call cc10 void %ln27k(i64* %Base_Arg, i64* %Sp_Arg, i64* %Hp_Arg, i64 %R1_Arg, i64 %R2_Arg, i64 %R3_Arg) nounwind
  ret void
}
