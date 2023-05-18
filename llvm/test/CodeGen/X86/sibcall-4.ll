; RUN: llc < %s -mtriple=i386-pc-linux-gnu | FileCheck %s
; pr7610

define ghccc void @t(ptr %Base_Arg, ptr %Sp_Arg, ptr %Hp_Arg, i32 %R1_Arg) nounwind {
cm1:
; CHECK-LABEL: t:
; CHECK: jmpl *%eax
  %nm3 = getelementptr i32, ptr %Sp_Arg, i32 1
  %nm9 = load i32, ptr %Sp_Arg
  %nma = inttoptr i32 %nm9 to ptr
  tail call ghccc void %nma(ptr %Base_Arg, ptr %nm3, ptr %Hp_Arg, i32 %R1_Arg) nounwind
  ret void
}
