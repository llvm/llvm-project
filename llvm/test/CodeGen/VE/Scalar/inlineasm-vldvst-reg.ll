; REQUIRES: asserts
; RUN: llc < %s -mtriple=ve -mattr=+vpu -o /dev/null -debug-only=selectiondag 2>&1 | FileCheck %s

;; Check that a vector register is not bitcasted or assigned to v512* type
;; like below.  Because, inline asm assigns registers by not a type but
;; register class.
;;
;;   t26: ch,glue = inlineasm t25, TargetExternalSymbol:i64'vld $0, $2, $1', MDNode:ch<null>, TargetConstant:i64<1>, TargetConstant:i32<589834>, Register:v512i32 %4, TargetConstant:i32<262153>, Register:i64 %5, TargetConstant:i32<262153>, Register:i64 %6, t25:1
;;   t28: v512i32 = bitcast t27

define void @vldvst(ptr %p, i64 %i) nounwind {
; CHECK-NOT: v512
  %lvl = tail call i64 asm sideeffect "lea $0, 256", "=r"() nounwind
  tail call void asm sideeffect "lvl $0", "r"(i64 %lvl) nounwind
  %1 = tail call <256 x double> asm sideeffect "vld $0, $2, $1", "=v,r,r"(ptr %p, i64 %i) nounwind
  tail call void asm sideeffect "vst $0, $2, $1", "v,r,r"(<256 x double> %1, ptr %p, i64 %i) nounwind
  ret void
}
