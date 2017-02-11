; RUN: llc -O2 -march=hexagon < %s | FileCheck %s
; CHECK: cmp.eq(r{{[0-9]+}},#0)
; Check that the result of the builtin is not stored directly, i.e. that
; there is an instruction that converts it to {0,1} from {0,-1}. Right now
; the instruction is "r4 = !cmp.eq(r0, #0)".

@var = common global i32 0, align 4
declare i32 @llvm.hexagon.C2.cmpgtup(i64,i64) nounwind

define void @foo(i64 %a98, i64 %a100) nounwind {
entry:
  %a101 = tail call i32 @llvm.hexagon.C2.cmpgtup(i64 %a98, i64 %a100)
  %tobool250 = icmp eq i32 %a101, 0
  %a102 = zext i1 %tobool250 to i8
  %detected.0 = xor i8 %a102, 1
  %conv253 = zext i8 %detected.0 to i32
  store i32 %conv253, i32* @var, align 4
  ret void
}
