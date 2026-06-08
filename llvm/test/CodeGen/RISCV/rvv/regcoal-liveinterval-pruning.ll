; RUN: llc -mtriple=riscv64 < %s | FileCheck %s

define <8 x i64> @ID156249() #0 {
; CHECK-LABEL: ID156249:

; Entry setup
; CHECK:       vsetivli {{.*}} e64
; CHECK-NEXT:  vmv.v.i

entry:
  br label %ac

ac:
  %e.2 = phi <8 x i64> [ zeroinitializer, %entry ], [ %vecins, %ac ], [ %vecins4, %asm.fallthrough3 ], [ %vecins2, %asm.fallthrough ]
  %vecins = insertelement <8 x i64> %e.2, i64 0, i64 0

; First scalar insert
; CHECK:       vsetivli {{.*}} e64
; CHECK-NEXT:  vmv.s.x

  callbr void asm sideeffect "", "!i"()
          to label %asm.fallthrough [label %ac]

asm.fallthrough:
  %vecins2 = shufflevector <8 x i64> %vecins, <8 x i64> splat (i64 1),
                             <8 x i32> <i32 0, i32 1, i32 2, i32 10, i32 4, i32 5, i32 6, i32 7>

; Shuffle lowering sequence
; CHECK:       vsetivli {{.*}} e8
; CHECK-NEXT:  vmv.v.i
; CHECK-NEXT:  vsetivli {{.*}} e64
; CHECK-NEXT:  vmv.v.i
; CHECK-NEXT:  vslideup.vi
; CHECK-NEXT:  vmv.v.v

  callbr void asm sideeffect "", "!i"()
          to label %asm.fallthrough3 [label %ac]

asm.fallthrough3:
  %vecins4 = insertelement <8 x i64> %vecins2, i64 0, i64 0

; Second scalar insert
; CHECK:       vsetivli {{.*}} e64
; CHECK-NEXT:  vmv.s.x

  callbr void asm sideeffect "", "!i"()
          to label %asm.fallthrough5 [label %ac]

asm.fallthrough5:
  %or = or <8 x i64> %vecins2, %vecins4

; Final combine + return
; CHECK:       vsetivli {{.*}} e64
; CHECK-NEXT:  vor.vv
; CHECK-NEXT:  ret

  ret <8 x i64> %or
}

attributes #0 = { "target-features"="+v" }
