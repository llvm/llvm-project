; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -O2 -filetype=obj < %s -o /dev/null

; The InitUndef pass replaces the placeholder input of an early-clobber inline asm
; with an INIT_UNDEF pseudo. If the packetizer pulls that INIT_UNDEF into a
; bundle, HexagonAsmPrinter must not lower it to MC, or assertion builds fail
; in HexagonMCShuffler with "!getDesc(MCII, MI).isPseudo()".

; CHECK-LABEL: f:
; CHECK:      {
; CHECK-DAG:    r{{[0-9]+}} = memw(r0+#0)
; CHECK-DAG:    r{{[0-9]+}} = memw(r0+#4)
; CHECK:      }
; CHECK-NEXT: //# InlineAsm Start
; CHECK-NEXT: {
; CHECK-NEXT:   r{{[0-9]+}} = r{{[0-9]+}}
; CHECK-NEXT: }
; CHECK-NEXT: //# InlineAsm End

; The INIT_UNDEF ends up in the same packet as the loads of the stack-passed
; arguments %G.1 and %C2. The leading arguments are needed to fill the argument
; registers so that those two are passed on the stack; do not remove them.
; CHECK-LABEL: fred:
; CHECK:      {
; CHECK-DAG:    r{{[0-9]+}} = memub(r29+#4)
; CHECK-DAG:    r{{[0-9]+}} = memw(r29+#0)
; CHECK:      }
; CHECK:      //# InlineAsm Start
; CHECK:        r[[REG0:[0-9]+]] = usr
; CHECK:        [[REG0]] = insert(r{{[0-9]+}},#1,#16)
; CHECK:      //# InlineAsm End
; CHECK:        memb(r{{[0-9]+}}+#0) = r{{[0-9]+}}.new

target triple = "hexagon"

define i32 @f(ptr %p) {
entry:
  %a0 = load i32, ptr %p, align 4
  %g1 = getelementptr i32, ptr %p, i32 1
  %a1 = load i32, ptr %g1, align 4
  %r = call i32 asm sideeffect "$0 = $1", "=&r,r"(i32 poison)
  %s0 = add i32 %r, %a0
  %s1 = add i32 %s0, %a1
  ret i32 %s1
}

define void @fred(ptr %A, double %L1, i16 %LGV, i1 %C1, ptr %G.1, i1 %C2) {
entry:
  %0 = call { i32, i32 } asm sideeffect " $0 = usr\0A $1 = $2\0A $0 = insert($1, #1, #16)\0Ausr = $0 \0A", "=&r,=&r,r"(i1 poison)
  store i1 %C2, ptr %G.1, align 1
  ret void
}

; llvm.arithmetic.fence lowers to ARITH_FENCE, which is also pseudo-only. Here
; it sits between two FP adds that can be packetized together. Unpatched, this
; input hits the same HexagonMCShuffler assertion at -O2.

; CHECK-LABEL: arith_fence:
; CHECK:       sfmpy
; CHECK:       sfadd
; CHECK:       sfadd
define float @arith_fence(ptr %p) {
entry:
  %a0 = load float, ptr %p, align 4
  %g1 = getelementptr float, ptr %p, i32 1
  %a1 = load float, ptr %g1, align 4
  %m = fmul float %a0, %a1
  %f = call float @llvm.arithmetic.fence.f32(float %m)
  %s = fadd float %f, %a0
  %t = fadd float %s, %a1
  ret float %t
}
declare float @llvm.arithmetic.fence.f32(float)
