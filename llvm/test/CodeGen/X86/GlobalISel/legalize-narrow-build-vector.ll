; RUN: not llc -O0 -mtriple=x86_64-unknown-linux-gnu -global-isel -global-isel-abort=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ABORT
; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -global-isel -global-isel-abort=2 %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK

; A G_BUILD_VECTOR below a native vector width cannot be legalized by adding
; elements: moreElementsVector rebuilds the same illegal build vector from the
; unmerged lanes, so the legalizer must report the failure instead of
; re-enqueueing the same query forever. On this triple the <2 x i32> multiply
; is scalarized and reassembling its lanes creates the build vector below.
; ABORT: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR

; abort=2 must fall back to SelectionDAG instead of hanging.
; FALLBACK: warning: Instruction selection used fallback path for f

define void @f(ptr %p, ptr %q) {
  %a = load <2 x i32>, ptr %p
  %b = load <2 x i32>, ptr %q
  %r = mul <2 x i32> %a, %b
  store <2 x i32> %r, ptr %p
  ret void
}

; A native-width build vector is still legalized through the constant pool.
; FALLBACK-LABEL: const4x32:
; FALLBACK: movaps
define void @const4x32(ptr %p) {
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %p
  ret void
}

; A narrow build vector whose user can be widened is still legalized: the
; <3 x i32> add is widened to a native width and lowered to paddd.
; FALLBACK-LABEL: add3x32:
; FALLBACK: paddd
define <3 x i32> @add3x32(<3 x i32> %a, <3 x i32> %b) {
  %r = add <3 x i32> %a, %b
  ret <3 x i32> %r
}
