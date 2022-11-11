; Test the ICBT instruction is not emitted on POWER7
; Based on the ppc64-prefetch.ll test
; RUN: not --crash llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s 2>&1 | FileCheck %s
 
declare void @llvm.prefetch(ptr, i32, i32, i32)

define void @test(ptr %a, ...) nounwind {
entry:
  call void @llvm.prefetch(ptr %a, i32 0, i32 3, i32 0)
  ret void

; FIXME: Crashing is not really the correct behavior here, we really should just emit nothing
; CHECK: Cannot select: {{0x[0-9,a-f]+|t[0-9]+}}: ch = Prefetch
; CHECK: {{0x[0-9,a-f]+|t[0-9]+}}: i32 = Constant<0>
; CHECK-NEXT: {{0x[0-9,a-f]+|t[0-9]+}}: i32 = Constant<3>
; CHECK-NEXT: {{0x[0-9,a-f]+|t[0-9]+}}: i32 = Constant<0>

}

