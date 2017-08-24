; RUN: llc -mtriple=x86_64-apple-darwin -mcpu skx < %s | FileCheck %s
; This test compliments the .c test under clang/test/CodeGen/. We check 
; if the inline asm constraints are respected in the generated code.

; Function Attrs: nounwind
define void @f_Ym(i64 %m.coerce) {
; Any mmx regiter constraint
; CHECK-LABEL: f_Ym:
; CHECK:         ## InlineAsm Start
; CHECK-NEXT:    movq %mm{{[0-9]+}}, %mm1
; CHECK:         ## InlineAsm End

entry:
  %0 = tail call x86_mmx asm sideeffect "movq $0, %mm1\0A\09", "=^Ym,~{dirflag},~{fpsr},~{flags}"() 
  ret void
}

; Function Attrs: nounwind
define void @f_Yi(<4 x float> %x, <4 x float> %y, <4 x float> %z) {
; Any SSE register when SSE2 is enabled (GCC when inter-unit moves enabled)
; CHECK-LABEL: f_Yi:
; CHECK:         ## InlineAsm Start
; CHECK-NEXT:    vpaddq %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; CHECK:         ## InlineAsm End

entry:
  %0 = tail call <4 x float> asm sideeffect "vpaddq $0, $1, $2\0A\09", "=^Yi,^Yi,^Yi,~{dirflag},~{fpsr},~{flags}"(<4 x float> %y, <4 x float> %z) 
  ret void
}

; Function Attrs: nounwind
define void @f_Yt(<4 x float> %x, <4 x float> %y, <4 x float> %z) {
; Any SSE register when SSE2 is enabled
; CHECK-LABEL: f_Yt:
; CHECK:         ## InlineAsm Start
; CHECK-NEXT:    vpaddq %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; CHECK:         ## InlineAsm End

entry:
  %0 = tail call <4 x float> asm sideeffect "vpaddq $0, $1, $2\0A\09", "=^Yt,^Yt,^Yt,~{dirflag},~{fpsr},~{flags}"(<4 x float> %y, <4 x float> %z)
  ret void
}

; Function Attrs: nounwind
define void @f_Y2(<4 x float> %x, <4 x float> %y, <4 x float> %z) {
; Any SSE register when SSE2 is enabled
; CHECK-LABEL: f_Y2:
; CHECK:         ## InlineAsm Start
; CHECK-NEXT:    vpaddq %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; CHECK:         ## InlineAsm End

entry:
  %0 = tail call <4 x float> asm sideeffect "vpaddq $0, $1, $2\0A\09", "=^Y2,^Y2,^Y2,~{dirflag},~{fpsr},~{flags}"(<4 x float> %y, <4 x float> %z)
  ret void
}

; Function Attrs: nounwind
define void @f_Yz(<4 x float> %x, <4 x float> %y, <4 x float> %z) {
; xmm0 SSE register(GCC)
; CHECK-LABEL: f_Yz:
; CHECK:         ## InlineAsm Start
; CHECK-NEXT:    vpaddq %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; CHECK:         ## InlineAsm End
entry:
  %0 = tail call { <4 x float>, <4 x float> } asm sideeffect "vpaddq $0,$2,$1\0A\09vpaddq $1,$0,$2\0A\09", "=^Yi,=^Yz,^Yi,0,~{dirflag},~{fpsr},~{flags}"(<4 x float> %y, <4 x float> %z)
  ret void
}

; Function Attrs: nounwind
define void @f_Y0(<4 x float> %x, <4 x float> %y, <4 x float> %z) {
; xmm0 SSE register
; CHECK-LABEL: f_Y0:
; CHECK:         ## InlineAsm Start
; CHECK-NEXT:    vpaddq %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; CHECK:         ## InlineAsm End

entry:
  %0 = tail call { <4 x float>, <4 x float> } asm sideeffect "vpaddq $0,$2,$1\0A\09vpaddq $1,$0,$2\0A\09", "=^Yi,=^Y0,^Yi,0,~{dirflag},~{fpsr},~{flags}"(<4 x float> %y, <4 x float> %z)
  ret void
}

