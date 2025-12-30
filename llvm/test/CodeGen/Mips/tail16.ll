; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic < %s | FileCheck %s

; Function Attrs: nounwind optsize
define float @h()  {
entry:
  %call = tail call float @g() 
  ret float %call
; CHECK:	.ent	h
; CHECK: 	save	$16, $ra, $18, 32
; CHECK: 	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_0)(${{[0-9]+}})
; CHECK: 	restore	$16, $ra, $18, 32
; CHECK: 	.end	h
}

; Function Attrs: optsize
declare float @g(...) 




