;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=BOTH %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=VI --check-prefix=BOTH %s

; BOTH-LABEL: {{^}}main:
; BOTH: s_mov_b32 m0, s0
; VI-NEXT: s_nop 0
; BOTH-NEXT: s_sendmsg Gs_done(nop)
; BOTH-NEXT: s_endpgm

define void @main(i32 inreg %a) #0 {
main_body:
  call void @llvm.SI.sendmsg(i32 3, i32 %a)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.SI.sendmsg(i32, i32) #1

attributes #0 = { "ShaderType"="2" "unsafe-fp-math"="true" }
attributes #1 = { nounwind }
