; RUN: llc < %s -mtriple=mipsel -mcpu=mips1 -relocation-model=static | FileCheck %s -check-prefixes=ALL,DELAY
; RUN: llc < %s -mtriple=mipsel -mcpu=mips2 -relocation-model=static | FileCheck %s -check-prefixes=ALL,DELAY
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32 -relocation-model=static | FileCheck %s -check-prefixes=ALL,INTERLOCK

define i32 @xfer(float %a, float %b, i32 %sel) nounwind {
; ALL-LABEL: xfer:
entry:
  %x = bitcast float %a to i32
  %y = bitcast float %b to i32
  %cmp = icmp ne i32 %sel, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %add = add i32 %x, %y
  ret i32 %add

if.else:
  %sub = sub i32 %x, %y
  ret i32 %sub

; DELAY:          {{b(ne|eq)z}} ${{[0-9]+}},
; DELAY-NEXT:     nop
; INTERLOCK:      {{b(ne|eq)z}} ${{[0-9]+}},
; INTERLOCK-NEXT: mfc1
}
