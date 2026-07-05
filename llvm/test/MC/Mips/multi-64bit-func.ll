; There is no real check here. If the test doesn't 
; assert it passes.
; RUN: llc -mtriple=mips64el -filetype=obj -mcpu=mips64r2 -disable-mips-delay-filler < %s 
; Run it again without extra nop in delay slot
; RUN: llc -mtriple=mips64el -filetype=obj -mcpu=mips64r2 < %s 

define i32 @bosco1(i32 %x) nounwind readnone {
entry:
  %inc = add i32 %x, 1
  ret i32 %inc
}

define i32 @bosco2(i32 %x) nounwind readnone {
entry:
  %inc = add i32 %x, 1
  ret i32 %inc
}

define i32 @bosco3(i32 %x) nounwind readnone {
entry:
  %inc = add i32 %x, 1
  ret i32 %inc
}
