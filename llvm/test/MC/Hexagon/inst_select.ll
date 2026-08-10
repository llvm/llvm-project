; RUN: llc -mtriple=hexagon -filetype=obj %s -o - \
; RUN: | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

define i32 @foo (i1 %a, i32 %b, i32 %c)
{
  %1 = select i1 %a, i32 %b, i32 %c
  ret i32 %1
}

; CHECK: 00 c0 00 85 8500c000 {  p0 = tstbit(r0,#0) } 
; CHECK: 00 42 01 f4 f4014200 {  r0 = mux(p0,r1,r2)
; CHECK: 00 c0 9f 52 529fc000    jumpr r31 } 
