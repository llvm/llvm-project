; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; CHECK-NOT:     {{(min|max|mov)}}
; CHECK:     mov
; CHECK-NOT:     {{(min|max|mov)}}
; CHECK:     min
; CHECK-NOT:     {{(min|max|mov)}}
; CHECK:     mov
; CHECK-NOT:     {{(min|max|mov)}}
; CHECK:     max
; CHECK-NOT:     {{(min|max|mov)}}

declare float @bar()

define float @foo(float %a) nounwind
{
  %s = call float @bar()
  %t = fcmp olt float %s, %a
  %u = select i1 %t, float %s, float %a
  ret float %u
}
define float @hem(float %a) nounwind
{
  %s = call float @bar()
  %t = fcmp ogt float %s, %a
  %u = select i1 %t, float %s, float %a
  ret float %u
}
