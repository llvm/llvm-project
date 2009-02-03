; RUN: llvm-as < %s | llc -march=x86-64 | grep min | count 1
; RUN: llvm-as < %s | llc -march=x86-64 | grep max | count 1
; RUN: llvm-as < %s | llc -march=x86-64 | grep mov | count 2

declare float @bar()

define float @foo(float %a)
{
  %s = call float @bar()
  %t = fcmp olt float %s, %a
  %u = select i1 %t, float %s, float %a
  ret float %u
}
define float @hem(float %a)
{
  %s = call float @bar()
  %t = fcmp uge float %s, %a
  %u = select i1 %t, float %s, float %a
  ret float %u
}
