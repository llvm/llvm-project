; RUN: llvm-as < %s | llc -march=x86

define i1 @foo(i64 %a)
{
  %t = bitcast i64 %a to <2 x float>
  %r = extractelement <2 x float> %t, i32 0
  %s = extractelement <2 x float> %t, i32 1
  %b = fcmp uno float %r, %s
  ret i1 %b
}
