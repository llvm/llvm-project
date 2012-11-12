; RUN: llc < %s -march=x86-64

define x86_fp80 @foo(x86_fp80 %a) {
  %1 = tail call x86_fp80 @copysignl(x86_fp80 0xK7FFF8000000000000000, x86_fp80 %a) nounwind readnone
  ret x86_fp80 %1
}

declare x86_fp80 @copysignl(x86_fp80, x86_fp80) nounwind readnone
