; RUN: llc < %s -mtriple=x86_64-apple-darwin -no-integrated-as | FileCheck %s
; PR 7528
; formerly crashed

%0 = type { [12 x i16] }
%union..0anon = type { [3 x <1 x i64>] }

@gsm_H.1466 = internal constant %0 { [12 x i16] [i16 -134, i16 -374, i16 0, i16 2054, i16 5741, i16 8192, i16 5741, i16 2054, i16 0, i16 -374, i16 -134, i16 0] }, align 8 ; <ptr> [#uses=1]

define void @weighting_filter() nounwind ssp {
entry:
; CHECK: leaq _gsm_H.1466(%rip),%rax;
  call void asm sideeffect "leaq $0,%rax;\0A", "*X,~{dirflag},~{fpsr},~{flags},~{memory},~{rax}"(ptr elementtype(%union..0anon) @gsm_H.1466) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}
