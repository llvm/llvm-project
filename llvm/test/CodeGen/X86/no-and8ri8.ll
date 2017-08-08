; RUN: llc -mtriple=x86_64-pc-linux -mattr=+avx512f --show-mc-encoding < %s | FileCheck %s

declare i1 @bar() 

; CHECK-LABEL: @foo
; CHECK-NOT: andb {{.*}} # encoding: [0x82,
define i1 @foo(i1 %i) nounwind {
entry:
  br i1 %i, label %if, label %else

if:
  %r = call i1 @bar()
  br label %else

else:
  %ret = phi i1 [%r, %if], [true, %entry]
  ret i1 %ret
}
