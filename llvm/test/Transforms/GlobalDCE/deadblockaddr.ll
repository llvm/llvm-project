; RUN: opt -globaldce -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s

; Tests whether globaldce does the right cleanup while removing @bar
; so that a dead BlockAddress reference to foo won't prevent other passes
; to work properly, e.g. simplifycfg
@bar = internal unnamed_addr constant ptr blockaddress(@foo, %L1)

; CHECK-LABEL: foo
; CHECK-NOT: br label %L1
; CHECK: ret void
define void @foo() {
entry:
  br label %L1
L1:
  ret void
}
