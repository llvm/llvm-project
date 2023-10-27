; RUN: not llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate output register for constraint 'v'

define signext i32 @int_and_v(i32 signext %cc_dep1) {
entry:
  %0 = tail call i32 asm sideeffect "", "=v,0"(i32 %cc_dep1)
  ret i32 %0
}
