; RUN: not llc -mtriple=s390x-linux-gnu -mcpu=z15 -mattr=soft-float < %s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate output register for constraint 'f'

define signext i32 @int_and_f(i32 signext %cc_dep1) {
entry:
  %0 = tail call i32 asm sideeffect "", "=f,0"(i32 %cc_dep1)
  ret i32 %0
}
