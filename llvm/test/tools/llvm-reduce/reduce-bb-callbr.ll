; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg %s --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=INTERESTING

; CHECK-INTERESTINGNESS: store i32 0,
; CHECK-INTERESTINGNESS: store i32 1,

define i32 @e(ptr %p, i1 %cond) {
entry:
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"()
  to label %for.cond [label %preheader]

for.cond:
  store i32 0, ptr %p
  br label %preheader

preheader:
  store i32 1, %p
  br i1 %%cond, label %for.cond, label %g

returnbb:
  ret i32 0
}
