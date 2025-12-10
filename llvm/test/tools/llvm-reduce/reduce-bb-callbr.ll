; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=CHECK,INTERESTING


; CHECK-LABEL: define i32 @keep_callbr(ptr %p, i1 %cond) {
; CHECK: entry1:
; CHECK-NEXT: callbr void asm
; INTERESTING: store i32 0,
; INTERESTING: store i32 1,

define i32 @keep_callbr(ptr %p, i1 %cond) {
entry1:
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"()
  to label %for.cond [label %preheader]

for.cond:
  store i32 0, ptr %p
  ret i32 0

preheader:
  store i32 1, ptr %p
  ret i32 1
}


; CHECK-LABEL: define i32 @drop_callbr(ptr %p, i1 %cond) {
; CHECK: entry1:
; CHECK-NEXT: br
; INTERESTING: store i32 0,

define i32 @drop_callbr(ptr %p, i1 %cond) {
entry1:
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"()
  to label %for.cond [label %preheader]

for.cond:
  store i32 0, ptr %p
  ret i32 0

preheader:
  store i32 1, ptr %p
  ret i32 1

}
