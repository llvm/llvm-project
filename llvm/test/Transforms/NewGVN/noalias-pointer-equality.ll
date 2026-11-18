; RUN: opt < %s -passes=newgvn -S | FileCheck %s
; RUN: opt < %s -passes='default<O2>' -enable-newgvn -S | FileCheck %s

@g = global i32 0, align 4

; Reproducer from issue #234099, with a non-speculatable call so the O2
; pipeline cannot merge the two stores into one through %p. On the 'yes' edge
; %p (noalias) is known equal to @g; NewGVN must not replace the store through
; %p with a store through @g. That changes provenance and breaks noalias, which
; lets a later pass forward the initial store and miscompile.
define i32 @f(ptr noalias %p) noinline {
; CHECK-LABEL: define i32 @f(
; CHECK:         store i32 2, ptr %p
; CHECK-NOT:     store i32 2, ptr @g
; CHECK-NOT:     ret i32 1
; CHECK:         ret i32
entry:
  store i32 1, ptr %p, align 4
  %eq = icmp eq ptr %p, @g
  br i1 %eq, label %yes, label %join

yes:
  call void asm sideeffect "", ""()
  store i32 2, ptr %p, align 4
  br label %join

join:
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @main() {
; CHECK-LABEL: define i32 @main(
; CHECK:         call i32 @f(ptr {{.*}}@g)
  %r = call i32 @f(ptr @g)
  ret i32 %r
}
