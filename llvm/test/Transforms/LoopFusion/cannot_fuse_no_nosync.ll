; RUN: opt -S -passes=loop-simplify,loop-fusion \
; RUN:   -pass-remarks-analysis=loop-fusion  < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts

; CHECK: [foo]: Loop is not a candidate for fusion: Loop contains a potentially synchronizing call

@sync = global i32 0, align 4
@value = global i32 0, align 4

; Initially, both globals contain zero. A second thread can perform these
; operations, in this order:
;
;   store i32 1, ptr @value
;   store atomic i32 1, ptr @sync release
;   store i32 2, ptr @value
;   store atomic i32 2, ptr @sync release
;
; Without fusion, both acquire_sync calls execute before either load of
; @value. Therefore (r1, r2, x, y) = (0, 1, 0, 1) is impossible: once r2
; observes the release of sync == 1, the later load of value cannot observe
; its initial value 0.


; If we fuse, however, the following becomes possible.
; Initially, @sync == 0 and @value == 0.
;
; Iteration 1:
;   %r1 = call i32 @acquire_sync() ; returns 0
;   %x  = load i32, ptr @value     ; returns 0
;
; Other thread:
;   store i32 1, ptr @value
;   store atomic i32 1, ptr @sync release
;
; Iteration 2:
;   %r2 = call i32 @acquire_sync() ; returns 1
;   %y  = load i32, ptr @value     ; returns 1
;
; Result:
;   (r1, r2, x, y) = (0, 1, 0, 1)

define i32 @acquire_sync() nounwind willreturn memory(read) {
entry:
  %s = load atomic i32, ptr @sync acquire, align 4
  ret i32 %s
}

define i32 @foo() {
entry:
  br label %loop1

loop1:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop1 ]
  %r1.phi = phi i32 [ poison, %entry ], [ %r1, %loop1 ]
  %r2.phi = phi i32 [ poison, %entry ], [ %r2, %loop1 ]
  %s = call i32 @acquire_sync()
  %is1 = icmp eq i32 %i, 1
  %r1 = select i1 %is1, i32 %r1.phi, i32 %s
  %r2 = select i1 %is1, i32 %s, i32 %r2.phi
  %inext = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inext, 2
  br i1 %done, label %loop1.end, label %loop1

loop1.end:
  br label %loop2

loop2:
  %j = phi i32 [ 0, %loop1.end ], [ %jnext, %loop2 ]
  %x.phi = phi i32 [ poison, %loop1.end ], [ %x, %loop2 ]
  %y.phi = phi i32 [ poison, %loop1.end ], [ %y, %loop2 ]
  %v = load i32, ptr @value, align 4
  %js1 = icmp eq i32 %j, 1
  %x = select i1 %js1, i32 %x.phi, i32 %v
  %y = select i1 %js1, i32 %v, i32 %y.phi
  %jnext = add nuw nsw i32 %j, 1
  %done2 = icmp eq i32 %jnext, 2
  br i1 %done2, label %exit, label %loop2

exit:
  %a = add i32 %r1, %r2
  %b = add i32 %x, %y
  %result = add i32 %a, %b
  ret i32 %result
}
