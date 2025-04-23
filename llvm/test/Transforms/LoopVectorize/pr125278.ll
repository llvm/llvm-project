; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -force-vector-width=4 -disable-output %s

define void @pr125278(ptr %dst, i64 %n) {
entry:
  %true.ext = zext i1 true to i32
  br label %cond

cond:
  br label %loop

loop:
  %iv = phi i64 [ 0, %cond ], [ %iv.next, %loop ]
  %false.ext = zext i1 false to i32
  %xor = xor i32 %false.ext, %true.ext
  %xor.trunc = trunc i32 %xor to i8
  store i8 %xor.trunc, ptr %dst, align 1
  %iv.next = add i64 %iv, 1
  %cmp = icmp ult i64 %iv.next, %n
  br i1 %cmp, label %loop, label %cond
}
