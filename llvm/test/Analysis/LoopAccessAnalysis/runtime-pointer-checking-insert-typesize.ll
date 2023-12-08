; RUN: opt -passes='print<access-info>' -disable-output < %s

; This regression test is defending against a use of the wrong interface
; of TypeSize (implicit conversion to a scalar). This issue cropped up
; in RuntimePointerChecking::insert when performing loop load elimination
; because this function was previously unaware of scalable types.

define void @runtime_pointer_checking_insert_typesize(ptr %a,
                                                      ptr %b) {
entry:
  br label %loop.body
loop.body:
  %0 = phi i64 [ 0, %entry ], [%1, %loop.body]
  %idx_a = getelementptr <vscale x 4 x i32>, ptr %a, i64 %0
  %idx_b = getelementptr <vscale x 4 x i32>, ptr %b, i64 %0
  %tmp = load <vscale x 4 x i32>, ptr %idx_a
  store <vscale x 4 x i32> %tmp, ptr %idx_b
  %1 = add i64 %0, 2
  %2 = icmp eq i64 %1, 1024
  br i1 %2, label %loop.end, label %loop.body
loop.end:
  ret void
}
