; RUN: opt -S -mergefunc < %s | FileCheck %s

; Ensure that we do not merge functions that are identical with the
; exception of the order of the incoming blocks to a phi.

; CHECK-LABEL: define linkonce_odr hidden i1 @first(i2)
define linkonce_odr hidden i1 @first(i2) {
entry:
; CHECK: switch i2
  switch i2 %0, label %default [
    i2 0, label %L1
    i2 1, label %L2
    i2 -2, label %L3
  ]
default:
  unreachable
L1:
  br label %done
L2:
  br label %done
L3:
  br label %done
done:
  %result = phi i1 [ true, %L1 ], [ false, %L2 ], [ false, %L3 ]
; CHECK: ret i1
  ret i1 %result
}

; CHECK-LABEL: define linkonce_odr hidden i1 @second(i2)
define linkonce_odr hidden i1 @second(i2) {
entry:
; CHECK: switch i2
  switch i2 %0, label %default [
    i2 0, label %L1
    i2 1, label %L2
    i2 -2, label %L3
  ]
default:
  unreachable
L1:
  br label %done
L2:
  br label %done
L3:
  br label %done
done:
  %result = phi i1 [ true, %L3 ], [ false, %L2 ], [ false, %L1 ]
; CHECK: ret i1
  ret i1 %result
}
