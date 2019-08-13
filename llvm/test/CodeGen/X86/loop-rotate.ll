; RUN: llc -mtriple=i686-linux < %s | FileCheck %s

; Don't rotate the loop if the number of fall through to exit is not larger
; than the number of fall through to header.
define void @no_rotate() {
; CHECK-LABEL: no_rotate
; CHECK: %entry
; CHECK: %header
; CHECK: %middle
; CHECK: %latch1
; CHECK: %latch2
; CHECK: %end
entry:
  br label %header

header:
  %val1 = call i1 @foo()
  br i1 %val1, label %middle, label %end

middle:
  %val2 = call i1 @foo()
  br i1 %val2, label %latch1, label %end

latch1:
  %val3 = call i1 @foo()
  br i1 %val3, label %latch2, label %header

latch2:
  %val4 = call i1 @foo()
  br label %header

end:
  ret void
}

define void @do_rotate() {
; CHECK-LABEL: do_rotate
; CHECK: %entry
; CHECK: %then
; CHECK: %else
; CHECK: %latch1
; CHECK: %latch2
; CHECK: %header
; CHECK: %end
entry:
  %val0 = call i1 @foo()
  br i1 %val0, label %then, label %else

then:
  call void @a()
  br label %header

else:
  call void @b()
  br label %header

header:
  %val1 = call i1 @foo()
  br i1 %val1, label %latch1, label %end

latch1:
  %val3 = call i1 @foo()
  br i1 %val3, label %latch2, label %header

latch2:
  %val4 = call i1 @foo()
  br label %header

end:
  ret void
}

; The loop structure is same as in @no_rotate, but the loop header's predecessor
; doesn't fall through to it, so it should be rotated to get exit fall through.
define void @do_rotate2() {
; CHECK-LABEL: do_rotate2
; CHECK: %entry
; CHECK: %then
; CHECK: %middle
; CHECK: %latch1
; CHECK: %latch2
; CHECK: %header
; CHECK: %exit
entry:
  %val0 = call i1 @foo()
  br i1 %val0, label %then, label %header, !prof !1

then:
  call void @a()
  br label %end

header:
  %val1 = call i1 @foo()
  br i1 %val1, label %middle, label %exit

middle:
  %val2 = call i1 @foo()
  br i1 %val2, label %latch1, label %exit

latch1:
  %val3 = call i1 @foo()
  br i1 %val3, label %latch2, label %header

latch2:
  %val4 = call i1 @foo()
  br label %header

exit:
  call void @b()
  br label %end

end:
  ret void
}

declare i1 @foo()
declare void @a()
declare void @b()

!1 = !{!"branch_weights", i32 10, i32 1}
