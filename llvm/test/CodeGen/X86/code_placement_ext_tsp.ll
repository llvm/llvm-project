;; See also llvm/unittests/Transforms/Utils/CodeLayoutTest.cpp
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -verify-machineinstrs -enable-ext-tsp-block-placement < %s | FileCheck %s

define void @func1a()  {
; Test that the algorithm positions the most likely successor first
;
; +-----+
; | b0  | -+
; +-----+  |
;   |      |
;   | 40   |
;   v      |
; +-----+  |
; | b1  |  | 100
; +-----+  |
;   |      |
;   | 40   |
;   v      |
; +-----+  |
; | b2  | <+
; +-----+
;
; CHECK-LABEL: func1a:
; CHECK: b0
; CHECK: b2
; CHECK: b1

b0:
  %call = call zeroext i1 @a()
  br i1 %call, label %b1, label %b2, !prof !1

b1:
  call void @d()
  call void @d()
  call void @d()
  br label %b2

b2:
  call void @e()
  ret void
}


define void @func1b()  {
; Test that the algorithm prefers many fallthroughs even in the presence of
; a heavy successor
;
; +-----+
; | b0  | -+
; +-----+  |
;   |      |
;   | 80   |
;   v      |
; +-----+  |
; | b1  |  | 100
; +-----+  |
;   |      |
;   | 80   |
;   v      |
; +-----+  |
; | b2  | <+
; +-----+
;
; CHECK-LABEL: func1b:
; CHECK: b0
; CHECK: b1
; CHECK: b2

b0:
  %call = call zeroext i1 @a()
  br i1 %call, label %b1, label %b2, !prof !2

b1:
  call void @d()
  call void @d()
  call void @d()
  br label %b2

b2:
  call void @e()
  ret void
}


define void @func2() !prof !3 {
; Test that the algorithm positions the hot chain continuously
;
; +----+  [7]   +-------+
; | b1 | <----- |  b0   |
; +----+        +-------+
;   |             |
;   |             | [15]
;   |             v
;   |           +-------+
;   |           |  b3   |
;   |           +-------+
;   |             |
;   |             | [15]
;   |             v
;   |           +-------+   [31]
;   |           |       | -------+
;   |           |  b4   |        |
;   |           |       | <------+
;   |           +-------+
;   |             |
;   |             | [15]
;   |             v
;   |    [7]    +-------+
;   +---------> |  b2   |
;               +-------+
;
; CHECK-LABEL: func2:
; CHECK: b0
; CHECK: b3
; CHECK: b4
; CHECK: b2
; CHECK: b1

b0:
  call void @d()
  call void @d()
  call void @d()
  %call = call zeroext i1 @a()
  br i1 %call, label %b1, label %b3, !prof !4

b1:
  call void @d()
  br label %b2

b2:
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  ret void

b3:
  call void @d()
  br label %b4

b4:
  call void @d()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %b2, label %b4, !prof !5
}


define void @func3() !prof !6 {
; A larger test where it is beneficial for locality to break the loop
;
;                 +--------+
;                 |   b0   |
;                 +--------+
;                   |
;                   | [177]
;                   v
; +----+  [177]   +---------------------------+
; | b5 | <------- |            b1             |
; +----+          +---------------------------+
;                   |         ^         ^
;                   | [196]   | [124]   | [70]
;                   v         |         |
; +----+  [70]    +--------+  |         |
; | b4 | <------- |   b2   |  |         |
; +----+          +--------+  |         |
;   |               |         |         |
;   |               | [124]   |         |
;   |               v         |         |
;   |             +--------+  |         |
;   |             |   b3   | -+         |
;   |             +--------+            |
;   |                                   |
;   +-----------------------------------+
;
; CHECK-LABEL: func3:
; CHECK: b0
; CHECK: b1
; CHECK: b2
; CHECK: b3
; CHECK: b5
; CHECK: b4

b0:
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  br label %b1

b1:
  %call = call zeroext i1 @a()
  br i1 %call, label %b5, label %b2, !prof !7

b2:
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %b3, label %b4, !prof !8

b3:
  call void @d()
  call void @f()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  br label %b1

b4:
  call void @d()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  call void @e()
  br label %b1

b5:
  ret void
}

define void @func_loop() !prof !9 {
; Test that the algorithm can rotate loops in the presence of profile data.
;
;                  +--------+
;                  | entry  |
;                  +--------+
;                    |
;                    | 1
;                    v
; +--------+  16   +--------+
; | if.then| <---- | header | <+
; +--------+       +--------+  |
;   |                |         |
;   |                | 16      |
;   |                v         |
;   |              +--------+  |
;   |              | if.else|  | 31
;   |              +--------+  |
;   |                |         |
;   |                | 16      |
;   |                v         |
;   |        16    +--------+  |
;   +------------> | if.end | -+
;                  +--------+
;                    |
;                    | 1
;                    v
;                  +--------+
;                  |  end   |
;                  +--------+
;
; CHECK-LABEL: func_loop:
; CHECK: if.else
; CHECK: if.end
; CHECK: header
; CHECK: if.then

entry:
  br label %header

header:
  call void @e()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !10

if.then:
  call void @f()
  br label %if.end

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header, label %end

end:
  ret void
}

define void @func4() !prof !11 {
; Test verifying that chains can be split in order to improve the objective
; by creating more fallthroughs
;
; +-------+
; | entry |--------+
; +-------+        |
;   |              |
;   | 27           |
;   v              |
; +-------+        |
; |  b1   | -+     |
; +-------+  |     |
;   |        |     |
;   | 10     |     | 0
;   v        |     |
; +-------+  |     |
; |  b3   |  | 17  |
; +-------+  |     |
;   |        |     |
;   | 10     |     |
;   v        |     |
; +-------+  |     |
; |  b2   | <+ ----+
; +-------+
;
; CHECK-LABEL: func4:
; CHECK: entry
; CHECK: b1
; CHECK: b3
; CHECK: b2

entry:
  call void @b()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %b1, label %b2, !prof !12

b1:
  call void @c()
  %call = call zeroext i1 @a()
  br i1 %call, label %b2, label %b3, !prof !13

b2:
  call void @d()
  ret void

b3:
  call void @e()
  br label %b2
}

declare zeroext i1 @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @g()
declare void @f()
declare void @h()

!1 = !{!"branch_weights", i32 40, i32 100}
!2 = !{!"branch_weights", i32 80, i32 100}
!3 = !{!"function_entry_count", i64 2200}
!4 = !{!"branch_weights", i32 700, i32 1500}
!5 = !{!"branch_weights", i32 1500, i32 3100}
!6 = !{!"function_entry_count", i64 177}
!7 = !{!"branch_weights", i32 177, i32 196}
!8 = !{!"branch_weights", i32 125, i32 70}
!9 = !{!"function_entry_count", i64 1}
!10 = !{!"branch_weights", i32 16, i32 16}
!11 = !{!"function_entry_count", i64 1}
!12 = !{!"branch_weights", i32 27, i32 0}
!13 = !{!"branch_weights", i32 17, i32 10}
