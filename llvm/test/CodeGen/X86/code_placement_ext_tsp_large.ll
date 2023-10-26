; REQUIRES: asserts
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -enable-ext-tsp-block-placement=1 -ext-tsp-chain-split-threshold=128 -debug-only=block-placement < %s 2>&1 | FileCheck %s
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -enable-ext-tsp-block-placement=1 -ext-tsp-chain-split-threshold=1 -debug-only=block-placement < %s 2>&1 | FileCheck %s -check-prefix=CHECK2
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -enable-ext-tsp-block-placement=0 -debug-only=block-placement < %s 2>&1 | FileCheck %s -check-prefix=CHECK3

@yydebug = dso_local global i32 0, align 4

define void @func_large() !prof !0 {
; A large CFG instance where chain splitting helps to
; compute a better basic block ordering. The test verifies that with chain
; splitting, the resulting layout is improved (e.g., the score is increased).
;
;                                     +----------------+
;                                     | b0 [76 bytes]  | -------------------+
;                                     +----------------+                    |
;                                       |                                   |
;                                       | 3,065,981,778                     |
;                                       v                                   |
; +----------------+  766,495,444     +----------------+                    |
; | b8 [244 bytes] | <--------------- |  b2 [4 bytes]  |                    |
; +----------------+                  +----------------+                    |
;   |        ^                          |                                   |
;   |        |                          | 2,299,486,333                     |
;   |        | 766,495,444              v                                   |
;   |        |                        +----------------+                    |
;   |        +----------------------- | b3 [12 bytes]  |                    |
;   |                                 +----------------+                    |
;   |                                   |                                   |
;   |                                   | 1,532,990,888                     |
;   |                                   v                                   |
;   |                                 +----------------+                    | 574,869,946
;   |                 +-------------- | b4 [12 bytes]  |                    |
;   |                 |               +----------------+                    |
;   |                 |                 |                                   |
;   |                 |                 | 574,871,583                       |
;   |                 |                 v                                   |
;   |                 |               +----------------+                    |
;   |                 |               | b5 [116 bytes] | -+                 |
;   |                 |               +----------------+  |                 |
;   |                 |                 |                 |                 |
;   |                 |                 | 1,636           |                 |
;   |                 |                 v                 |                 |
;   |                 |               +----------------+  |                 |
;   |                 |       +------ | b6 [32 bytes]  |  |                 |
;   |                 |       |       +----------------+  |                 |
;   |                 |       |         |                 |                 |
;   |                 |       |         | 7               | 3,065,981,778   |
;   |                 |       |         v                 |                 |
;   |                 |       |       +----------------+  |                 |
;   |                 |       | 1,628 | b9 [16 bytes]  |  |                 |
;   |                 |       |       +----------------+  |                 |
;   |                 |       |         |                 |                 |
;   |                 |       |         | 7               |                 |
;   |                 |       |         v                 |                 |
;   |                 |       |       +----------------+  |                 |
;   |                 |       +-----> | b7 [12 bytes]  |  |                 |
;   |                 |               +----------------+  |                 |
;   |                 |                 |                 |                 |
;   |                 | 958,119,305     | 1,636           |                 |
;   |                 |                 v                 v                 v
;   |                 |               +------------------------------------------+
;   |                 +-------------> |                                          |
;   |       1,532,990,889             |                  b1 [36 bytes]           |
;   +-------------------------------> |                                          |
;                                     +------------------------------------------+
;
; An expected output with a large chain-split-threshold -- the layout score is
; increased by ~17%
;
; CHECK-LABEL: Applying ext-tsp layout
; CHECK:   original  layout score: 23587612604815436.00
; CHECK:   optimized layout score: 27891096739311172.00
; CHECK: b0
; CHECK: b2
; CHECK: b3
; CHECK: b4
; CHECK: b5
; CHECK: b8
; CHECK: b1
; CHECK: b6
; CHECK: b7
; CHECK: b9
;
; An expected output with chain-split-threshold=1 (disabling split point enumeration)
;
; CHECK2-LABEL: Applying ext-tsp layout
; CHECK2:   original  layout score: 23587612604815436.00
; CHECK2:   optimized layout score: 27891096739311172.00
; CHECK2: b0
; CHECK2: b2
; CHECK2: b3
; CHECK2: b4
; CHECK2: b5
; CHECK2: b8
; CHECK2: b1
; CHECK2: b6
; CHECK2: b7
; CHECK2: b9
;
; An expected output with ext-tsp disabled -- the layout is not modified:
;
; CHECK3-LABEL: func_large:
; CHECK3: b0
; CHECK3: b1
; CHECK3: b2
; CHECK3: b3
; CHECK3: b4
; CHECK3: b5
; CHECK3: b6
; CHECK3: b7
; CHECK3: b8
; CHECK3: b9

b0:
  %0 = load i32, i32* @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br i1 %cmp, label %b1, label %b2, !prof !1
b1:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  ret void
b2:
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  br i1 %cmp, label %b3, label %b8, !prof !2
b3:
  call void @d()
  call void @d()
  br i1 %cmp, label %b4, label %b8, !prof !3
b4:
  call void @e()
  call void @e()
  br i1 %cmp, label %b5, label %b1, !prof !4
b5:
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
  br i1 %cmp, label %b1, label %b6, !prof !5
b6:
  call void @g()
  call void @g()
  call void @g()
  call void @g()
  call void @g()
  call void @g()
  call void @g()
  br i1 %cmp, label %b7, label %b9, !prof !6
b7:
  call void @h()
  call void @h()
  br label %b1
b8:
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  call void @i()
  br label %b1
b9:
  call void @j()
  call void @j()
  call void @j()
  br label %b7
}


declare void @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @f()
declare void @g()
declare void @h()
declare void @i()
declare void @j()

!0 = !{!"function_entry_count", i64 6131963556}
!1 = !{!"branch_weights", i32 3065981778, i32 3065981778}
!2 = !{!"branch_weights", i32 2299486333, i32 766495444}
!3 = !{!"branch_weights", i32 1532990888, i32 766495444}
!4 = !{!"branch_weights", i32 574871583, i32 958119305}
!5 = !{!"branch_weights", i32 574869946, i32 1636}
!6 = !{!"branch_weights", i32 1628, i32 7}
