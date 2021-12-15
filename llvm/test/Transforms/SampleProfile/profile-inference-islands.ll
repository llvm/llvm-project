; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-islands.prof -S -o %t
; RUN: FileCheck %s < %t -check-prefix=CHECK-ENTRY-COUNT
; RUN: opt < %t -analyze -block-freq -enable-new-pm=0 | FileCheck %s


; The test contains an isolated flow component ("island") that needs to be
; reconnected to the entry point via edges with a positive flow.
; The corresponding CFG is shown below:
;
; +--------+     +--------+     +----------+
; | b6 [1] | <-- | b4 [1] | <-- |  b1 [1]  |
; +--------+     +--------+     +----------+
;                  |              |
;                  |              |
;                  v              v
;                +--------+     +----------+
;                | b5 [0] |     | b2 [100] | <+
;                +--------+     +----------+  |
;                                 |           |
;                                 |           |
;                                 v           |
;                               +----------+  |
;                               | b3 [100] | -+
;                               +----------+
;                                 |
;                                 |
;                                 v
;                               +----------+
;                               |  b7 [0]  |
;                               +----------+


; Function Attrs: nounwind uwtable
define dso_local i32 @islands_1(i32 %0, i32 %1) #0 {
b1:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b2, label %b4
; CHECK: - b1: float = {{.*}}, int = {{.*}}, count = 2

b2:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 2, i32 0, i64 -1)
  br label %b3
; CHECK: - b2: float = {{.*}}, int = {{.*}}, count = 101

b3:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b2, label %b7
; CHECK: - b3: float = {{.*}}, int = {{.*}}, count = 101

b4:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 4, i32 0, i64 -1)
  br i1 %cmp, label %b5, label %b6
; CHECK: - b4: float = {{.*}}, int = {{.*}}, count = 1

b5:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 5, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b5: float = {{.*}}, int = {{.*}}, count = 0

b6:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 6, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b6: float = {{.*}}, int = {{.*}}, count = 1

b7:
  call void @llvm.pseudoprobe(i64 -5646793257986063976, i64 7, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b7: float = {{.*}}, int = {{.*}}, count = 1

}

; Another test with an island.
;
;  +----------+
;  |  b1 [0]  |
;  +----------+
;   |
;   |
;   v
;  +----------+
;  | b2 [100] | <+
;  +----------+  |
;   |            |
;   |            |
;   v            |
;  +----------+  |
;  | b3 [100] | -+
;  +----------+
;   |
;   |
;   v
;  +----------+
;  |  b4 [0]  |
;  +----------+

; Function Attrs: nounwind uwtable
define dso_local i32 @islands_2(i32 %0, i32 %1) #1 {
b1:
  call void @llvm.pseudoprobe(i64 -7683376842751444845, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br label %b2
; CHECK: - b1: float = {{.*}}, int = {{.*}}, count = 1

b2:
  call void @llvm.pseudoprobe(i64 -7683376842751444845, i64 2, i32 0, i64 -1)
  br label %b3
; CHECK: - b2: float = {{.*}}, int = {{.*}}, count = 10001

b3:
  call void @llvm.pseudoprobe(i64 -7683376842751444845, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b2, label %b4
; CHECK: - b3: float = {{.*}}, int = {{.*}}, count = 10001

b4:
  call void @llvm.pseudoprobe(i64 -7683376842751444845, i64 4, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b4: float = {{.*}}, int = {{.*}}, count = 1
}


; The test verifies that the island is connected to the entry block via a
; cheapest path (that is, passing through blocks with large counts).
;
; +---------+     +---------+     +----------+     +--------+
; | b8 [10] | <-- | b3 [10] | <-- | b1 [10]  | --> | b4 [0] |
; +---------+     +---------+     +----------+     +--------+
;                   |               |                |
;                   |               |                |
;                   |               v                |
;                   |             +----------+       |
;                   |             |  b2 [0]  |       |
;                   |             +----------+       |
;                   |               |                |
;                   |               |                |
;                   |               v                v
;                   |             +-------------------------+
;                   +-----------> |        b5 [100]         |
;                                 +-------------------------+
;                                   |           ^
;                                   |           |
;                                   v           |
;                                 +----------+  |
;                                 | b6 [100] | -+
;                                 +----------+
;                                   |
;                                   |
;                                   v
;                                 +----------+
;                                 |  b7 [0]  |
;                                 +----------+

; Function Attrs: nounwind uwtable
define dso_local i32 @islands_3(i32 %0, i32 %1) #1 {
b1:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  switch i32 %1, label %b2 [
    i32 1, label %b3
    i32 2, label %b4
  ]
; CHECK: - b1: float = {{.*}}, int = {{.*}}, count = 11

b2:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 2, i32 0, i64 -1)
  br label %b5
; CHECK: - b2: float = {{.*}}, int = {{.*}}, count = 0

b3:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b8, label %b5
; CHECK: - b3: float = {{.*}}, int = {{.*}}, count = 11

b4:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 4, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b4: float = {{.*}}, int = {{.*}}, count = 0

b5:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 5, i32 0, i64 -1)
  br label %b6
; CHECK: - b5: float = {{.*}}, int = {{.*}}, count = 1001

b6:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 6, i32 0, i64 -1)
  br i1 %cmp, label %b7, label %b5
; CHECK: - b6: float = {{.*}}, int = {{.*}}, count = 1001

b7:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 7, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b7: float = {{.*}}, int = {{.*}}, count = 1

b8:
  call void @llvm.pseudoprobe(i64 -9095645063288297061, i64 8, i32 0, i64 -1)
  ret i32 %1
; CHECK: - b8: float = {{.*}}, int = {{.*}}, count = 10
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #2

attributes #0 = { noinline nounwind uwtable "use-sample-profile"}
attributes #1 = { noinline nounwind uwtable "use-sample-profile"}
attributes #2 = { nounwind }

!llvm.pseudo_probe_desc = !{!7, !8}

!7 = !{i64 -5646793257986063976, i64 120879332589, !"islands_1"}
!8 = !{i64 -7683376842751444845, i64 69495280403, !"islands_2"}
!9 = !{i64 -9095645063288297061, i64 156608410269, !"islands_3"}

; CHECK-ENTRY-COUNT: = !{!"function_entry_count", i64 2}
