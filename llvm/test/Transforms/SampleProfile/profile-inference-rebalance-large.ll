; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-rebalance-large.prof | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-rebalance-large.prof | opt -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK2

; The test verifies that counts can rebalanced in switch statements that contain
; both 'known' and 'unknown' basic blocks.
;
;                      +---------+
;   +----------------- | b15 [?] |
;   |                  +---------+
;   |                    ^
;   |                    |
;   |                    |
;   |  +---------+     +--------------+     +---------+
;   |  | b13 [?] | <-- |  b11 [3300]  | --> | b14 [?] |
;   |  +---------+     +--------------+     +---------+
;   |    |               |          |         |
;   |    |               |          |         |
;   |    |               v          |         |
;   |    |             +---------+  |         |
;   |    |             | b12 [0] |  |         |
;   |    |             +---------+  |         |
;   |    |               |          |         |
;   |    |               |          |         |
;   |    |               v          v         |
;   |    |             +--------------+       |
;   |    +-----------> |              | <-----+
;   |                  |  b16 [3300]  |
;   +----------------> |              |
;                      +--------------+

@yydebug = dso_local global i32 0, align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @foo1(i32 %0, i32 %1) #0 {
b11:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  switch i32 %1, label %b12 [
    i32 1, label %b13
    i32 2, label %b14
    i32 3, label %b15
    i32 4, label %b16
  ]
; CHECK:  edge %b11 -> %b12 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK:  edge %b11 -> %b13 probability is 0x20000000 / 0x80000000 = 25.00%
; CHECK:  edge %b11 -> %b14 probability is 0x20000000 / 0x80000000 = 25.00%
; CHECK:  edge %b11 -> %b15 probability is 0x20000000 / 0x80000000 = 25.00%
; CHECK:  edge %b11 -> %b16 probability is 0x20000000 / 0x80000000 = 25.00%
; CHECK2: - b11: float = {{.*}}, int = {{.*}}, count = 3300

b12:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 2, i32 0, i64 -1)
  br label %b16
; CHECK2: - b12: float = {{.*}}, int = {{.*}}, count = 0

b13:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 3, i32 0, i64 -1)
  br label %b16
; CHECK2: - b13: float = {{.*}}, int = {{.*}}, count = 825

b14:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 4, i32 0, i64 -1)
  br label %b16
; CHECK2: - b14: float = {{.*}}, int = {{.*}}, count = 825

b15:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 5, i32 0, i64 -1)
  br label %b16
; CHECK2: - b15: float = {{.*}}, int = {{.*}}, count = 825

b16:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 6, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b16: float = {{.*}}, int = {{.*}}, count = 3300
}


; The test verifies that counts can rebalanced even when control-flow ends at
; a basic block with an unknown count.
;
;                 +-----------+
;                 | b21 [128] | -+
;                 +-----------+  |
;                   |            |
;                   v            |
;                 +-----------+  |
;                 | b22 [128] |  |
;                 +-----------+  |
;                   |            |
;                   v            |
;                 +-----------+  |
;   +------------ | b23 [128] | <+
;   |             +-----------+
;   |               |
;   v               v
; +---------+     +-----------+
; | b26 [?] | <-- | b24 [128] |
; +---------+     +-----------+
;   |               |
;   |               v
;   |             +-----------+
;   |             |  b25 [?]  |
;   |             +-----------+
;   |               |
;   |               v
;   |             +-----------+
;   +-----------> |  b27 [?]  | -+
;                 +-----------+  |
;                   |            |
;                   v            |
;                 +-----------+  |
;                 |  b28 [?]  |  |
;                 +-----------+  |
;                   |            |
;                   v            |
;                 +-----------+  |
;                 |  b29 [?]  | <+
;                 +-----------+

define dso_local i32 @foo2(i32 %0, i32 %1) #0 {
b21:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b22, label %b23
; CHECK:  edge %b21 -> %b22 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK:  edge %b21 -> %b23 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK2: - b21: float = {{.*}}, int = {{.*}}, count = 128

b22:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 2, i32 0, i64 -1)
  br label %b23

b23:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b24, label %b26
; CHECK:  edge %b23 -> %b24 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK:  edge %b23 -> %b26 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK2: - b23: float = {{.*}}, int = {{.*}}, count = 128

b24:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 4, i32 0, i64 -1)
  br i1 %cmp, label %b25, label %b26
; CHECK:  edge %b24 -> %b25 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge %b24 -> %b26 probability is 0x40000000 / 0x80000000 = 50.00%

b25:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 5, i32 0, i64 -1)
  br label %b27
; CHECK2: - b25: float = {{.*}}, int = {{.*}}, count = 64

b26:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 6, i32 0, i64 -1)
  br label %b27
; CHECK2: - b26: float = {{.*}}, int = {{.*}}, count = 64

b27:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 7, i32 0, i64 -1)
  br i1 %cmp, label %b28, label %b29
; CHECK:  edge %b27 -> %b28 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge %b27 -> %b29 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b27: float = {{.*}}, int = {{.*}}, count = 128

b28:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 8, i32 0, i64 -1)
  br label %b29
; CHECK2: - b28: float = {{.*}}, int = {{.*}}, count = 64

b29:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 9, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b29: float = {{.*}}, int = {{.*}}, count = 128
}


; The test verifies a flexible mode of rebalancing in which some jumps to known
; basic blocks are ignored.
;
;                 +------------+
;                 | b31 [500]  |
;                 +------------+
;                   |
;                   v
; +---------+     +------------+
; | b33 [?] | <-- | b32 [1500] | <-----+
; +---------+     +------------+       |
;   |               |                  |
;   |               v                  |
;   |             +------------+     +-----------+
;   |             | b34 [1200] | --> | b36 [900] |
;   |             +------------+     +-----------+
;   |               |
;   |               v
;   |             +------------+
;   |             |  b35 [?]   |
;   |             +------------+
;   |               |
;   |               v
;   |             +------------+
;   +-----------> |  b37 [?]   | -+
;                 +------------+  |
;                   |             |
;                   v             |
;                 +------------+  |
;                 |  b38 [?]   |  |
;                 +------------+  |
;                   |             |
;                   v             |
;                 +------------+  |
;                 | b39 [500]  | <+
;                 +------------+
;

define dso_local i32 @foo3(i32 %0, i32 %1) #0 {
b31:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br label %b32
; CHECK2: - b31: float = {{.*}}, int = {{.*}}, count = 500

b32:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 2, i32 0, i64 -1)
  br i1 %cmp, label %b33, label %b34
; CHECK:  edge %b32 -> %b33 probability is 0x1999999a / 0x80000000 = 20.00%
; CHECK:  edge %b32 -> %b34 probability is 0x66666666 / 0x80000000 = 80.00%
; CHECK2: - b32: float = {{.*}}, int = {{.*}}, count = 1500

b33:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 3, i32 0, i64 -1)
  br label %b37
; CHECK2: - b33: float = {{.*}}, int = {{.*}}, count = 300

b34:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 4, i32 0, i64 -1)
  br i1 %cmp, label %b35, label %b36
; CHECK:  edge %b34 -> %b35 probability is 0x15555555 / 0x80000000 = 16.67%
; CHECK:  edge %b34 -> %b36 probability is 0x6aaaaaab / 0x80000000 = 83.33% [HOT edge]
; CHECK2: - b34: float = {{.*}}, int = {{.*}}, count = 1200

b35:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 5, i32 0, i64 -1)
  br label %b37
; CHECK2: - b35: float = {{.*}}, int = {{.*}}, count = 200

b36:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 6, i32 0, i64 -1)
  br label %b32
; CHECK2: - b36: float = {{.*}}, int = {{.*}}, count = 1000

b37:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 7, i32 0, i64 -1)
  br i1 %cmp, label %b38, label %b39
; CHECK:  edge %b37 -> %b38 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge %b37 -> %b39 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b37: float = {{.*}}, int = {{.*}}, count = 500

b38:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 8, i32 0, i64 -1)
  br label %b39
; CHECK2: - b38: float = {{.*}}, int = {{.*}}, count = 250

b39:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 9, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b39: float = {{.*}}, int = {{.*}}, count = 500
}


; The test verifies that flow rebalancer can ignore 'unlikely' jumps.
;
;                            +-----------+
;                            | b41 [400] | -+
;                            +-----------+  |
;                              |            |
;                              |            |
;                              v            |
;                            +-----------+  |
;                            |  b42 [?]  |  |
;                            +-----------+  |
;                              |            |
;                              |            |
;                              v            v
; +---------++---------+     +---------------------------+     +---------++---------+
; | b48 [?] || b46 [?] | <-- |                           | --> | b47 [?] || b49 [?] |
; +---------++---------+     |                           |     +---------++---------+
;   |  ^       |             |                           |       |          ^
;   |  |       |             |         b43 [400]         |       |          |
;   |  +-------+-------------|                           |       |          |
;   |          |             |                           |       |          |
;   |          |             |                           | ------+----------+
;   |          |             +---------------------------+       |
;   |          |               |                 |               |
;   |          |               |                 |               |
;   |          |               v                 v               |
;   |          |             +-----------+     +---------+       |
;   |          |             |  b44 [?]  |     | b45 [?] |       |
;   |          |             +-----------+     +---------+       |
;   |          |               |                 |               |
;   |          |               |                 |               |
;   |          |               v                 v               |
;   |          |             +---------------------------+       |
;   |          +-----------> |                           | <-----+
;   |                        |        b410 [400]         |
;   |                        |                           |
;   +----------------------> |                           |
;                            +---------------------------+


define dso_local void @foo4(i32 %0, i32 %1) #0 {
b41:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b42, label %b43
; CHECK:  edge %b41 -> %b42 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge %b41 -> %b43 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b41: float = {{.*}}, int = {{.*}}, count = 400

b42:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 2, i32 0, i64 -1)
  br label %b43
; CHECK2: - b42: float = {{.*}}, int = {{.*}}, count = 200

b43:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 3, i32 0, i64 -1)
  switch i32 %1, label %b49 [
    i32 1, label %b44
    i32 2, label %b45
    i32 3, label %b46
    i32 4, label %b47
    i32 5, label %b48
  ]
; CHECK:  edge %b43 -> %b49 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK:  edge %b43 -> %b44 probability is 0x1999999a / 0x80000000 = 20.00%
; CHECK:  edge %b43 -> %b45 probability is 0x1999999a / 0x80000000 = 20.00%
; CHECK:  edge %b43 -> %b46 probability is 0x1999999a / 0x80000000 = 20.00%
; CHECK:  edge %b43 -> %b47 probability is 0x1999999a / 0x80000000 = 20.00%
; CHECK:  edge %b43 -> %b48 probability is 0x1999999a / 0x80000000 = 20.00%
; CHECK2: - b43: float = {{.*}}, int = {{.*}}, count = 400

b44:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 4, i32 0, i64 -1)
  br label %b410
; CHECK2: - b44: float = {{.*}}, int = {{.*}}, count = 80

b45:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 5, i32 0, i64 -1)
  br label %b410
; CHECK2: - b45: float = {{.*}}, int = {{.*}}, count = 80

b46:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 6, i32 0, i64 -1)
  br label %b410
; CHECK2: - b46: float = {{.*}}, int = {{.*}}, count = 80

b47:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 7, i32 0, i64 -1)
  br label %b410
; CHECK2: - b47: float = {{.*}}, int = {{.*}}, count = 80

b48:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 8, i32 0, i64 -1)
  br label %b410
; CHECK2: - b48: float = {{.*}}, int = {{.*}}, count = 80

b49:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 9, i32 0, i64 -1)
  unreachable
; CHECK2: - b49: float = {{.*}}, int = {{.*}}, count = 0

b410:
  call void @llvm.pseudoprobe(i64 -6882312132165544686, i64 10, i32 0, i64 -1)
  ret void
; CHECK2: - b410: float = {{.*}}, int = {{.*}}, count = 400
}


; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

attributes #0 = { noinline nounwind uwtable "use-sample-profile" }
attributes #4 = { inaccessiblememonly nounwind willreturn }

!llvm.pseudo_probe_desc = !{!7, !8, !9, !10}

!7 = !{i64 7682762345278052905, i64 157181141624, !"foo1", null}
!8 = !{i64 2494702099028631698, i64 208782362068, !"foo2", null}
!9 = !{i64 -7908226060800700466, i64 189901498683, !"foo3", null}
!10 = !{i64 -6882312132165544686, i64 241030178952, !"foo4", null}
