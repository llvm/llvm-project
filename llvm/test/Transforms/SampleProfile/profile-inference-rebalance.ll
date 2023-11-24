; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-rebalance.prof | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-rebalance.prof | opt -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK2

; The test contains a "diamond" and a "triangle" that needs to be rebalanced
; after basic profile inference.
;
;                  +----------------+
;                  |    b11 [?]     |
;                  +----------------+
;                    |
;                    v
; +----------+     +----------------+
; | b13 [10] | <-- | b12 [65536]    |
; +----------+     +----------------+
;                    |
;                    v
; +----------+     +----------------+
; | b16 [?]  | <-- | b14 [65536]    |
; +----------+     +----------------+
;   |                |
;   |                v
;   |              +----------------+
;   |              |    b15 [?]     |
;   |              +----------------+
;   |                |
;   |                v
;   |              +----------------+
;   +------------> | b17 [65536]  | -+
;                  +----------------+  |
;                    |                 |
;                    v                 |
;                  +----------------+  |
;                  |    b18 [?]     |  |
;                  +----------------+  |
;                    |                 |
;                    v                 |
;                  +----------------+  |
;                  | b19 [65536]    | <+
;                  +----------------+
;                    |
;                    v
;                  +----------------+
;                  | b110 [65536]   |
;                  +----------------+

@yydebug = dso_local global i32 0, align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @countMultipliers(i32 %0, i32 %1) #0 {
b11:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br label %b12
; CHECK2: - b11: float = {{.*}}, int = {{.*}}, count = 65546

b12:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 2, i32 0, i64 -1)
  br i1 %cmp, label %b14, label %b13
; CHECK2: - b12: float = {{.*}}, int = {{.*}}, count = 65546

b13:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 3, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b13: float = {{.*}}, int = {{.*}}, count = 10

b14:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 4, i32 0, i64 -1)
  br i1 %cmp, label %b15, label %b16
; CHECK:  edge b14 -> b15 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b14 -> b16 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b14: float = {{.*}}, int = {{.*}}, count = 65536

b15:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 5, i32 0, i64 -1)
  br label %b17
; CHECK2: - b15: float = {{.*}}, int = {{.*}}, count = 32768

b16:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 6, i32 0, i64 -1)
  br label %b17
; CHECK2: - b16: float = {{.*}}, int = {{.*}}, count = 32768

b17:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 7, i32 0, i64 -1)
  br i1 %cmp, label %b18, label %b19
; CHECK:  edge b17 -> b18 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b17 -> b19 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b17: float = {{.*}}, int = {{.*}}, count = 65536

b18:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 8, i32 0, i64 -1)
  br label %b19
; CHECK2: - b18: float = {{.*}}, int = {{.*}}, count = 32768

b19:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 9, i32 0, i64 -1)
  br label %b110
; CHECK2: - b19: float = {{.*}}, int = {{.*}}, count = 65536

b110:
  call void @llvm.pseudoprobe(i64 -5758218299531803684, i64 10, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b110: float = {{.*}}, int = {{.*}}, count = 65536
}


; The test contains a triangle comprised of dangling blocks.
;
;                     +-----------+
;                     | b0 [2100] | -+
;                     +-----------+  |
;                       |            |
;                       |            |
;                       v            |
;                     +-----------+  |
;                  +- | b1 [2000] |  |
;                  |  +-----------+  |
;                  |    |            |
;                  |    |            |
;                  |    v            |
; +--------+       |  +-----------+  |
; | b4 [?] | <-----+- |  b2 [?]   |  |
; +--------+       |  +-----------+  |
;   |              |    |            |
;   |              |    |            |
;   |              |    v            |
;   |              |  +-----------+  |
;   |              +> |  b3 [?]   |  |
;   |                 +-----------+  |
;   |                   |            |
;   |                   |            |
;   |                   v            |
;   |                 +-----------+  |
;   +---------------> | b5 [2100] | <+
;                     +-----------+

define dso_local i32 @countMultipliers2(i32 %0, i32 %1) #0 {
b0:
  call void @llvm.pseudoprobe(i64 2506109673213838996, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b1, label %b5
; CHECK:  edge b0 -> b1 probability is 0x79e79e7a / 0x80000000 = 95.24% [HOT edge]
; CHECK:  edge b0 -> b5 probability is 0x06186186 / 0x80000000 = 4.76%
; CHECK2: - b0: float = {{.*}}, int = {{.*}}, count = 2100

b1:
  call void @llvm.pseudoprobe(i64 2506109673213838996, i64 2, i32 0, i64 -1)
  br i1 %cmp, label %b2, label %b3
; CHECK:  edge b1 -> b2 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b1 -> b3 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b1: float = {{.*}}, int = {{.*}}, count = 2000

b2:
  call void @llvm.pseudoprobe(i64 2506109673213838996, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b3, label %b4
; CHECK:  edge b2 -> b3 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b2 -> b4 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b2: float = {{.*}}, int = {{.*}}, count = 1000

b3:
  call void @llvm.pseudoprobe(i64 2506109673213838996, i64 4, i32 0, i64 -1)
  br label %b5
; CHECK:  edge b3 -> b5 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK2: - b3: float = {{.*}}, int = {{.*}}, count = 1500

b4:
  call void @llvm.pseudoprobe(i64 2506109673213838996, i64 5, i32 0, i64 -1)
  br label %b5
; CHECK:  edge b4 -> b5 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK2: - b4: float = {{.*}}, int = {{.*}}, count = 500

b5:
  call void @llvm.pseudoprobe(i64 2506109673213838996, i64 6, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b5: float = {{.*}}, int = {{.*}}, count = 2100

}


; The test contains a dangling subgraph that contains an exit dangling block.
; No rebalancing is necessary here.
;
;                 +-----------+
;                 | b31 [100] |
;                 +-----------+
;                   |
;                   |
;                   v
; +---------+     +-----------+
; | b34 [?] | <-- | b32 [100] |
; +---------+     +-----------+
;                   |
;                   |
;                   v
;                 +-----------+
;                 | b33 [100] |
;                 +-----------+

define dso_local i32 @countMultipliers3(i32 %0, i32 %1) #0 {
b31:
  call void @llvm.pseudoprobe(i64 -544905447084884130, i64 1, i32 0, i64 -1)
  br label %b32
; CHECK2: - b31: float = {{.*}}, int = {{.*}}, count = 100

b32:
  call void @llvm.pseudoprobe(i64 -544905447084884130, i64 2, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b34, label %b33
; CHECK:  edge b32 -> b34 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK:  edge b32 -> b33 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK2: - b32: float = {{.*}}, int = {{.*}}, count = 100

b33:
  call void @llvm.pseudoprobe(i64 -544905447084884130, i64 3, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b33: float = {{.*}}, int = {{.*}}, count = 100

b34:
  call void @llvm.pseudoprobe(i64 -544905447084884130, i64 4, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b34: float = {{.*}}, int = {{.*}}, count = 0

}

; Another dangling subgraph (b42, b43, b44) containing a single dangling block.
;
;      +----------+     +-----------+
;   +- | b42 [50] | <-- | b40 [100] |
;   |  +----------+     +-----------+
;   |    |                |
;   |    |                |
;   |    |                v
;   |    |              +-----------+
;   |    |              | b41 [50]  |
;   |    |              +-----------+
;   |    |                |
;   |    |                |
;   |    |                v
;   |    |              +-----------+
;   |    +------------> |  b43 [?]  |
;   |                   +-----------+
;   |                     |
;   |                     |
;   |                     v
;   |                   +-----------+
;   +-----------------> | b44 [100] |
;                       +-----------+

define dso_local i32 @countMultipliers4(i32 %0, i32 %1) #0 {
b40:
  call void @llvm.pseudoprobe(i64 -2989539179265513123, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b41, label %b42
; CHECK2: - b40: float = {{.*}}, int = {{.*}}, count = 100

b41:
  call void @llvm.pseudoprobe(i64 -2989539179265513123, i64 2, i32 0, i64 -1)
  br label %b43
; CHECK2: - b41: float = {{.*}}, int = {{.*}}, count = 50

b42:
  call void @llvm.pseudoprobe(i64 -2989539179265513123, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b43, label %b44
; CHECK:  edge b42 -> b43 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b42 -> b44 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b42: float = {{.*}}, int = {{.*}}, count = 50

b43:
  call void @llvm.pseudoprobe(i64 -2989539179265513123, i64 4, i32 0, i64 -1)
  br label %b44
; CHECK2: - b43: float = {{.*}}, int = {{.*}}, count = 75

b44:
  call void @llvm.pseudoprobe(i64 -2989539179265513123, i64 5, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b44: float = {{.*}}, int = {{.*}}, count = 100
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

attributes #0 = { noinline nounwind uwtable "use-sample-profile" }
attributes #4 = { inaccessiblememonly nounwind willreturn }

!llvm.pseudo_probe_desc = !{!7, !8, !9, !10}

!7 = !{i64 -5758218299531803684, i64 223598586707, !"countMultipliers", null}
!8 = !{i64 2506109673213838996, i64 2235985, !"countMultipliers2", null}
!9 = !{i64 -544905447084884130, i64 22985, !"countMultipliers3", null}
!10 = !{i64 -2989539179265513123, i64 2298578, !"countMultipliers4", null}
