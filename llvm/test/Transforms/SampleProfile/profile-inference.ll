; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference.prof | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference.prof | opt -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK2

; The test verifies that profile inference correctly builds branch probabilities
; from sampling-based block counts.
;
; +---------+     +----------+
; | b3 [40] | <-- | b1 [100] |
; +---------+     +----------+
;                   |
;                   |
;                   v
;                 +----------+
;                 | b2 [60]  |
;                 +----------+

@yydebug = dso_local global i32 0, align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @test_1() #0 {
b1:
  call void @llvm.pseudoprobe(i64 7964825052912775246, i64 1, i32 0, i64 -1)
  %0 = load i32, ptr @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b2, label %b3
; CHECK:  edge b1 -> b2 probability is 0x4ccccccd / 0x80000000 = 60.00%
; CHECK:  edge b1 -> b3 probability is 0x33333333 / 0x80000000 = 40.00%
; CHECK2: - b1: float = {{.*}}, int = {{.*}}, count = 100

b2:
  call void @llvm.pseudoprobe(i64 7964825052912775246, i64 2, i32 0, i64 -1)
  ret i32 %0
; CHECK2: - b2: float = {{.*}}, int = {{.*}}, count = 60

b3:
  call void @llvm.pseudoprobe(i64 7964825052912775246, i64 3, i32 0, i64 -1)
  ret i32 %0
; CHECK2: - b3: float = {{.*}}, int = {{.*}}, count = 40
}


; The test verifies that profile inference correctly builds branch probabilities
; from sampling-based block counts in the presence of "dangling" probes (whose
; block counts are missing).
;
; +---------+     +----------+
; | b3 [10] | <-- | b1 [100] |
; +---------+     +----------+
;                   |
;                   |
;                   v
;                 +----------+
;                 | b2 [?]  |
;                 +----------+

; Function Attrs: nounwind uwtable
define dso_local i32 @test_2() #0 {
b1:
  call void @llvm.pseudoprobe(i64 -6216829535442445639, i64 1, i32 0, i64 -1)
  %0 = load i32, ptr @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b2, label %b3
; CHECK:  edge b1 -> b2 probability is 0x73333333 / 0x80000000 = 90.00%
; CHECK:  edge b1 -> b3 probability is 0x0ccccccd / 0x80000000 = 10.00%
; CHECK2: - b1: float = {{.*}}, int = {{.*}}, count = 100

b2:
  call void @llvm.pseudoprobe(i64 -6216829535442445639, i64 2, i32 0, i64 -1)
  ret i32 %0
; CHECK2: - b2: float = {{.*}}, int = {{.*}}, count = 90

b3:
  call void @llvm.pseudoprobe(i64 -6216829535442445639, i64 3, i32 0, i64 -1)
  ret i32 %0
}
; CHECK2: - b3: float = {{.*}}, int = {{.*}}, count = 10


; The test verifies that profi is able to infer block counts from hot subgraphs.
;
; +---------+     +---------+
; | b4 [?]  | <-- | b1 [?]  |
; +---------+     +---------+
;   |               |
;   |               |
;   v               v
; +---------+     +---------+
; | b5 [89] |     | b2 [?]  |
; +---------+     +---------+
;                   |
;                   |
;                   v
;                 +---------+
;                 | b3 [13] |
;                 +---------+

; Function Attrs: nounwind uwtable
define dso_local i32 @test_3() #0 {
b1:
  call void @llvm.pseudoprobe(i64 1649282507922421973, i64 1, i32 0, i64 -1)
  %0 = load i32, ptr @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b2, label %b4
; CHECK:  edge b1 -> b2 probability is 0x10505050 / 0x80000000 = 12.75%
; CHECK:  edge b1 -> b4 probability is 0x6fafafb0 / 0x80000000 = 87.25%
; CHECK2: - b1: float = {{.*}}, int = {{.*}}, count = 102

b2:
  call void @llvm.pseudoprobe(i64 1649282507922421973, i64 2, i32 0, i64 -1)
  br label %b3
; CHECK:  edge b2 -> b3 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b2: float = {{.*}}, int = {{.*}}, count = 13

b3:
  call void @llvm.pseudoprobe(i64 1649282507922421973, i64 3, i32 0, i64 -1)
  ret i32 %0
; CHECK2: - b3: float = {{.*}}, int = {{.*}}, count = 13

b4:
  call void @llvm.pseudoprobe(i64 1649282507922421973, i64 4, i32 0, i64 -1)
  br label %b5
; CHECK:  edge b4 -> b5 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b4: float = {{.*}}, int = {{.*}}, count = 89

b5:
  call void @llvm.pseudoprobe(i64 1649282507922421973, i64 5, i32 0, i64 -1)
  ret i32 %0
; CHECK2: - b5: float = {{.*}}, int = {{.*}}, count = 89
}


; A larger test to verify that profile inference correctly identifies hot parts
; of the control-flow graph.
;
;                +-----------+
;                |  b1 [?]   |
;                +-----------+
;                  |
;                  |
;                  v
; +--------+     +-----------+
; | b3 [1] | <-- | b2 [5993] |
; +--------+     +-----------+
;   |              |
;   |              |
;   |              v
;   |            +-----------+     +--------+
;   |            | b4 [5992] | --> | b6 [?] |
;   |            +-----------+     +--------+
;   |              |                 |
;   |              |                 |
;   |              v                 |
;   |            +-----------+       |
;   |            | b5 [5992] |       |
;   |            +-----------+       |
;   |              |                 |
;   |              |                 |
;   |              v                 |
;   |            +-----------+       |
;   |            |  b7 [?]   |       |
;   |            +-----------+       |
;   |              |                 |
;   |              |                 |
;   |              v                 |
;   |            +-----------+       |
;   |            | b8 [5992] | <-----+
;   |            +-----------+
;   |              |
;   |              |
;   |              v
;   |            +-----------+
;   +----------> |  b9 [?]   |
;                +-----------+

; Function Attrs: nounwind uwtable
define dso_local i32 @sum_of_squares() #0 {
b1:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 1, i32 0, i64 -1)
  %0 = load i32, ptr @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br label %b2
; CHECK:  edge b1 -> b2 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b1: float = {{.*}}, int = {{.*}}, count = 5993

b2:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 2, i32 0, i64 -1)
  br i1 %cmp, label %b4, label %b3
; CHECK:  edge b2 -> b4 probability is 0x7ffa8844 / 0x80000000 = 99.98%
; CHECK:  edge b2 -> b3 probability is 0x000577bc / 0x80000000 = 0.02%
; CHECK2: - b2: float = {{.*}}, int = {{.*}}, count = 5993

b3:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 3, i32 0, i64 -1)
  br label %b9
; CHECK:  edge b3 -> b9 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b3: float = {{.*}}, int = {{.*}}, count = 1

b4:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 4, i32 0, i64 -1)
  br i1 %cmp, label %b5, label %b6
; CHECK:  edge b4 -> b5 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK:  edge b4 -> b6 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK2: - b4: float = {{.*}}, int = {{.*}}, count = 5992

b5:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 5, i32 0, i64 -1)
  br label %b7
; CHECK:  edge b5 -> b7 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b5: float = {{.*}}, int = {{.*}}, count = 5992

b6:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 6, i32 0, i64 -1)
  br label %b8
; CHECK:  edge b6 -> b8 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b6: float = {{.*}}, int = {{.*}}, count = 0

b7:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 7, i32 0, i64 -1)
  br label %b8
; CHECK:  edge b7 -> b8 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b7: float = {{.*}}, int = {{.*}}, count = 5992

b8:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 8, i32 0, i64 -1)
  br label %b9
; CHECK:  edge b8 -> b9 probability is 0x80000000 / 0x80000000 = 100.00%
; CHECK2: - b8: float = {{.*}}, int = {{.*}}, count = 5992

b9:
  call void @llvm.pseudoprobe(i64 -907520326213521421, i64 9, i32 0, i64 -1)
  ret i32 %0
}
; CHECK2: - b9: float = {{.*}}, int = {{.*}}, count = 5993


declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { noinline nounwind uwtable "use-sample-profile"}
attributes #1 = { nounwind }

!llvm.pseudo_probe_desc = !{!6, !7, !8, !9}

!6 = !{i64 7964825052912775246, i64 4294967295, !"test_1", null}
!7 = !{i64 -6216829535442445639, i64 37753817093, !"test_2", null}
!8 = !{i64 1649282507922421973, i64 69502983527, !"test_3", null}
!9 = !{i64 -907520326213521421, i64 175862120757, !"sum_of_squares", null}
