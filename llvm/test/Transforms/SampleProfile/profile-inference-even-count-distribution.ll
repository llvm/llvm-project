; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-even-count-distribution.prof | opt -analyze -branch-prob -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-even-count-distribution.prof | opt -analyze -block-freq  -enable-new-pm=0 | FileCheck %s --check-prefix=CHECK2

; The test verifies that counts are evenly distributed among branches with
; equal weights.
;
; +-----------+     +-----------+
; | b3 [0]    | <-- | b1 [1000] |
; +-----------+     +-----------+
;   |                 |
;   |                 |
;   |                 v
;   |               +-----------+
;   |               | b2 [0]    |
;   |               +-----------+
;   |                 |
;   |                 |
;   |                 v
;   |               +-----------+
;   +-------------> | b4 [1000] |
;                   +-----------+

@yydebug = dso_local global i32 0, align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @foo1(i32 %0, i32 %1) #0 {
b11:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b12, label %b13
; CHECK:  edge b11 -> b12 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b11 -> b13 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b11: float = {{.*}}, int = {{.*}}, count = 1000

b12:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 2, i32 0, i64 -1)
  br label %b14
; CHECK2: - b12: float = {{.*}}, int = {{.*}}, count = 500

b13:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 3, i32 0, i64 -1)
  br label %b14
; CHECK2: - b13: float = {{.*}}, int = {{.*}}, count = 500

b14:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 4, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b14: float = {{.*}}, int = {{.*}}, count = 1000
}


; The test verifies that counts are evenly distributed when the entry basic
; block is dangling.
;
; +-----------+
; |  b1 [?]   | -+
; +-----------+  |
;   |            |
;   |            |
;   v            |
; +-----------+  |
; |  b2 [?]   |  |
; +-----------+  |
;   |            |
;   |            |
;   v            |
; +-----------+  |
; | b3 [1000] | <+
; +-----------+

define dso_local i32 @foo2(i32 %0, i32 %1) #0 {
b21:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b22, label %b23
; CHECK:  edge b21 -> b22 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b21 -> b23 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b21: float = {{.*}}, int = {{.*}}, count = 1000

b22:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 2, i32 0, i64 -1)
  br label %b23
; CHECK2: - b22: float = {{.*}}, int = {{.*}}, count = 500

b23:
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 3, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b23: float = {{.*}}, int = {{.*}}, count = 1000

}

; The test verifies even count distribution in the presence of multiple sinks.
;
;                +-----------+
;                | b1 [1000] |
;                +-----------+
;                  |
;                  |
;                  v
;                +-----------+
;                |  b2 [?]   | -+
;                +-----------+  |
;                  |            |
;                  |            |
;                  v            |
; +--------+     +-----------+  |
; | b5 [?] | <-- |  b3 [?]   |  |
; +--------+     +-----------+  |
;   |              |            |
;   |              |            |
;   |              v            |
;   |            +-----------+  |
;   |            | b4 [1000] | <+
;   |            +-----------+
;   |              |
;   |              |
;   |              v
;   |            +-----------+
;   +----------> | b6 [1000] |
;                +-----------+
;

define dso_local i32 @foo3(i32 %0, i32 %1) #0 {
b31:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 1, i32 0, i64 -1)
  %cmp = icmp ne i32 %0, 0
  br label %b32
; CHECK2: - b31: float = {{.*}}, int = {{.*}}, count = 1000

b32:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 2, i32 0, i64 -1)
  br i1 %cmp, label %b33, label %b34
; CHECK:  edge b32 -> b33 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge b32 -> b34 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK2: - b32: float = {{.*}}, int = {{.*}}, count = 1000

b33:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b35, label %b34
; CHECK:  edge b33 -> b35 probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK:  edge b33 -> b34 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK2: - b33: float = {{.*}}, int = {{.*}}, count = 500

b34:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 4, i32 0, i64 -1)
  br label %b36
; CHECK2: - b34: float = {{.*}}, int = {{.*}}, count = 1000

b35:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 5, i32 0, i64 -1)
  br label %b36
; CHECK2: - b35: float = {{.*}}, int = {{.*}}, count = 0

b36:
  call void @llvm.pseudoprobe(i64 -7908226060800700466, i64 6, i32 0, i64 -1)
  ret i32 %1
; CHECK2: - b36: float = {{.*}}, int = {{.*}}, count = 1000
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
