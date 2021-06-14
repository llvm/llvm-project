; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/profile-correlation-irreducible-loops.prof | opt -analyze -block-freq -enable-new-pm=0  -use-iterative-bfi-inference | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/profile-correlation-irreducible-loops.prof -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -analyze -block-freq -enable-new-pm=0  -use-iterative-bfi-inference | FileCheck %s --check-prefix=CHECK3

; The C++ code for this test case is from c-parse.c in 403.gcc (SPEC2006)
; The problem with BFI for the test is solved by applying iterative inference.
; The corresponding CFG graph is shown below, with intended counts for every
; basic block. The hot loop, b3->b4->b2, is not getting proper (large) counts
; unless the -use-iterative-bfi-inference option is specified.
;
;   +-------------------------------------------+
;   |                                           |
;   |                   +----------+            |
;   |                   |  b1 [1]  |            |
;   |                   +----------+            |
;   |                     |                     |
;   |                     |                     |
;   |                     v                     |
;   |                   +----------+            |
;   |    +------------> | b2 [625] | -+         |
;   |    |              +----------+  |         |
;   |    |                |           |         |
;   |    |                |           |         |
;   |    |                v           |         |
;   |  +----------+     +----------+  |         |
;   |  | b4 [624] | <-- | b3 [625] | <+---------+
;   |  +----------+     +----------+  |
;   |                     |           |
;   +----+                |           |
;        |                v           v
;      +----------+     +--------------------+
;      |  b8 [1]  | <-- |       b7 [2]       |
;      +----------+     +--------------------+
;                         |           ^
;                         |           |
;                         v           |
;      +----------+     +----------+  |
;      |  b9 [1]  | <-- |  b5 [2]  |  |
;      +----------+     +----------+  |
;                         |           |
;                         |           |
;                         v           |
;                       +----------+  |
;                       |  b6 [1]  | -+
;                       +----------+

@yydebug = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @yyparse_1() #0 {
b1:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 1, i32 0, i64 -1)
  %0 = load i32, i32* @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br label %b2
; CHECK: - b1: float = {{.*}}, int = {{.*}}, count = 1

b2:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 2, i32 0, i64 -1)
  br i1 %cmp, label %b7, label %b3
; CHECK: - b2: float = {{.*}}, int = {{.*}}, count = 625

b3:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 3, i32 0, i64 -1)
  br i1 %cmp, label %b7, label %b4
; CHECK: - b3: float = {{.*}}, int = {{.*}}, count = 625
; CHECK2: br i1 %cmp, label %b7, label %b4,
; CHECK2-SAME: !prof ![[END172_PROF:[0-9]+]]

b4:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 4, i32 0, i64 -1)
  br label %b2
; CHECK: - b4: float = {{.*}}, int = {{.*}}, count = 624

b5:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 5, i32 0, i64 -1)
  br i1 %cmp, label %b9, label %b6
; CHECK: - b5: float = {{.*}}, int = {{.*}}, count = 2

b6:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 6, i32 0, i64 -1)
  br label %b7
; CHECK: - b6: float = {{.*}}, int = {{.*}}, count = 1

b7:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 7, i32 0, i64 -1)
  br i1 %cmp, label %b5, label %b8
; CHECK: - b7: float = {{.*}}, int = {{.*}}, count = 2
; CHECK2: br i1 %cmp, label %b5, label %b8,
; CHECK2-SAME: !prof ![[FALSE4858_PROF:[0-9]+]]

b8:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 8, i32 0, i64 -1)
  br label %b3
; CHECK: - b8: float = {{.*}}, int = {{.*}}, count = 1

b9:
  call void @llvm.pseudoprobe(i64 -7702751003264189226, i64 9, i32 0, i64 -1)
  %1 = load i32, i32* @yydebug, align 4
  ret i32 %1
; CHECK: - b9: float = {{.*}}, int = {{.*}}, count = 1

}

; Another difficult (for BFI) instance with irreducible loops,
; containing 'indirectbr'. The corresponding CFG graph is shown below, with
; intended counts for every basic block.
;
;      +-----------+
;      |  b1 [1]   |
;      +-----------+
;        |
;        |
;        v
;      +------------------------+
;   +- |        b2 [86]         | <+
;   |  +------------------------+  |
;   |    |            |            |
;   |    |            |            |
;   |    v            |            |
;   |  +-----------+  |            |
;   |  | b3 [8212] | <+-------+    |
;   |  +-----------+  |       |    |
;   |    |            |       |    |
;   |    |            |       |    |
;   |    v            v       |    |
;   |  +------------------------+  |
;   |  |  indirectgoto [17747]  | -+
;   |  +------------------------+
;   |    |            ^  |
;   |    |            +--+
;   |    v
;   |  +-----------+
;   +> |  b4 [1]   |
;      +-----------+

; Function Attrs: nounwind uwtable
define dso_local i32 @foo1() #0 !prof !132 {
b1:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 1, i32 0, i64 -1)
  %0 = load i32, i32* @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br label %b2
; CHECK3: - b1: float = {{.*}}, int = {{.*}}, count = 1

b2:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 2, i32 0, i64 -1)
  %1 = load i32, i32* @yydebug, align 4
  switch i32 %1, label %b4 [
    i32 1, label %indirectgoto
    i32 2, label %b3
  ], !prof !133
; CHECK3: - b2: float = {{.*}}, int = {{.*}}, count = 86

b3:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 3, i32 0, i64 -1)
  br label %indirectgoto
; CHECK3: - b3: float = {{.*}}, int = {{.*}}, count = 8212

b4:
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 4, i32 0, i64 -1)
  %2 = load i32, i32* @yydebug, align 4
  ret i32 %2
; CHECK3: - b4: float = {{.*}}, int = {{.*}}, count = 1

indirectgoto:
  %indirect.goto.dest = alloca i8, align 4
  call void @llvm.pseudoprobe(i64 7682762345278052905, i64 5, i32 0, i64 -1)
  indirectbr i8* %indirect.goto.dest, [label %b2, label %indirectgoto, label %b4, label %b3], !prof !134
; CHECK3: - indirectgoto: float = {{.*}}, int = {{.*}}, count = 17747

}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { noinline nounwind uwtable "use-sample-profile"}
attributes #1 = { nounwind }

!llvm.pseudo_probe_desc = !{!1079, !4496}
!1079 = !{i64 -7702751003264189226, i64 158496288380146391, !"yyparse_1", null}
!4496 = !{i64 7682762345278052905, i64 404850113186107133, !"foo1", null}
!132 = !{!"function_entry_count", i64 1}
!133 = !{!"branch_weights", i32 0, i32 86, i32 0}
!134 = !{!"branch_weights", i32 85, i32 9449, i32 1, i32 8212}

; CHECK2: ![[END172_PROF]] = !{!"branch_weights", i32 1, i32 1003}
; CHECK2: ![[FALSE4858_PROF]] = !{!"branch_weights", i32 2, i32 1}
