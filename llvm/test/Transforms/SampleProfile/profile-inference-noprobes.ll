; RUN: opt < %s -passes=sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-noprobes.prof -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-use-profi -sample-profile-file=%S/Inputs/profile-inference-noprobes.prof | opt -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK2


; The test verifies that profile inference can be applied for non-probe-based
; profiles.
;
; +---------+     +----------+
; | b3 [40] | <-- | b1 [100] |
; +---------+     +----------+
;                   |
;                   |
;                   v
;                 +----------+
;                 | b2 [0]   |
;                 +----------+

@yydebug = dso_local global i32 0, align 4

define void @test_4() #0 !dbg !4 {
;entry:
;  ret void, !dbg !9
b1:
  %0 = load i32, i32* @yydebug, align 4
  %cmp = icmp ne i32 %0, 0, !dbg !9
  br i1 %cmp, label %b2, label %b3, !dbg !9
; CHECK2: - b1: float = {{.*}}, int = {{.*}}, count = 100

b2:
  ret void
; CHECK2: - b2: float = {{.*}}, int = {{.*}}, count = 60

b3:
  ret void, !dbg !10
; CHECK2: - b3: float = {{.*}}, int = {{.*}}, count = 40
}

; CHECK: {{.*}} = !{!"function_entry_count", i64 100}
; CHECK: {{.*}} = !{!"branch_weights", i32 60, i32 40}

attributes #0 = { noinline nounwind uwtable "use-sample-profile"}

!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 237249) (llvm/trunk 237261)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "entry_counts.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "test_4", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 3.7.0 (trunk 237249) (llvm/trunk 237261)"}
!9 = !DILocation(line: 1, column: 15, scope: !4)
!10 = !DILocation(line: 3, column: 15, scope: !4)
