; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mattr=+sse2 -S | FileCheck %s

; Verify that X86PartialReduction::tryByteSumReplacement carries the
; original add instruction's !dbg location onto the freshly emitted
; psadbw / shuffle sequence so source locations survive into the
; optimized IR under -g.

@a = global [1024 x i8] zeroinitializer, align 16

; CHECK-LABEL: @byte_sum_v16_i32
; CHECK: call <2 x i64> @llvm.x86.sse2.psad.bw({{.*}}), !dbg ![[#LOC:]]
; CHECK: ![[#LOC]] = !DILocation(line: 42,
define i32 @byte_sum_v16_i32() nounwind !dbg !6 {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <16 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <16 x i8>, ptr %p, align 16
  %z = zext <16 x i8> %wide.load to <16 x i32>, !dbg !8
  %add = add nsw <16 x i32> %z, %vec.phi
  %index.next = add i64 %index, 16
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %add)
  ret i32 %ext
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "byte-sum-debugloc.c", directory: "/tmp")
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "byte_sum_v16_i32", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !{})
!8 = !DILocation(line: 42, column: 1, scope: !6)
