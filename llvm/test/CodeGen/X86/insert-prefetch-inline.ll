; RUN: llc < %s -prefetch-hints-file=%S/insert-prefetch-inline.afdo | FileCheck %s
;
; Verify we can insert prefetch instructions in code belonging to inlined
; functions.
;
; ModuleID = 'test.cc'

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local i32 @sum(i32* nocapture readonly %arr, i32 %pos1, i32 %pos2) local_unnamed_addr #0 !dbg !7 {
entry:
  %idxprom = sext i32 %pos1 to i64, !dbg !10
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom, !dbg !10
  %0 = load i32, i32* %arrayidx, align 4, !dbg !10, !tbaa !11
  %idxprom1 = sext i32 %pos2 to i64, !dbg !15
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1, !dbg !15
  %1 = load i32, i32* %arrayidx2, align 4, !dbg !15, !tbaa !11
  %add = add nsw i32 %1, %0, !dbg !16
  ret i32 %add, !dbg !17
}

; "caller" inlines "sum". The associated .afdo file references instructions
; in "caller" that came from "sum"'s inlining.
;
; Function Attrs: norecurse nounwind readonly uwtable
define dso_local i32 @caller(i32* nocapture readonly %arr) local_unnamed_addr #0 !dbg !18 {
entry:
  %0 = load i32, i32* %arr, align 4, !dbg !19, !tbaa !11
  %arrayidx2.i = getelementptr inbounds i32, i32* %arr, i64 2, !dbg !21
  %1 = load i32, i32* %arrayidx2.i, align 4, !dbg !21, !tbaa !11
  %add.i = add nsw i32 %1, %0, !dbg !22
  ret i32 %add.i, !dbg !23
}

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk 324940) (llvm/trunk 324941)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, debugInfoForProfiling: true)
!1 = !DIFile(filename: "test.cc", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 324940) (llvm/trunk 324941)"}
!7 = distinct !DISubprogram(name: "sum", linkageName: "sum", scope: !8, file: !8, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!8 = !DIFile(filename: "./test.h", directory: "/tmp")
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 6, column: 10, scope: !7)
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}
!15 = !DILocation(line: 6, column: 22, scope: !7)
!16 = !DILocation(line: 6, column: 20, scope: !7)
!17 = !DILocation(line: 6, column: 3, scope: !7)
!18 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: !1, file: !1, line: 4, type: !9, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!19 = !DILocation(line: 6, column: 10, scope: !7, inlinedAt: !20)
!20 = distinct !DILocation(line: 6, column: 10, scope: !18)
!21 = !DILocation(line: 6, column: 22, scope: !7, inlinedAt: !20)
!22 = !DILocation(line: 6, column: 20, scope: !7, inlinedAt: !20)
!23 = !DILocation(line: 6, column: 3, scope: !18)

; CHECK-LABEL: caller:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: .loc 1 6 22 prologue_end
; CHECK-NEXT: prefetchnta 23464(%rdi)
; CHECK-NEXT: movl 8(%rdi), %eax
; CHECK-NEXT: .loc 1 6 20 is_stmt 0 discriminator 2
; CHECK-NEXT: prefetchnta 8764(%rdi)
; CHECK-NEXT: prefetchnta 64(%rdi)
; CHECK-NEXT: addl (%rdi), %eax
