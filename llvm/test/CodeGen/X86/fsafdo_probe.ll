; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=false --debug-only=mirfs-discriminators  < %s  -o - 2>&1 | FileCheck %s --check-prefixes=V0
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=true --debug-only=mirfs-discriminators  < %s  -o - 2>&1 | FileCheck %s --check-prefixes=V1

; Check that fs-afdo discriminators are generated.
; V0: foo.c:7:3: add FS discriminator, from 0 -> 11264
; V0: foo.c:9:5: add FS discriminator, from 0 -> 11264
; V0: Num of FS Discriminators: 2

; V1: foo.c:7:3: add FS discriminator, from 0 -> 256
; V1: foo.c:9:5: add FS discriminator, from 0 -> 256
; V1: Num of FS Discriminators: 2

target triple = "x86_64-unknown-linux-gnu"

%struct.Node = type { ptr }

define i32 @foo(ptr readonly %node, ptr readnone %root) !dbg !6  {
entry:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !8
  %cmp = icmp eq ptr %node, %root, !dbg !8
  br i1 %cmp, label %while.end4, label %while.cond1.preheader.lr.ph, !dbg !10

while.cond1.preheader.lr.ph:
  %tobool = icmp eq ptr %node, null
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !11
  br i1 %tobool, label %while.cond1.preheader.us.preheader, label %while.body2.preheader, !dbg !11

while.body2.preheader:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !11
  br label %while.body2, !dbg !11

while.cond1.preheader.us.preheader:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1), !dbg !10
  br label %while.cond1.preheader.us, !dbg !10

while.cond1.preheader.us:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1), !dbg !10
  br label %while.cond1.preheader.us, !dbg !10

while.body2:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !11
  br label %while.body2, !dbg !11

while.end4:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !12
  ret i32 0, !dbg !12
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.pseudo_probe_desc = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, debugInfoForProfiling: true, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 7, column: 15, scope: !9)
!9 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 0)
!10 = !DILocation(line: 7, column: 3, scope: !9)
!11 = !DILocation(line: 9, column: 5, scope: !9)
!12 = !DILocation(line: 14, column: 3, scope: !6)
!13 = !{i64 6699318081062747564, i64 138464321060, !"foo"}
