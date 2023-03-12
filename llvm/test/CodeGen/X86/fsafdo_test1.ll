; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=false < %s | FileCheck %s --check-prefixes=V0,V01
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=true < %s | FileCheck %s --check-prefixes=V1,V01
;
; Check that fs-afdo discriminators are generated.
; V01: .loc    1 7 3 is_stmt 0 discriminator 2 # foo.c:7:3
; V01: .loc    1 9 5 is_stmt 1 discriminator 2 # foo.c:9:5
; V0: .loc    1 9 5 is_stmt 0 discriminator 11266 # foo.c:9:5
; V0: .loc    1 7 3 is_stmt 1 discriminator 11266 # foo.c:7:3
; V1: .loc    1 9 5 is_stmt 0 discriminator 258 # foo.c:9:5
; V1: .loc    1 7 3 is_stmt 1 discriminator 258 # foo.c:7:3
; Check that variable __llvm_fs_discriminator__ is generated.
; V01: .type   __llvm_fs_discriminator__,@object # @__llvm_fs_discriminator__
; V01: .section        .rodata,"a",@progbits
; V01: .weak   __llvm_fs_discriminator__
; V01: __llvm_fs_discriminator__:
; V01: .byte   1
; V01: .size   __llvm_fs_discriminator__, 1

target triple = "x86_64-unknown-linux-gnu"

%struct.Node = type { %struct.Node* }

define i32 @foo(%struct.Node* readonly %node, %struct.Node* readnone %root) !dbg !6 {
entry:
  %cmp = icmp eq %struct.Node* %node, %root, !dbg !8
  br i1 %cmp, label %while.end4, label %while.cond1.preheader.lr.ph, !dbg !10

while.cond1.preheader.lr.ph:
  %tobool = icmp eq %struct.Node* %node, null
  br i1 %tobool, label %while.cond1.preheader.us.preheader, label %while.body2.preheader, !dbg !11

while.body2.preheader:
  br label %while.body2, !dbg !11

while.cond1.preheader.us.preheader:
  br label %while.cond1.preheader.us, !dbg !10

while.cond1.preheader.us:
  br label %while.cond1.preheader.us, !dbg !10

while.body2:
  br label %while.body2, !dbg !11

while.end4:
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, debugInfoForProfiling: true, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 7, column: 15, scope: !9)
!9 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 2)
!10 = !DILocation(line: 7, column: 3, scope: !9)
!11 = !DILocation(line: 9, column: 5, scope: !9)
!12 = !DILocation(line: 14, column: 3, scope: !6)
