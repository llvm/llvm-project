; REQUIRES: x86_64-linux
; RUN: opt < %s -passes='pseudo-probe,cgscc(inline)' -function-sections -mtriple=x86_64-unknown-linux-gnu -S -o %t
; RUN: FileCheck %s < %t --check-prefix=CHECK-IL
; RUN: llc -pseudo-probe-for-profiling -function-sections <%t -filetype=asm -o %t1
; RUN: FileCheck %s < %t1 --check-prefix=CHECK-ASM
; RUN: llc -pseudo-probe-for-profiling -function-sections <%t -filetype=obj -o %t2
; RUN: llvm-objdump --section-headers  %t2 | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llvm-mc -filetype=asm <%t1 -o %t3
; RUN: FileCheck %s < %t3 --check-prefix=CHECK-ASM
; RUN: llvm-mc -filetype=obj <%t1 -o %t4
; RUN: llvm-objdump --section-headers  %t4 | FileCheck %s --check-prefix=CHECK-OBJ

define dso_local void @foo2() !dbg !7 {
; CHECK-IL:  call void @llvm.pseudoprobe(i64 [[#GUID1:]], i64 1, i32 0, i64 -1), !dbg ![[#]]
; CHECK-ASM: .pseudoprobe	[[#GUID1:]] 1 0 0
  ret void, !dbg !10
}

define dso_local void @foo() #0 !dbg !11 {
; CHECK-IL:  call void @llvm.pseudoprobe(i64 [[#GUID2:]], i64 1, i32 0, i64 -1), !dbg ![[#]]
; CHECK-IL:  call void @llvm.pseudoprobe(i64 [[#GUID1]], i64 1, i32 0, i64 -1), !dbg ![[#DL1:]]
; CHECK-ASM: .pseudoprobe	[[#GUID2:]] 1 0 0
; CHECK-ASM: .pseudoprobe	[[#GUID1]] 1 0 0 @ [[#GUID2]]:2
  call void @foo2(), !dbg !12
  ret void, !dbg !13
}

define dso_local i32 @entry() !dbg !14 {
; CHECK-IL:  call void @llvm.pseudoprobe(i64 [[#GUID3:]], i64 1, i32 0, i64 -1), !dbg ![[#]]
; CHECK-IL:  call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 1, i32 0, i64 -1), !dbg ![[#DL2:]]
; CHECK-IL:  call void @llvm.pseudoprobe(i64 [[#GUID1]], i64 1, i32 0, i64 -1), !dbg ![[#DL3:]]
; CHECK-ASM: .pseudoprobe	[[#GUID3:]] 1 0 0
; CHECK-ASM: .pseudoprobe	[[#GUID2]] 1 0 0 @ [[#GUID3]]:2
; CHECK-ASM: .pseudoprobe	[[#GUID1]] 1 0 0 @ [[#GUID3]]:2 @ [[#GUID2]]:2
  call void @foo(), !dbg !18
  ret i32 0, !dbg !19
}


; CHECK-IL: ![[#SCOPE1:]] = distinct !DISubprogram(name: "foo2"
; CHECK-IL: ![[#SCOPE2:]] = distinct !DISubprogram(name: "foo"
; CHECK-IL: ![[#DL1]] = !DILocation(line: 3, column: 1,  scope: ![[#SCOPE1]], inlinedAt: ![[#INL1:]])
; CHECK-IL: ![[#INL1]] = distinct !DILocation(line: 7, column: 3, scope: ![[#BL1:]])
;; A discriminator of 186646551 which is 0xb200017 in hexdecimal, stands for a direct call probe
;; with an index of 2 and a scale of 100%.
; CHECK-IL: ![[#BL1]] = !DILexicalBlockFile(scope: ![[#SCOPE2]], file: !1, discriminator: 186646551)
; CHECK-IL: ![[#SCOPE3:]] = distinct !DISubprogram(name: "entry"
; CHECK-IL: ![[#DL2]] = !DILocation(line: 7, column: 3,  scope: ![[#SCOPE2]], inlinedAt: ![[#INL2:]])
; CHECK-IL: ![[#INL2]] = distinct !DILocation(line: 11, column: 3, scope: ![[#BL2:]])
; CHECK-IL: ![[#BL2]] = !DILexicalBlockFile(scope: ![[#SCOPE3]], file: !1, discriminator: 186646551)
; CHECK-IL: ![[#DL3]] = !DILocation(line: 3, column: 1,  scope: ![[#SCOPE1]], inlinedAt: ![[#INL3:]])
; CHECK-IL: ![[#INL3]] = distinct !DILocation(line: 7, column: 3,  scope: ![[#BL1]], inlinedAt: ![[#INL2]])


; Check the generation of .pseudo_probe_desc section
; CHECK-ASM: .section .pseudo_probe_desc,"G",@progbits,.pseudo_probe_desc_foo2,comdat
; CHECK-ASM-NEXT: .quad [[#GUID1]]
; CHECK-ASM-NEXT: .quad [[#HASH1:]]
; CHECK-ASM-NEXT: .byte	4
; CHECK-ASM-NEXT: .ascii "foo2"
; CHECK-ASM-NEXT: .section .pseudo_probe_desc,"G",@progbits,.pseudo_probe_desc_foo,comdat
; CHECK-ASM-NEXT: .quad [[#GUID2]]
; CHECK-ASM-NEXT: .quad [[#HASH2:]]
; CHECK-ASM-NEXT: .byte	3
; CHECK-ASM-NEXT: .ascii "foo"
; CHECK-ASM-NEXT: .section .pseudo_probe_desc,"G",@progbits,.pseudo_probe_desc_entry,comdat
; CHECK-ASM-NEXT: .quad [[#GUID3]]
; CHECK-ASM-NEXT: .quad [[#HASH3:]]
; CHECK-ASM-NEXT: .byte	5
; CHECK-ASM-NEXT: .ascii "entry"

; CHECK-OBJ: .pseudo_probe_desc
; CHECK-OBJ: .pseudo_probe

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.c", directory: "any")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "foo2", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 3, column: 1, scope: !7)
!11 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 7, column: 3, scope: !11)
!13 = !DILocation(line: 8, column: 1, scope: !11)
!14 = distinct !DISubprogram(name: "entry", scope: !1, file: !1, line: 10, type: !15, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!17}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 11, column: 3, scope: !14)
!19 = !DILocation(line: 12, column: 3, scope: !14)
