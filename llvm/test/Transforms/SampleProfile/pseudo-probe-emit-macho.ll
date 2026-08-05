; REQUIRES: aarch64-registered-target
; RUN: opt < %s -passes=pseudo-probe -S -o %t
; RUN: FileCheck %s < %t --check-prefix=CHECK-IL
; RUN: llc %t -mtriple=arm64-apple-darwin -stop-after=pseudo-probe-inserter -o - | FileCheck %s --check-prefix=CHECK-MIR

; MachO
; RUN: llc %t -function-sections -mtriple=arm64-apple-darwin -filetype=asm -o %t1
; RUN: FileCheck %s < %t1 --check-prefix=CHECK-ASM-MACHO
; RUN: llc %t -function-sections -mtriple=arm64-apple-darwin -filetype=obj -o %t2
; RUN: llvm-readobj -Ss %t2 | FileCheck %s --check-prefix=CHECK-SEC-MACHO
; RUN: llvm-mc %t1 -triple=arm64-apple-darwin -filetype=obj -o %t3
; RUN: llvm-readobj -Ss %t3 | FileCheck %s --check-prefix=CHECK-SEC-MACHO-MC

@a = dso_local global i32 0, align 4

define void @foo(i32 %x) !dbg !3 {
bb0:
  %cmp = icmp eq i32 %x, 0
; CHECK-IL-LABEL: void @foo(i32 %x) !dbg ![[#]] {
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1), !dbg ![[#FAKELINE:]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID:]], 1, 0, 0
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID:]] 1 0 0 _foo
  br i1 %cmp, label %bb1, label %bb2

bb1:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 0, i64 -1), !dbg ![[#FAKELINE]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 3, 0, 0
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 4, 0, 0
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID]] 3 0 0 _foo
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID]] 4 0 0 _foo
  store i32 6, ptr @a, align 4
  br label %bb3

bb2:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 3, i32 0, i64 -1), !dbg ![[#FAKELINE]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 2, 0, 0
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 4, 0, 0
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID]] 2 0 0 _foo
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID]] 4 0 0 _foo
  store i32 8, ptr @a, align 4
  br label %bb3

bb3:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 4, i32 0, i64 -1), !dbg ![[#REALLINE:]]
  ret void, !dbg !12
}

declare void @bar(i32 %x)

define internal void @foo2(ptr %f) !dbg !4 {
entry:
; CHECK-IL-LABEL: void @foo2(ptr %f) !dbg ![[#]] {
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID2:]], i64 1, i32 0, i64 -1)
; CHECK-MIR: PSEUDO_PROBE [[#GUID2:]], 1, 0, 0
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID2:]] 1 0 0 _foo2
; Check pseudo_probe metadata attached to the indirect call instruction.
; CHECK-IL: call void %f(i32 1), !dbg ![[#PROBE0:]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID2]], 2, 1, 0
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID2]] 2 1 0 _foo2
  call void %f(i32 1), !dbg !13
; Check pseudo_probe metadata attached to the direct call instruction.
; CHECK-IL: call void @bar(i32 1), !dbg ![[#PROBE1:]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID2]], 3, 2, 0
; CHECK-ASM-MACHO: .pseudoprobe	[[#GUID2]] 3 2 0 _foo2
  call void @bar(i32 1)
  ret void
}

; CHECK-IL: Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
; CHECK-IL-NEXT: declare void @llvm.pseudoprobe(i64, i64, i32, i64)

; CHECK-IL: ![[#FOO:]] = distinct !DISubprogram(name: "foo"
; CHECK-IL: ![[#FAKELINE]] = !DILocation(line: 0, scope: ![[#FOO]])
; CHECK-IL: ![[#REALLINE]] = !DILocation(line: 2, scope: ![[#DISC0:]])
; CHECK-IL: ![[#DISC0]] = !DILexicalBlockFile(scope: ![[#FOO]], file: ![[#]], discriminator: 0)
; CHECK-IL: ![[#PROBE0]] = !DILocation(line: 2, column: 20, scope: ![[#SCOPE0:]])
;; A discriminator of 387973143 which is 0x17200017 in hexdecimal, stands for a direct call probe
;; with an index of 2.
; CHECK-IL: ![[#SCOPE0]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 387973143)
; CHECK-IL: ![[#PROBE1]] = !DILocation(line: 0, scope: ![[#SCOPE1:]])
;; A discriminator of 455082015 which is 0x1b20001f in hexdecimal, stands for a direct call probe
;; with an index of 3.
; CHECK-IL: ![[#SCOPE1]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 455082015)

; Check the generation of pseudo_probe_desc section for MachO
; CHECK-ASM-MACHO:      .section	__PSEUDO_PROBE,__probe_descs,regular,no_dead_strip+debug
; CHECK-ASM-MACHO-NEXT: .quad	[[#GUID]]
; CHECK-ASM-MACHO-NEXT: .quad	[[#HASH:]]
; CHECK-ASM-MACHO-NEXT: .byte	3
; CHECK-ASM-MACHO-NEXT: .ascii	"foo"
; CHECK-ASM-MACHO-NEXT: .quad	[[#GUID2]]
; CHECK-ASM-MACHO-NEXT: .quad	[[#HASH2:]]
; CHECK-ASM-MACHO-NEXT: .byte	4
; CHECK-ASM-MACHO-NEXT: .ascii	"foo2"

; CHECK-SEC-MACHO-LABEL: Sections [
; CHECK-SEC-MACHO:       Name: __probe_descs
; CHECK-SEC-MACHO-NEXT:  Segment: __PSEUDO_PROBE
; CHECK-SEC-MACHO:       Attributes [ (0x120000)
; CHECK-SEC-MACHO-NEXT:    Debug (0x20000)
; CHECK-SEC-MACHO-NEXT:    NoDeadStrip (0x100000)
; CHECK-SEC-MACHO:       Name: __probes
; CHECK-SEC-MACHO-NEXT:  Segment: __PSEUDO_PROBE
; CHECK-SEC-MACHO:       Attributes [ (0x120000)
; CHECK-SEC-MACHO-NEXT:    Debug (0x20000)
; CHECK-SEC-MACHO-NEXT:    NoDeadStrip (0x100000)

; CHECK-SEC-MACHO-MC-LABEL: Sections [
; CHECK-SEC-MACHO-MC:       Name: __probe_descs
; CHECK-SEC-MACHO-MC-NEXT:  Segment: __PSEUDO_PROBE
; CHECK-SEC-MACHO-MC:       Attributes [ (0x120000)
; CHECK-SEC-MACHO-MC-NEXT:    Debug (0x20000)
; CHECK-SEC-MACHO-MC-NEXT:    NoDeadStrip (0x100000)
; CHECK-SEC-MACHO-MC:       Name: __probes
; CHECK-SEC-MACHO-MC-NEXT:  Segment: __PSEUDO_PROBE
; CHECK-SEC-MACHO-MC:       Attributes [ (0x120000)
; CHECK-SEC-MACHO-MC-NEXT:    Debug (0x20000)
; CHECK-SEC-MACHO-MC-NEXT:    NoDeadStrip (0x100000)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, unit: !0, retainedNodes: !2)
!4 = distinct !DISubprogram(name: "foo2", scope: !1, file: !1, line: 2, type: !5, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.9.0"}
!12 = !DILocation(line: 2, scope: !14)
!13 = !DILocation(line: 2, column: 20, scope: !4)
!14 = !DILexicalBlockFile(scope: !3, file: !1, discriminator: 1)
