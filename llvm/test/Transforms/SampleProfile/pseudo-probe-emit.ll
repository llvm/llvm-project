; REQUIRES: x86_64-linux
; RUN: opt < %s -passes=pseudo-probe -function-sections -S -o %t
; RUN: FileCheck %s < %t --check-prefix=CHECK-IL
; RUN: llc %t -stop-after=pseudo-probe-inserter -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc %t -function-sections -filetype=asm -o %t1
; RUN: FileCheck %s < %t1 --check-prefix=CHECK-ASM
; RUN: llc %t -function-sections -filetype=obj -o %t2
; RUN: llvm-readelf -S -g %t2 | FileCheck %s --check-prefix=CHECK-SEC
; RUN: llvm-mc %t1 -filetype=obj -o %t3
; RUN: llvm-readelf -S -g %t3 | FileCheck %s --check-prefix=CHECK-SEC

; RUN: llc %t -function-sections -unique-section-names=0 -filetype=obj -o %t4
; RUN: llvm-readelf -S %t4 | FileCheck %s --check-prefix=CHECK-SEC2

;; Check the generation of pseudoprobe intrinsic call.

@a = dso_local global i32 0, align 4

define void @foo(i32 %x) !dbg !3 {
bb0:
  %cmp = icmp eq i32 %x, 0
; CHECK-IL-LABEL: void @foo(i32 %x) !dbg ![[#]] {
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1), !dbg ![[#FAKELINE:]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID:]], 1, 0, 0
; CHECK-ASM: .pseudoprobe	[[#GUID:]] 1 0 0 foo
  br i1 %cmp, label %bb1, label %bb2

bb1:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 0, i64 -1), !dbg ![[#FAKELINE]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 3, 0, 0
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 4, 0, 0
; CHECK-ASM: .pseudoprobe	[[#GUID]] 3 0 0 foo
; CHECK-ASM: .pseudoprobe	[[#GUID]] 4 0 0 foo
  store i32 6, ptr @a, align 4
  br label %bb3

bb2:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 3, i32 0, i64 -1), !dbg ![[#FAKELINE]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 2, 0, 0
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 4, 0, 0
; CHECK-ASM: .pseudoprobe	[[#GUID]] 2 0 0 foo
; CHECK-ASM: .pseudoprobe	[[#GUID]] 4 0 0 foo
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
; CHECK-ASM: .pseudoprobe	[[#GUID2:]] 1 0 0 foo2
; Check pseudo_probe metadata attached to the indirect call instruction.
; CHECK-IL: call void %f(i32 1), !dbg ![[#PROBE0:]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID2]], 2, 1, 0
; CHECK-ASM: .pseudoprobe	[[#GUID2]] 2 1 0 foo2
  call void %f(i32 1), !dbg !13
; Check pseudo_probe metadata attached to the direct call instruction.
; CHECK-IL: call void @bar(i32 1), !dbg ![[#PROBE1:]]
; CHECK-MIR: PSEUDO_PROBE	[[#GUID2]], 3, 2, 0
; CHECK-ASM: .pseudoprobe	[[#GUID2]] 3 2 0 foo2
  call void @bar(i32 1)
  ret void
}

$foo3 = comdat any

define void @foo3(i32 %x) comdat {
entry:
  ret void
}

; CHECK-IL: Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
; CHECK-IL-NEXT: declare void @llvm.pseudoprobe(i64, i64, i32, i64)

; CHECK-IL: ![[#FOO:]] = distinct !DISubprogram(name: "foo"
; CHECK-IL: ![[#FAKELINE]] = !DILocation(line: 0, scope: ![[#FOO]])
; CHECK-IL: ![[#REALLINE]] = !DILocation(line: 2, scope: ![[#DISC0:]])
; CHECK-IL: ![[#DISC0]] = !DILexicalBlockFile(scope: ![[#FOO]], file: ![[#]], discriminator: 0)
; CHECK-IL: ![[#PROBE0]] = !DILocation(line: 2, column: 20, scope: ![[#SCOPE0:]])
;; A discriminator of 67108887 which is 0x7200017 in hexdecimal, stands for a direct call probe
;; with an index of 2.
; CHECK-IL: ![[#SCOPE0]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 119537687)
; CHECK-IL: ![[#PROBE1]] = !DILocation(line: 0, scope: ![[#SCOPE1:]])
;; A discriminator of 186646559 which is 0xb20001f in hexdecimal, stands for a direct call probe
;; with an index of 3.
; CHECK-IL: ![[#SCOPE1]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 186646559)

; Check the generation of .pseudo_probe_desc section
; CHECK-ASM: .section .pseudo_probe_desc,"G",@progbits,.pseudo_probe_desc_foo,comdat
; CHECK-ASM-NEXT: .quad [[#GUID]]
; CHECK-ASM-NEXT: .quad [[#HASH:]]
; CHECK-ASM-NEXT: .byte  3
; CHECK-ASM-NEXT: .ascii	"foo"
; CHECK-ASM-NEXT: .section  .pseudo_probe_desc,"G",@progbits,.pseudo_probe_desc_foo2,comdat
; CHECK-ASM-NEXT: .quad [[#GUID2]]
; CHECK-ASM-NEXT: .quad [[#HASH2:]]
; CHECK-ASM-NEXT: .byte 4
; CHECK-ASM-NEXT: .ascii	"foo2"

; CHECK-SEC:       [Nr] Name               Type     {{.*}} ES Flg Lk Inf Al
; CHECK-SEC:       [ 3] .text.foo          PROGBITS {{.*}} 00  AX  0   0 16
; CHECK-SEC:       [ 5] .text.foo2         PROGBITS {{.*}} 00  AX  0   0 16
; CHECK-SEC:       [ 8] .text.foo3         PROGBITS {{.*}} 00  AXG 0   0 16
; CHECK-SEC-COUNT-3:    .pseudo_probe_desc PROGBITS
; CHECK-SEC:            .pseudo_probe      PROGBITS {{.*}} 00   LG 8   0  1
; CHECK-SEC-NEXT:       .pseudo_probe      PROGBITS {{.*}} 00   L  5   0  1
; CHECK-SEC-NEXT:       .pseudo_probe      PROGBITS {{.*}} 00   L  3   0  1
; CHECK-SEC-NOT:   .rela.pseudo_probe

; CHECK-SEC:       COMDAT group section [    7] `.group' [foo3] contains 2 sections:
; CHECK-SEC-NEXT:     [Index]    Name
; CHECK-SEC-NEXT:     [    8]   .text.foo3
; CHECK-SEC-NEXT:     [   19]   .pseudo_probe
; CHECK-SEC-EMPTY:
; CHECK-SEC-NEXT:  COMDAT group section [   10] `.group' [.pseudo_probe_desc_foo] contains 1 sections:
; CHECK-SEC-NEXT:     [Index]    Name
; CHECK-SEC-NEXT:     [   11]   .pseudo_probe_desc
; CHECK-SEC-EMPTY:
; CHECK-SEC-NEXT:  COMDAT group section [   12] `.group' [.pseudo_probe_desc_foo2] contains 1 sections:
; CHECK-SEC-NEXT:     [Index]    Name
; CHECK-SEC-NEXT:     [   13]   .pseudo_probe_desc
; CHECK-SEC-EMPTY:
; CHECK-SEC-NEXT:  COMDAT group section [   14] `.group' [.pseudo_probe_desc_foo3] contains 1 sections:
; CHECK-SEC-NEXT:     [Index]    Name
; CHECK-SEC-NEXT:     [   15]   .pseudo_probe_desc


; CHECK-SEC2:      [Nr] Name               Type     {{.*}} ES Flg Lk Inf Al
; CHECK-SEC2:      [ 3] .text              PROGBITS {{.*}} 00  AX  0   0 16
; CHECK-SEC2:      [ 5] .text              PROGBITS {{.*}} 00  AX  0   0 16
; CHECK-SEC2:      [ 8] .text              PROGBITS {{.*}} 00  AXG 0   0 16
; CHECK-SEC2-COUNT-3:   .pseudo_probe_desc PROGBITS
; CHECK-SEC2:           .pseudo_probe      PROGBITS {{.*}} 00   LG 8   0  1
; CHECK-SEC2-NEXT:      .pseudo_probe      PROGBITS {{.*}} 00   L  5   0  1
; CHECK-SEC2-NEXT:      .pseudo_probe      PROGBITS {{.*}} 00   L  3   0  1
; CHECK-SEC2-NOT:  .rela.pseudo_probe

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
