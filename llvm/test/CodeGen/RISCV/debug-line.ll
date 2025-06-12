; RUN: llc -mtriple=riscv64 < %s | FileCheck %s

define void @foo() #0 !dbg !3 {
; CHECK-LABEL: foo:
; CHECK: .Lfunc_begin0:
; CHECK-NEXT: 	.file	1 "test.c"
; CHECK-NEXT: 	.loc	1 5 0                           # test.c:5:0
; CHECK-NEXT: 	.cfi_startproc
; CHECK-NEXT: # %bb.0:                                # %entry
; CHECK-NEXT: 	addi	sp, sp, -16
; CHECK-NEXT: 	.cfi_def_cfa_offset 16
; CHECK-NEXT: 	sd	ra, 8(sp)                       # 8-byte Folded Spill
; CHECK-NEXT: 	sd	s0, 0(sp)                       # 8-byte Folded Spill
; CHECK-NEXT: 	.cfi_offset ra, -8
; CHECK-NEXT: 	.cfi_offset s0, -16
; CHECK-NEXT: 	addi	s0, sp, 16
; CHECK-NEXT: 	.cfi_def_cfa s0, 0
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT: 	.loc	1 6 4 prologue_end              # test.c:6:4
; CHECK-NEXT: 	sw	zero, 0(zero)
; CHECK-NEXT:   .cfi_def_cfa sp, 16
; CHECK-NEXT: 	.loc	1 7 1 epilogue_begin            # test.c:7:1
; CHECK-NEXT: 	ld	ra, 8(sp)                       # 8-byte Folded Reload
; CHECK-NEXT: 	ld	s0, 0(sp)                       # 8-byte Folded Reload
; CHECK-NEXT:   .cfi_restore ra
; CHECK-NEXT:   .cfi_restore s0
; CHECK-NEXT: 	addi	sp, sp, 16
; CHECK-NEXT:   .cfi_def_cfa_offset 0 
; CHECK-NEXT: 	ret
entry:
  store i32 0, ptr null, align 4, !dbg !6
  ret void, !dbg !7
}

attributes #0 = { "frame-pointer"="all" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DILocation(line: 6, column: 4, scope: !3)
!7 = !DILocation(line: 7, column: 1, scope: !3)
