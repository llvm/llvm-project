; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o - \
; RUN:     | llvm-readobj -r - | FileCheck %s
; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=-relax %s -o - \
; RUN:     | llvm-readobj -r - | FileCheck %s

;; b-a in the inline asm is unaffected by linker relaxation, so we don't emit
;; relocations.

;; b-a in the inline asm and DW_AT_high_pc are unaffected by linker relaxation,
;; so we don't emit ADD/SUB relocations.
;; See also MC/RISCV/fixups-expr.s.

source_filename = "tmp.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32"

define i32 @main() !dbg !7 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void asm sideeffect "a:\0Ab:\0A.dword b-a", ""()
  ret i32 0
}

; CHECK:      Section {{.*}} .rela.debug_info {
; CHECK-NOT:         R_RISCV_ADD32
; CHECK:      }

; CHECK:      Section {{.*}} .rela.eh_frame {
; CHECK-NEXT:   0x1C R_RISCV_32_PCREL <null> 0x0
; CHECK-NEXT: }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "fixups-diff.ll", directory: "test/MC/RISCV")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 3, scope: !7)
