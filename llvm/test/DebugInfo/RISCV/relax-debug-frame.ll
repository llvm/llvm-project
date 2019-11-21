; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o - \
; RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX %s
; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o - \
; RUN:     | llvm-dwarfdump --debug-frame - 2>&1 \
; RUN:     | FileCheck -check-prefix=RELAX-DWARFDUMP %s
;
; RELAX: Section{{.*}}.rela.{{eh|debug}}_frame {
; RELAX-NOT: {{[}]}}
; RELAX-NOT: 0x0 R_RISCV_ADD32
; RELAX-NOT: 0x0 R_RISCV_SUB32
; RELAX-NOT: {{[}]}}
; RELAX: 0x20 R_RISCV_ADD32
; RELAX: 0x20 R_RISCV_SUB32
; RELAX-NOT: {{[}]}}
; RELAX: 0x39 R_RISCV_SET6
; RELAX: 0x39 R_RISCV_SUB6
;
; RELAX-DWARFDUMP-NOT: error: failed to compute relocation
; RELAX-DWARFDUMP: CIE
; RELAX-DWARFDUMP: DW_CFA_advance_loc
; RELAX-DWARFDUMP: DW_CFA_def_cfa_offset
; RELAX-DWARFDUMP: DW_CFA_offset
source_filename = "frame.c"

; Function Attrs: noinline nounwind optnone
define i32 @init() {
entry:
  ret i32 0
}

; Function Attrs: noinline nounwind optnone
define i32 @foo(i32 signext %value) {
entry:
  %value.addr = alloca i32, align 4
  store i32 %value, i32* %value.addr, align 4
  %0 = load i32, i32* %value.addr, align 4
  ret i32 %0
}

; Function Attrs: noinline nounwind optnone
define i32 @bar() {
entry:
  %result = alloca i32, align 4
  %v = alloca i32, align 4
  %call = call i32 @init()
  store i32 %call, i32* %v, align 4
  %0 = load i32, i32* %v, align 4
  %call1 = call i32 @foo(i32 signext %0)
  store i32 %call1, i32* %result, align 4
  %1 = load i32, i32* %result, align 4
  ret i32 %1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "line.c", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
