; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELAX %s
; RUN: llvm-dwarfdump --debug-frame %t.o 2>&1 \
; RUN:     | FileCheck -check-prefix=RELAX-DWARFDUMP %s
;
; RELAX:      Section ({{.*}}) .rela.eh_frame {
; RELAX-NEXT:   0x1C R_RISCV_32_PCREL <null> 0x0
; RELAX-NEXT:   0x30 R_RISCV_32_PCREL <null> 0x0
; RELAX-NEXT:   0x44 R_RISCV_32_PCREL <null> 0x0
; RELAX-NEXT:   0x48 R_RISCV_ADD32 <null> 0x0
; RELAX-NEXT:   0x48 R_RISCV_SUB32 <null> 0x0
; RELAX-NEXT:  }

; RELAX-DWARFDUMP-NOT: error: failed to compute relocation
; RELAX-DWARFDUMP:      FDE
; RELAX-DWARFDUMP-NEXT: Format:
; RELAX-DWARFDUMP:      DW_CFA_advance_loc: 4
; RELAX-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset: +16
; RELAX-DWARFDUMP-EMPTY:

; RELAX-DWARFDUMP:      FDE
; RELAX-DWARFDUMP:      Format:
; RELAX-DWARFDUMP-NEXT: DW_CFA_advance_loc: 4
; RELAX-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset: +16
; RELAX-DWARFDUMP-NEXT: DW_CFA_advance_loc: 4
; RELAX-DWARFDUMP-NEXT: DW_CFA_offset: X1 -4
; RELAX-DWARFDUMP-NEXT: DW_CFA_nop
; RELAX-DWARFDUMP-EMPTY:
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
  store i32 %value, ptr %value.addr, align 4
  %0 = load i32, ptr %value.addr, align 4
  ret i32 %0
}

; Function Attrs: noinline nounwind optnone
define i32 @bar() {
entry:
  %result = alloca i32, align 4
  %v = alloca i32, align 4
  %call = call i32 @init()
  store i32 %call, ptr %v, align 4
  %0 = load i32, ptr %v, align 4
  %call1 = call i32 @foo(i32 signext %0)
  store i32 %call1, ptr %result, align 4
  %1 = load i32, ptr %result, align 4
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
