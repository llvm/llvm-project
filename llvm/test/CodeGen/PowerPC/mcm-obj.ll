; RUN: llc -relocation-model=static -verify-machineinstrs -O0 -mcpu=pwr7 -code-model=medium -filetype=obj -fast-isel=false %s -o - | \
; RUN: llvm-readobj -r - | FileCheck -check-prefix=MEDIUM %s
; RUN: llc -relocation-model=static -verify-machineinstrs -O0 -mcpu=pwr7 -code-model=large -filetype=obj -fast-isel=false %s -o - | \
; RUN: llvm-readobj -r - | FileCheck -check-prefix=LARGE %s

; Run jump table test separately since jump tables aren't generated at -O0.
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -code-model=medium -filetype=obj -fast-isel=false %s -o - | \
; RUN: llvm-readobj -r - | FileCheck -check-prefix=MEDIUM-JT %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -code-model=large -filetype=obj -fast-isel=false %s -o - | \
; RUN: llvm-readobj -r - | FileCheck -check-prefix=LARGE-JT %s

; FIXME: When asm-parse is available, could make this an assembly test.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@ei = external global i32

define dso_local signext i32 @test_external() nounwind {
entry:
  %0 = load i32, i32* @ei, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @ei, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing external variable ei.
;
; MEDIUM:      Relocations [
; MEDIUM:        Section {{.*}} .rela.text {
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA .toc 0x0
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS .toc 0x0
;
; LARGE:       Relocations [
; LARGE:         Section {{.*}} .rela.text {
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM1:[^ ]+]]
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM1]]

@test_fn_static.si = internal global i32 0, align 4

define dso_local signext i32 @test_fn_static() nounwind {
entry:
  %0 = load i32, i32* @test_fn_static.si, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @test_fn_static.si, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing function-scoped variable si.
;
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA .bss 0x0
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO .bss 0x0
;
; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing function-scoped variable si.
;
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM2:[^ ]+]]
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM2]]

@gi = dso_local global i32 5, align 4

define dso_local signext i32 @test_file_static() nounwind {
entry:
  %0 = load i32, i32* @gi, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @gi, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing file-scope variable gi.
;
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA gi 0x0
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO gi 0x0
;
; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing file-scope variable gi.
;
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM3:[^ ]+]]
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM3]]

define dso_local double @test_double_const() nounwind {
entry:
  ret double 0x3F4FD4920B498CF0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing a constant.
;
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA .rodata.cst8
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO .rodata.cst8
;
; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing a constant.
;
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM4:[^ ]+]]
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM4]]

@ti = common dso_local global i32 0, align 4

define dso_local signext i32 @test_tentative() nounwind {
entry:
  %0 = load i32, i32* @ti, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @ti, align 4
  ret i32 %0
}

; Verify generation of relocations foraccessing variable ti.
;
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA ti 0x0
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO ti 0x0
;
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM6:[^ ]+]]
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM6]]

define i8* @test_fnaddr() nounwind {
entry:
  %func = alloca i32 (i32)*, align 8
  store i32 (i32)* @foo, i32 (i32)** %func, align 8
  %0 = load i32 (i32)*, i32 (i32)** %func, align 8
  %1 = bitcast i32 (i32)* %0 to i8*
  ret i8* %1
}

declare signext i32 @foo(i32 signext)

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing function address foo.
;
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA .toc 0x8
; MEDIUM-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS .toc 0x8
;
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM7:[^ ]+]]
; LARGE-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM7]]


define dso_local signext i32 @test_jump_table(i32 signext %i) nounwind {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  switch i32 %0, label %sw.default [
    i32 3, label %sw.bb
    i32 4, label %sw.bb1
    i32 5, label %sw.bb2
    i32 6, label %sw.bb3
  ]

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.bb:                                            ; preds = %entry
  %1 = load i32, i32* %i.addr, align 4
  %mul = mul nsw i32 %1, 7
  store i32 %mul, i32* %i.addr, align 4
  br label %sw.bb1

sw.bb1:                                           ; preds = %entry, %sw.bb
  %2 = load i32, i32* %i.addr, align 4
  %dec = add nsw i32 %2, -1
  store i32 %dec, i32* %i.addr, align 4
  br label %sw.bb2

sw.bb2:                                           ; preds = %entry, %sw.bb1
  %3 = load i32, i32* %i.addr, align 4
  %add = add nsw i32 %3, 3
  store i32 %add, i32* %i.addr, align 4
  br label %sw.bb3

sw.bb3:                                           ; preds = %entry, %sw.bb2
  %4 = load i32, i32* %i.addr, align 4
  %shl = shl i32 %4, 1
  store i32 %shl, i32* %i.addr, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb3, %sw.default
  %5 = load i32, i32* %i.addr, align 4
  ret i32 %5
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing a jump table address.
;
; MEDIUM-JT:      Relocations [
; MEDIUM-JT:        Section ({{.*}}) .rela.text {
; MEDIUM-JT-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM:[^ ]+]]
; MEDIUM-JT-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM]]
;
; LARGE-JT:       Relocations [
; LARGE-JT:         Section ({{.*}}) .rela.text {
; LARGE-JT-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_HA [[SYM:[^ ]+]]
; LARGE-JT-NEXT:      0x{{[0-9,A-F]+}} R_PPC64_TOC16_LO_DS [[SYM]]
