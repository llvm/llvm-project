; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff  < %s | \
; RUN:   FileCheck --check-prefix=ASM %s

@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_main = private global { i64, i64, i64, ptr, ptr, i32, [4 x i16] } { i64 -2624081020897602054, i64 742261418966908927, i64 sub (i64 ptrtoint (ptr @__profc_main to i64), i64 ptrtoint (ptr @__profd_main to i64)), ptr @main, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8

; Test fallback of using sub expr for lowerRelativeReference
define signext i32 @main() {
; ASM-LABEL: main:
; ASM:  L..__profd_main:
; ASM:        .vbyte  8, L..__profc_main-L..__profd_main
entry:
  %pgocount = load i64, ptr @__profc_main, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_main, align 8
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}

