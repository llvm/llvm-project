;; Test section-relative address syntax for XCOFF
;; The syntax (SECTION_TYPE)(+offset) represents: offset from section base

; REQUIRES: system-aix
; RUN: llc -filetype=obj -o %t -mtriple=powerpc-aix-ibm-xcoff -function-sections < %s

;; Test 1: Symbolize .foo using section-relative offset
; RUN: llvm-nm --numeric-sort %t | grep " T \.foo$" | awk '{printf "CODE (TEXT)(+0x%%s)", $1}' > %t.foo_query
; RUN: llvm-symbolizer --obj=%t @%t.foo_query | FileCheck %s --check-prefix=TEST-FOO

;; Test 2: Symbolize .bar using section-relative offset
; RUN: llvm-nm --numeric-sort %t | grep " T \.bar$" | awk '{printf "CODE (TEXT)(+0x%%s)", $1}' > %t.bar_query
; RUN: llvm-symbolizer --obj=%t @%t.bar_query | FileCheck %s --check-prefix=TEST-BAR

;; Test 3: Symbolize global_var using section-relative offset in DATA section
; RUN: llvm-readobj --sections %t | awk '/Name: \.data/{found=1} found && /VirtualAddress:/{print $2; exit}' > %t.data_base
; RUN: llvm-nm --numeric-sort %t | grep " D global_var$" | awk '{print $1}' > %t.global_var_vma
; RUN: sh -c 'printf "%%d\n" $(cat %t.data_base)' > %t.data_base_dec
; RUN: sh -c 'printf "%%d\n" 0x$(cat %t.global_var_vma)' > %t.global_var_dec
; RUN: awk 'NR==FNR{base=$1; next} {vma=$1; printf "DATA (DATA)(+0x%%x)", vma-base}' %t.data_base_dec %t.global_var_dec > %t.data_query
; RUN: llvm-symbolizer --obj=%t @%t.data_query | FileCheck %s --check-prefix=TEST-DATA

;; Test 4: Verify section structure with llvm-readobj
; RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix=SECTIONS

define void @foo() {
entry:
  ret void
}

define void @bar() {
entry:
  ret void
}

@global_var = global i32 42, align 4

;; Verify correct symbolization with section-relative syntax
; TEST-FOO: .foo
; TEST-FOO-NEXT: ??:0:0

; TEST-BAR: .bar
; TEST-BAR-NEXT: ??:0:0

; TEST-DATA: global_var

;; Verify XCOFF sections exist with correct types
; SECTIONS: Name: .text
; SECTIONS: Type: STYP_TEXT
; SECTIONS: Name: .data
; SECTIONS: Type: STYP_DATA
