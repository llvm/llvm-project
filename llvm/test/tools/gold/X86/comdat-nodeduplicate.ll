;; Keep __profd_foo in a nodeduplicate comdat, despite a comdat of the same name
;; in a previous object file.

;; Regular LTO

; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-as %t/b.ll -o %t/b.bc
; RUN: llvm-as %t/c.ll -o %t/c.bc

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext --plugin-opt=save-temps \
; RUN:   -u foo %t/a.bc --start-lib %t/b.bc --end-lib -o %t/ab

; RUN: FileCheck %s --check-prefix=RESOL_AB < %t/ab.resolution.txt
; RUN: llvm-readelf -x .data %t/ab | FileCheck %s --check-prefix=DATA

; RESOL_AB: -r={{.*}}b.bc,__profc_foo,pl{{$}}

;; .data contains a.bc:data. b.bc:data and c.bc:data are discarded.
; DATA: 0x[[#%x,]] 01000000 00000000  ........

;; __profc_foo from c.bc is non-prevailing and thus discarded.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext --plugin-opt=save-temps \
; RUN:   -u foo -u c %t/a.bc --start-lib %t/b.bc %t/c.bc --end-lib -o %t/abc
; RUN: FileCheck %s --check-prefix=RESOL_ABC < %t/abc.resolution.txt
; RUN: llvm-readelf -x .data %t/abc | FileCheck %s --check-prefix=DATA

; RESOL_ABC: -r={{.*}}b.bc,__profc_foo,pl{{$}}
; RESOL_ABC: -r={{.*}}c.bc,__profc_foo,l{{$}}

;; ThinLTO

; RUN: rm -rf %t && split-file %s %t
; RUN: opt --module-summary %t/a.ll -o %t/a.bc
; RUN: opt --module-summary %t/b.ll -o %t/b.bc
; RUN: opt --module-summary %t/c.ll -o %t/c.bc

;; gold from binutils>=2.27 discards b.bc:__profc_foo even in the absence of --gc-sections.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   -u foo %t/a.bc --start-lib %t/b.bc %t/c.bc --end-lib -o %t/abc --gc-sections
; RUN: llvm-readelf -x .data %t/abc | FileCheck %s --check-prefix=DATA

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__profc_foo = comdat nodeduplicate
@__profc_foo = private global i64 1, comdat, align 8
@__profd_foo = private global i64* @__profc_foo, comdat($__profc_foo), align 8

declare void @b()

define i64 @foo() {
  %v = load i64, i64* @__profc_foo
  %inc = add i64 1, %v
  store i64 %inc, i64* @__profc_foo
  ret i64 %inc
}

define void @_start() {
  call void @b()
  ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__profc_foo = comdat nodeduplicate
@__profc_foo = weak hidden global i64 2, comdat, align 8
@__profd_foo = private global i64* @__profc_foo, comdat($__profc_foo)

define weak i64 @foo() {
  %v = load i64, i64* @__profc_foo
  %inc = add i64 1, %v
  store i64 %inc, i64* @__profc_foo
  ret i64 %inc
}

define void @b() {
  ret void
}

;--- c.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__profc_foo = comdat nodeduplicate
@__profc_foo = weak hidden global i64 3, comdat, align 8
@__profd_foo = private global i64* @__profc_foo, comdat($__profc_foo)

define weak i64 @foo() {
  %v = load i64, i64* @__profc_foo
  %inc = add i64 1, %v
  store i64 %inc, i64* @__profc_foo
  ret i64 %inc
}

define void @c() {
  ret void
}
