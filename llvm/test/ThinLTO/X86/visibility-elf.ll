; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %ta.bc
; RUN: opt -module-summary %t/b.ll -o %tb.bc

;; Test visibility propagation. The prevailing definitions are all from %tb.bc.
; RUN: llvm-lto2 run -save-temps -enable-import-metadata -o %t1.bc %ta.bc %tb.bc \
; RUN:   -r=%ta.bc,var1,l -r=%ta.bc,var2,l \
; RUN:   -r=%ta.bc,hidden_def_weak_def,l -r=%ta.bc,protected_def_weak_def,l -r=%ta.bc,protected_def_weak_hidden_def,l \
; RUN:   -r=%ta.bc,protected_def_hidden_ref,l -r=%ta.bc,not_imported,l -r=%ta.bc,hidden_def_ref,l \
; RUN:   -r=%ta.bc,hidden_def_weak_ref,l \
; RUN:   -r=%ta.bc,ext, -r=%ta.bc,main,plx \
; RUN:   -r=%tb.bc,var1,plx -r=%tb.bc,var2,plx \
; RUN:   -r=%tb.bc,hidden_def_weak_def,pl -r=%tb.bc,protected_def_weak_def,pl -r=%tb.bc,protected_def_weak_hidden_def,pl \
; RUN:   -r=%tb.bc,protected_def_hidden_ref,pl -r=%tb.bc,not_imported,pl -r=%tb.bc,hidden_def_ref,pl \
; RUN:   -r=%tb.bc,hidden_def_weak_ref,pl
; RUN: llvm-dis < %t1.bc.1.3.import.bc | FileCheck %s

;; %tb.bc does not import anything, so we just check 1.promote.
; RUN: llvm-dis < %t1.bc.2.1.promote.bc | FileCheck %s --check-prefix=CHECK2

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK:      @var1 = external hidden global i32, align 4
; CHECK-NEXT: @var2 = available_externally hidden global i32 1, align 4

@var1 = weak global i32 1, align 4
@var2 = extern_weak global i32

declare void @ext(ptr)

; CHECK: declare hidden i32 @hidden_def_weak_def()
; CHECK: declare protected void @protected_def_weak_def()
; CHECK: declare hidden void @protected_def_weak_hidden_def()
;; Currently the visibility is not propagated onto an unimported function,
;; because we don't have summaries for declarations.
; CHECK: declare extern_weak void @not_imported()
; CHECK: define available_externally hidden void @hidden_def_ref() !thinlto_src_module !0
; CHECK: define available_externally hidden void @hidden_def_weak_ref() !thinlto_src_module !0
;; This can be hidden, but we cannot communicate the declaration's visibility
;; to other modules because declarations don't have summaries, and the IRLinker
;; overrides it when importing the protected def.
; CHECK: define available_externally protected void @protected_def_hidden_ref() !thinlto_src_module !0

; CHECK2: define hidden i32 @hidden_def_weak_def()
; CHECK2: define protected void @protected_def_weak_def()
; CHECK2: define hidden void @protected_def_weak_hidden_def()
; CHECK2: define hidden void @hidden_def_ref()
; CHECK2: define hidden void @hidden_def_weak_ref()
; CHECK2: define protected void @protected_def_hidden_ref()
; CHECK2: define hidden void @not_imported()

define weak i32 @hidden_def_weak_def() {
entry:
  %0 = load i32, ptr @var2
  ret i32 %0
}

define weak void @protected_def_weak_def() {
entry:
  ret void
}

define weak hidden void @protected_def_weak_hidden_def() {
entry:
  ret void
}

declare extern_weak void @not_imported()

declare void @hidden_def_ref()
declare extern_weak void @hidden_def_weak_ref()
declare hidden void @protected_def_hidden_ref()

define i32 @main() {
entry:
  call void @ext(ptr @hidden_def_weak_def)
  call void @ext(ptr @protected_def_weak_def)
  call void @ext(ptr @protected_def_weak_hidden_def)
  call void @ext(ptr @hidden_def_ref)
  call void @ext(ptr @hidden_def_weak_ref)
  call void @ext(ptr @protected_def_hidden_ref)
  call void @ext(ptr @not_imported)

  ;; Calls ensure the functions are imported.
  call i32 @hidden_def_weak_def()
  call void @protected_def_weak_def()
  call void @protected_def_weak_hidden_def()
  call void @hidden_def_ref()
  call void @hidden_def_weak_ref()
  call void @protected_def_hidden_ref()
  ret i32 0
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@var1 = hidden global i32 1, align 4
@var2 = hidden global i32 1, align 4

define hidden i32 @hidden_def_weak_def() {
entry:
  %0 = load i32, ptr @var1
  ret i32 %0
}

define protected void @protected_def_weak_def() {
entry:
  ret void
}

define protected void @protected_def_weak_hidden_def() {
entry:
  ret void
}

define hidden void @hidden_def_ref() {
entry:
  ret void
}

define hidden void @hidden_def_weak_ref() {
entry:
  ret void
}

define protected void @protected_def_hidden_ref() {
entry:
  ret void
}

define hidden void @not_imported() {
entry:
  ret void
}
