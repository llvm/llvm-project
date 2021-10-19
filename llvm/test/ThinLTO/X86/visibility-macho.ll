; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %ta.bc
; RUN: opt -module-summary %t/b.ll -o %tb.bc

;; Test visibility propagation. The prevailing definitions are all from %tb.bc.
; RUN: llvm-lto2 run -save-temps -enable-import-metadata -o %t1.bc %ta.bc %tb.bc \
; RUN:   -r=%ta.bc,_var1,l -r=%ta.bc,_var2,l \
; RUN:   -r=%ta.bc,_hidden_def_weak_def,l -r=%ta.bc,_not_imported,l -r=%ta.bc,_hidden_def_ref,l \
; RUN:   -r=%ta.bc,_hidden_def_weak_ref,l \
; RUN:   -r=%ta.bc,_ext, -r=%ta.bc,_main,plx \
; RUN:   -r=%tb.bc,_var1,plx -r=%tb.bc,_var2,plx \
; RUN:   -r=%tb.bc,_hidden_def_weak_def,pl -r=%tb.bc,_not_imported,pl -r=%tb.bc,_hidden_def_ref,pl \
; RUN:   -r=%tb.bc,_hidden_def_weak_ref,pl
; RUN: llvm-dis < %t1.bc.1.3.import.bc | FileCheck %s
; RUN: llvm-dis < %t1.bc.2.1.promote.bc | FileCheck %s --check-prefix=CHECK2

;--- a.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

; CHECK:      @var1 = external hidden global i32, align 4
; CHECK-NEXT: @var2 = available_externally hidden global i32 1, align 4

@var1 = weak global i32 1, align 4
@var2 = extern_weak global i32

declare void @ext(void ()*)

; CHECK: declare hidden i32 @hidden_def_weak_def()
;; Currently the visibility is not propagated onto an unimported function,
;; because we don't have summaries for declarations.
; CHECK: declare extern_weak dso_local void @not_imported()
; CHECK: define available_externally hidden void @hidden_def_ref() !thinlto_src_module !0
; CHECK: define available_externally hidden void @hidden_def_weak_ref() !thinlto_src_module !0

; CHECK2: define hidden i32 @hidden_def_weak_def()
; CHECK2: define hidden void @hidden_def_ref()
; CHECK2: define hidden void @hidden_def_weak_ref()
; CHECK2: define hidden void @not_imported()

define weak i32 @hidden_def_weak_def() {
entry:
  %0 = load i32, i32* @var2
  ret i32 %0
}

declare extern_weak void @not_imported()

declare void @hidden_def_ref()
declare extern_weak void @hidden_def_weak_ref()

define i32 @main() {
entry:
  call void @ext(void ()* bitcast (i32 ()* @hidden_def_weak_def to void ()*))
  call void @ext(void ()* @hidden_def_ref)
  call void @ext(void ()* @hidden_def_weak_ref)
  call void @ext(void ()* @not_imported)

  ;; Calls ensure the functions are imported.
  call i32 @hidden_def_weak_def()
  call void @hidden_def_ref()
  call void @hidden_def_weak_ref()
  ret i32 0
}

;--- b.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@var1 = hidden global i32 1, align 4
@var2 = hidden global i32 1, align 4

define hidden i32 @hidden_def_weak_def() {
entry:
  %0 = load i32, i32* @var1
  ret i32 %0
}

define hidden void @hidden_def_ref() {
entry:
  ret void
}

define hidden void @hidden_def_weak_ref() {
entry:
  ret void
}

define hidden void @not_imported() {
entry:
  ret void
}
