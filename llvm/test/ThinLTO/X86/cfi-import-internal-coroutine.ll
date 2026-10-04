; When ThinLTOBitcodeWriter promotes an internal function by creating an external
; alias, the alias must use the function's value type (FunctionType), not ptr.
; Otherwise, when an internal coroutine is imported into another module, IRMover
; sees a non-function value type for the unimported alias and materializes it as
; an external GlobalVariable rather than a Function declaration. That violates the
; verifier requirement that the coroutine argument of @llvm.coro.id must refer to
; a function.
;
; REQUIRES: x86-registered-target

; RUN: rm -rf %t.dir && split-file %s %t.dir
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t.dir/a.ll -o %t.dir/a.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t.dir/b.ll -o %t.dir/b.bc
; RUN: llvm-lto2 run -save-temps %t.dir/a.bc %t.dir/b.bc -o %t.dir/out \
; RUN:   -r=%t.dir/a.bc,get_vtable,plx \
; RUN:   -r=%t.dir/a.bc,vtable.b66cc3a748330e63b2451fa89522eb06,l \
; RUN:   -r=%t.dir/a.bc,coro.b66cc3a748330e63b2451fa89522eb06,plx \
; RUN:   -r=%t.dir/a.bc,vtable.b66cc3a748330e63b2451fa89522eb06,plx \
; RUN:   -r=%t.dir/a.bc,coro.b66cc3a748330e63b2451fa89522eb06,l \
; RUN:   -r=%t.dir/b.bc,caller,plx
; RUN: llvm-dis %t.dir/out.2.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT

; IMPORT: define available_externally hidden ptr @coro.llvm.{{[0-9]+}}()
; IMPORT-NEXT: %id = call token @llvm.coro.id(i32 8, ptr null, ptr nonnull @coro.b66cc3a748330e63b2451fa89522eb06, ptr null)
; IMPORT: declare ptr @coro.b66cc3a748330e63b2451fa89522eb06()

;--- a.ll
source_filename = "a.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vtable = internal constant [1 x ptr] [ptr @coro], !type !0

define internal ptr @coro() {
  %id = call token @llvm.coro.id(i32 8, ptr null, ptr nonnull @coro, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  ret ptr %hdl
}

define hidden ptr @get_vtable() {
  ret ptr @vtable
}

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)

!0 = !{i64 0, !"_ZTS1A"}

;--- b.ll
source_filename = "b.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Indirect call with VP metadata targeting the GUID of a.ll:coro (1497692708810309344)
define hidden ptr @caller(ptr %fp) {
  %r = call ptr %fp(), !prof !0
  ret ptr %r
}

!0 = !{!"VP", i32 0, i64 1, i64 1497692708810309344, i64 1}
