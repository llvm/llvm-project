; Test that the arrays of an unnamed function are not put in a comdat named
; after it (there is no name to use), and are retained through llvm.used.
; An unnamed function that is already in a comdat keeps using it.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-8bit-counters -sanitizer-coverage-pc-table -mtriple x86_64-linux-gnu -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-8bit-counters -sanitizer-coverage-pc-table -mtriple x86_64-windows-msvc -S | FileCheck %s

$Comdat = comdat any

define void @Named() {
entry:
  ret void
}

define private void @0() {
entry:
  ret void
}

define private void @1() comdat($Comdat) {
entry:
  ret void
}

; CHECK-NOT:  $ = comdat
; CHECK:      @__sancov_gen_ = private global [1 x i8] zeroinitializer, section "{{[^"]*}}", comdat($Named), align 1{{$}}
; CHECK-NEXT: @__sancov_gen_.1 = private constant [2 x ptr] [ptr @Named, ptr inttoptr (i64 1 to ptr)], section "{{[^"]*}}", comdat($Named), align 8{{$}}
; CHECK-NEXT: @__sancov_gen_.2 = private global [1 x i8] zeroinitializer, section "{{[^"]*}}", align 1{{$}}
; CHECK-NEXT: @__sancov_gen_.3 = private constant [2 x ptr] [ptr @0, ptr inttoptr (i64 1 to ptr)], section "{{[^"]*}}", align 8{{$}}
; CHECK-NEXT: @__sancov_gen_.4 = private global [1 x i8] zeroinitializer, section "{{[^"]*}}", comdat($Comdat), align 1{{$}}
; CHECK-NEXT: @__sancov_gen_.5 = private constant [2 x ptr] [ptr @1, ptr inttoptr (i64 1 to ptr)], section "{{[^"]*}}", comdat($Comdat), align 8{{$}}
; CHECK:      @llvm.used = appending global [{{[0-9]+}} x ptr] [{{.*}}ptr @__sancov_gen_.2, ptr @__sancov_gen_.3{{.*}}], section "llvm.metadata"
; CHECK:      @llvm.compiler.used = appending global [4 x ptr] [ptr @__sancov_gen_, ptr @__sancov_gen_.1, ptr @__sancov_gen_.4, ptr @__sancov_gen_.5], section "llvm.metadata"

; CHECK: define void @Named() comdat {
; CHECK: define private void @0() {
; CHECK: define private void @1() comdat($Comdat) {
