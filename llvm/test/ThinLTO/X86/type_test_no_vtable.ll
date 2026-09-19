; Regression tests for https://github.com/llvm/llvm-project/issues/187049.
;
; Test that WPD removes type test assumes that are merged through a select or
; phi when there are no corresponding vtables in the ThinLTO index. If an
; assume is left behind, LowerTypeTests resolves the type tests to false,
; producing assume(false) and incorrectly making the virtual call unreachable.
;
; REQUIRES: x86-registered-target
;
; RUN: opt -thinlto-bc -thinlto-split-lto-unit=false -o %t.o %s
; RUN: llvm-lto2 run %t.o -save-temps -whole-program-visibility \
; RUN:   -r=%t.o,test_select,plx \
; RUN:   -r=%t.o,test_select_one_missing,plx \
; RUN:   -r=%t.o,test_select_chain,plx \
; RUN:   -r=%t.o,test_phi,plx \
; RUN:   -r=%t.o,seed_known,plx \
; RUN:   -r=%t.o,get_base, \
; RUN:   -r=%t.o,get_derived, \
; RUN:   -r=%t.o,observe, \
; RUN:   -r=%t.o,puts, \
; RUN:   -o %t2
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@message = private unnamed_addr constant [6 x i8] c"hello\00"
@known_vtable = internal constant [1 x ptr] [ptr @known_target], !type !0
@llvm.compiler.used = appending global [1 x ptr] [ptr @known_vtable],
    section "llvm.metadata"

define void @test_select(ptr %object, i1 %derived) {
entry:
  %vtable = load ptr, ptr %object, align 8
  %base.test = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS4Base")
  %derived.test = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS7Derived")
  %type.test = select i1 %derived, i1 %derived.test, i1 %base.test
  call void @llvm.assume(i1 %type.test)
  %target = load ptr, ptr %vtable, align 8
  call void %target(ptr %object)
  call i32 @puts(ptr @message)
  ret void
}

; CHECK-LABEL: @test_select(
; CHECK-NOT: @llvm.type.test
; CHECK-NOT: @llvm.assume
; CHECK: call void %{{.*}}(ptr {{.*}}%object)
; CHECK: call i32 @puts(ptr {{.*}})
; CHECK: ret void

; Removing the assumption is necessary if even one of its type tests has no
; type information. The other type test has a vtable entry in this module.
define void @test_select_one_missing(ptr %object, i1 %use.missing) {
entry:
  %vtable = load ptr, ptr %object, align 8
  %known.test = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS5Known")
  %missing.test = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS7Missing")
  %type.test = select i1 %use.missing, i1 %missing.test, i1 %known.test
  call void @llvm.assume(i1 %type.test)
  %target = load ptr, ptr %vtable, align 8
  call void %target(ptr %object)
  call i32 @puts(ptr @message)
  ret void
}

; CHECK-LABEL: @test_select_one_missing(
; CHECK-NOT: @llvm.type.test
; CHECK-NOT: @llvm.assume
; CHECK: call void %{{.*}}(ptr {{.*}}%object)
; CHECK: call i32 @puts(ptr {{.*}})
; CHECK: ret void

; An if/else-if/else chain can produce nested selects when its values are
; available in the entry block. The outer type test has a valid summary, while
; the two tests nested inside the inner select do not. The latter must still
; cause the dependent assume to be removed.
define void @test_select_chain(ptr %object, i1 %first, i1 %second) {
entry:
  %vtable = load ptr, ptr %object, align 8
  %known.test = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS5Known")
  %missing.0 = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS8Missing0")
  %missing.1 = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS8Missing1")
  %inner = select i1 %second, i1 %missing.0, i1 %missing.1
  %type.test = select i1 %first, i1 %known.test, i1 %inner
  call void @llvm.assume(i1 %type.test)
  %target = load ptr, ptr %vtable, align 8
  call void %target(ptr %object)
  call void @observe(i1 %first)
  call i32 @puts(ptr @message)
  ret void
}

; CHECK-LABEL: @test_select_chain(
; CHECK-NOT: @llvm.type.test
; CHECK-NOT: @llvm.assume
; CHECK: call void %{{.*}}(ptr {{.*}}%object)
; CHECK: call void @observe(i1 %first)
; CHECK: call i32 @puts(ptr {{.*}})
; CHECK: ret void

; Separate dispatch arms can be merged into a phi instead of a select.
define void @test_phi(i1 %derived) {
entry:
  br i1 %derived, label %derived.block, label %base.block

base.block:
  %base.object = call ptr @get_base()
  %base.vtable = load ptr, ptr %base.object, align 8
  %base.test = call i1 @llvm.public.type.test(
      ptr %base.vtable, metadata !"_ZTS4Base")
  br label %merge

derived.block:
  %derived.object = call ptr @get_derived()
  %derived.vtable = load ptr, ptr %derived.object, align 8
  %derived.test = call i1 @llvm.public.type.test(
      ptr %derived.vtable, metadata !"_ZTS7Derived")
  br label %merge

merge:
  %object = phi ptr [ %base.object, %base.block ],
                    [ %derived.object, %derived.block ]
  %vtable = phi ptr [ %base.vtable, %base.block ],
                    [ %derived.vtable, %derived.block ]
  %type.test = phi i1 [ %base.test, %base.block ],
                      [ %derived.test, %derived.block ]
  call void @llvm.assume(i1 %type.test)
  %target = load ptr, ptr %vtable, align 8
  call void %target(ptr %object)
  call i32 @puts(ptr @message)
  ret void
}

; CHECK-LABEL: @test_phi(
; CHECK-NOT: @llvm.type.test
; CHECK-NOT: @llvm.assume
; CHECK: call void %{{.*}}(ptr {{.*}})
; CHECK: call i32 @puts(ptr {{.*}})
; CHECK: ret void

declare i1 @llvm.public.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare i32 @puts(ptr)
declare ptr @get_base()
declare ptr @get_derived()
declare void @observe(i1)

; Ensure that _ZTS5Known gets a TypeIdSummary, so only the nested leaves are
; responsible for removing the assumption in test_select_chain.
define void @seed_known(ptr %object) {
  %vtable = load ptr, ptr %object, align 8
  %type.test = call i1 @llvm.public.type.test(
      ptr %vtable, metadata !"_ZTS5Known")
  call void @llvm.assume(i1 %type.test)
  %target = load ptr, ptr %vtable, align 8
  call void %target(ptr %object)
  ret void
}

define internal void @known_target(ptr %object) {
  ret void
}

!0 = !{i64 0, !"_ZTS5Known"}
