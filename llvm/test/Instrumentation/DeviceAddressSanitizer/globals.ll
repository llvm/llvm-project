; RUN: opt < %s -passes=dasan -S | FileCheck %s
; RUN: opt < %s -passes=dasan -S | FileCheck %s --check-prefix=ALIAS
; RUN: opt < %s -passes=dasan -dasan-instrument-globals=0 -S | FileCheck %s --check-prefix=OFF

; REQUIRES: amdgpu-registered-target
; RUN: opt < %s -passes=dasan -S | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - | FileCheck %s --check-prefix=ASM
; ASM: .section .sbss.dasan.globals,"awG",@nobits,odr,comdat
; ASM: .size init, 16
; ASM: .size __dasan_guard, 4032

; RUN: opt < %s -passes=dasan,globalopt -S | FileCheck %s --check-prefix=FOLD
; FOLD: @init = alias [4 x i32], ptr addrspace(1) @init.dasan
; FOLD: @literal = internal alias [4 x i8], ptr addrspace(4) @literal.dasan

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

@init = addrspace(1) global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
; CHECK: @init.dasan = private addrspace(1) global { [4 x i32], [48 x i8] } { [4 x i32] [i32 1, i32 2, i32 3, i32 4], [48 x i8] zeroinitializer }, section ".data.dasan.globals", align 64
; ALIAS: @init = alias [4 x i32], ptr addrspace(1) @init.dasan
; ALIAS-NOT: @__dasan_guard = internal alias {{.*}}, getelementptr (i8, ptr addrspace(1) @init.dasan
; OFF: @init = addrspace(1) global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
; OFF-NOT: @__dasan_guard

@zero = addrspace(1) global [4 x i32] zeroinitializer, align 4
; CHECK: @zero.dasan = private addrspace(1) global { [4 x i32], [48 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 64
; ALIAS: @zero = alias [4 x i32], ptr addrspace(1) @zero.dasan

@konst = addrspace(4) constant [2 x i32] [i32 7, i32 8], align 4
; CHECK: @konst.dasan = private addrspace(4) constant { [2 x i32], [56 x i8] } { [2 x i32] [i32 7, i32 8], [56 x i8] zeroinitializer }, section ".rodata.dasan.globals", align 64
; ALIAS: @konst = alias [2 x i32], ptr addrspace(4) @konst.dasan

@relro = addrspace(1) constant ptr addrspace(1) @init, align 8
; CHECK: @relro.dasan = private addrspace(1) constant { ptr addrspace(1), [56 x i8] } { ptr addrspace(1) @init, [56 x i8] zeroinitializer }, section ".data.rel.ro.dasan.globals", align 64

@literal = private unnamed_addr addrspace(4) constant [4 x i8] c"abc\00", align 1
; ALIAS: @literal = internal alias [4 x i8], ptr addrspace(4) @literal.dasan
; OFF: @literal = private unnamed_addr addrspace(4) constant [4 x i8] c"abc\00", align 1

@overaligned = addrspace(1) global [4 x i32] zeroinitializer, align 4096
; CHECK: @overaligned.dasan = private addrspace(1) global { [4 x i32], [4080 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 4096
; ALIAS: @overaligned = alias [4 x i32], ptr addrspace(1) @overaligned.dasan
; ALIAS-NEXT: @__dasan_guard = internal alias [4032 x i8], getelementptr (i8, ptr addrspace(1) @overaligned.dasan, i64 64)

@filled = addrspace(1) global [64 x i32] zeroinitializer, align 4
; CHECK: @filled.dasan = private addrspace(1) global { [64 x i32], [64 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 64
; ALIAS: @filled = alias [64 x i32], ptr addrspace(1) @filled.dasan
; ALIAS-NEXT: @__dasan_guard.1 = internal alias [64 x i8], getelementptr (i8, ptr addrspace(1) @filled.dasan, i64 256)

@wide = addrspace(1) global [256 x i32] zeroinitializer, align 4
; CHECK: @wide.dasan = private addrspace(1) global { [256 x i32], [256 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 64
; ALIAS: @wide = alias [256 x i32], ptr addrspace(1) @wide.dasan
; ALIAS-NEXT: @__dasan_guard.2 = internal alias [256 x i8], getelementptr (i8, ptr addrspace(1) @wide.dasan, i64 1024)

@slack = addrspace(1) global [63 x i32] zeroinitializer, align 4
; CHECK: @slack.dasan = private addrspace(1) global { [63 x i32], [68 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 64
; ALIAS: @slack = alias [63 x i32], ptr addrspace(1) @slack.dasan
; ALIAS-NEXT: @__dasan_guard.3 = internal alias [64 x i8], getelementptr (i8, ptr addrspace(1) @slack.dasan, i64 256)

@__omp_rtl_thing = addrspace(1) constant i32 1, align 4
; CHECK: @__omp_rtl_thing = addrspace(1) constant i32 1, section ".rodata.dasan.globals", align 64
; ALIAS-NOT: @__omp_rtl_thing =

@.str = private unnamed_addr addrspace(4) constant [4 x i8] c"abc\00", align 1
; CHECK: @.str = internal unnamed_addr addrspace(4) constant [4 x i8] c"abc\00", section ".rodata.dasan.globals", align 64
; ALIAS-NOT: @.str = {{.*}}alias

$odr = comdat any
@odr = linkonce_odr addrspace(1) global [64 x i32] zeroinitializer, comdat, align 4
; CHECK: @odr.dasan = private addrspace(1) global { [64 x i32], [64 x i8] } zeroinitializer, section ".sbss.dasan.globals", comdat($odr), align 64
; ALIAS: @odr = linkonce_odr alias [64 x i32], ptr addrspace(1) @odr.dasan
; ALIAS-NEXT: @__dasan_guard.4 = internal alias [64 x i8], getelementptr (i8, ptr addrspace(1) @odr.dasan, i64 256)
; ASM: .size odr, 256

@weak = weak addrspace(1) global [64 x i32] zeroinitializer, align 4
; CHECK: @weak.dasan = private addrspace(1) global { [64 x i32], [64 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 64
; ALIAS: @weak = weak alias [64 x i32], ptr addrspace(1) @weak.dasan
; ALIAS-NEXT: @__dasan_guard.5 = internal alias [64 x i8], getelementptr (i8, ptr addrspace(1) @weak.dasan, i64 256)

@big = addrspace(1) global [4096 x i32] zeroinitializer, align 4
; CHECK: @big.dasan = private addrspace(1) global { [4096 x i32], [4096 x i8] } zeroinitializer, section ".sbss.dasan.globals", align 64
; ALIAS: @big = alias [4096 x i32], ptr addrspace(1) @big.dasan
; ALIAS-NEXT: @__dasan_guard.6 = internal alias [4096 x i8], getelementptr (i8, ptr addrspace(1) @big.dasan, i64 16384)

@lds = addrspace(3) global [4 x i32] poison, align 4
; CHECK: @lds = addrspace(3) global [4 x i32] poison, align 4

@ext = external addrspace(1) global [4 x i32]
; CHECK: @ext = external addrspace(1) global [4 x i32]

@pinned = addrspace(1) global i32 0, section "mine", align 4
; CHECK: @pinned = addrspace(1) global i32 0, section "mine", align 4

@vtable = addrspace(1) constant [2 x ptr] zeroinitializer, align 8, !type !0
; CHECK: @vtable = addrspace(1) constant [2 x ptr] zeroinitializer, align 8

; CHECK: @__dasan_no_entry = private addrspace(1) constant i64 0, align 8

; CHECK: @llvm.compiler.used = appending addrspace(1) global {{.*}} @init to ptr), {{.*}} @init.dasan {{.*}} @overaligned.dasan {{.*}} @__dasan_guard {{.*}} @slack.dasan {{.*}} @__dasan_guard.3 {{.*}} @only.dasan
; CHECK: @__dasan_instrumented = linkonce_odr protected addrspace(1) constant i8 1, align 64

define void @use(ptr addrspace(1) %p) sanitize_device_address {
  store i32 1, ptr addrspace(1) %p, align 4
  ret void
}

define i32 @readconst(i64 %i) sanitize_device_address {
; CHECK-LABEL: define i32 @readconst(
; CHECK: [[P:%.*]] = getelementptr inbounds [2 x i32], ptr addrspace(4) @konst, i64 0, i64 %i
; CHECK-NEXT: lshr i64 ptrtoint (ptr addrspace(4) @konst to i64), 45
; CHECK: ptrtoint ptr addrspace(4) [[P]] to i64
; CHECK: call void @__dasan_report_load(
  %p = getelementptr inbounds [2 x i32], ptr addrspace(4) @konst, i64 0, i64 %i
  %v = load i32, ptr addrspace(4) %p, align 4
  ret i32 %v
}

; RUN: opt < %s -passes=dasan -dasan-instrument-writes=0 -S | FileCheck %s --check-prefix=MARK
; MARK: @only.dasan = private addrspace(1) global { i32, [60 x i8] } { i32 1, [60 x i8] zeroinitializer }, section ".data.dasan.globals", align 64
; MARK: @__dasan_instrumented
@only = addrspace(1) global i32 1, align 4

!0 = !{i64 0, !"typeid"}
