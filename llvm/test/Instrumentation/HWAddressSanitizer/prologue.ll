; Test -hwasan-with-ifunc flag.
;
; RUN: opt -passes=hwasan -S < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-NOGLOBAL,CHECK-TLS-SLOT,CHECK-HISTORY,CHECK-HISTORY-TLS-SLOT,CHECK-HISTORY-TLS
; RUN: opt -passes=hwasan -S -hwasan-with-ifunc=0 -hwasan-with-tls=1 -hwasan-record-stack-history=instr < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-NOGLOBAL,CHECK-TLS-SLOT,CHECK-HISTORY,CHECK-HISTORY-TLS-SLOT,CHECK-HISTORY-TLS
; RUN: opt -passes=hwasan -S -hwasan-with-ifunc=0 -hwasan-with-tls=1 -hwasan-record-stack-history=none < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-NOGLOBAL,CHECK-IFUNC,CHECK-NOHISTORY
; RUN: opt -passes=hwasan -S -hwasan-with-ifunc=0 -hwasan-with-tls=0 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-GLOBAL,CHECK-NOHISTORY
; RUN: opt -passes=hwasan -S -hwasan-with-ifunc=1  -hwasan-with-tls=0 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC,CHECK-NOHISTORY
; RUN: opt -passes=hwasan -S -mtriple=aarch64-fuchsia < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-ZERO-OFFSET,CHECK-SHORT-GRANULES,CHECK-HISTORY,CHECK-HWASAN-TLS,CHECK-HISTORY-HWASAN-TLS,CHECK-HISTORY-TLS
; RUN: opt -passes=hwasan -S -mtriple=aarch64-fuchsia -hwasan-record-stack-history=libcall < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-HISTORY,CHECK-HISTORY-LIBCALL

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android22"

; CHECK-IFUNC: @__hwasan_shadow = external global [0 x i8]
; CHECK-NOIFUNC: @__hwasan_shadow_memory_dynamic_address = external global i64

define i32 @test_load(i32* %a) sanitize_hwaddress {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:

; CHECK-NOGLOBAL:   %[[A:[^ ]*]] = call i8* asm "", "=r,0"([0 x i8]* @__hwasan_shadow)
; CHECK-NOGLOBAL:   @llvm.hwasan.check.memaccess(i8* %[[A]]
; CHECK-ZERO-OFFSET:  %[[A:[^ ]*]] = call i8* asm "", "=r,0"(i8* null)
; CHECK-SHORT-GRANULES:  @llvm.hwasan.check.memaccess.shortgranules(i8* %[[A]]

; CHECK-GLOBAL: load i8*, i8** @__hwasan_shadow_memory_dynamic_address

; "store i64" is only used to update stack history (this input IR intentionally does not use any i64)
; W/o any allocas, the history is not updated, even if it is enabled explicitly with -hwasan-record-stack-history=1
; CHECK-NOT: store i64

; CHECK: ret i32

entry:
  %x = load i32, i32* %a, align 4
  ret i32 %x
}

declare void @use(i32* %p)

define void @test_alloca() sanitize_hwaddress {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_alloca
; CHECK: entry:

; CHECK-IFUNC:   %[[A:[^ ]*]] = call i8* asm "", "=r,0"([0 x i8]* @__hwasan_shadow)
; CHECK-IFUNC:   getelementptr i8, i8* %[[A]]

; CHECK-GLOBAL: load i8*, i8** @__hwasan_shadow_memory_dynamic_address

; CHECK-TLS-SLOT:   %[[A:[^ ]*]] = call i8* @llvm.thread.pointer()
; CHECK-TLS-SLOT:   %[[B:[^ ]*]] = getelementptr i8, i8* %[[A]], i32 48
; CHECK-TLS-SLOT:   %[[C:[^ ]*]] = bitcast i8* %[[B]] to i64*
; CHECK-TLS-SLOT:   %[[D:[^ ]*]] = load i64, i64* %[[C]]
; CHECK-TLS-SLOT:   %[[E:[^ ]*]] = ashr i64 %[[D]], 3
; CHECK-HWASAN-TLS: %[[D:[^ ]*]] = load i64, i64* @__hwasan_tls, align 8
; CHECK-HWASAN-TLS: %[[E:[^ ]*]] = ashr i64 %[[D]], 3

; CHECK-NOHISTORY-NOT: store i64

; When watching stack history, all code paths attempt to get PC and SP and mix them together.
; CHECK-HISTORY: %[[PC:[^ ]*]] = call i64 @llvm.read_register.i64(metadata [[MD:![0-9]*]])
; CHECK-HISTORY: %[[SP0:[^ ]*]] = call i8* @llvm.frameaddress.p0i8(i32 0)
; CHECK-HISTORY: %[[SP1:[^ ]*]] = ptrtoint i8* %[[SP0]] to i64
; CHECK-HISTORY: %[[SP2:[^ ]*]] = shl i64 %[[SP1]], 44

; CHECK-HISTORY: %[[MIX:[^ ]*]] = or i64 %[[PC]], %[[SP2]]
; CHECK-HISTORY-TLS: %[[PTR:[^ ]*]] = inttoptr i64 %[[D]] to i64*
; CHECK-HISTORY-TLS: store i64 %[[MIX]], i64* %[[PTR]]
; CHECK-HISTORY-TLS: %[[D1:[^ ]*]] = ashr i64 %[[D]], 56
; CHECK-HISTORY-TLS: %[[D2:[^ ]*]] = shl nuw nsw i64 %[[D1]], 12
; CHECK-HISTORY-TLS: %[[D3:[^ ]*]] = xor i64 %[[D2]], -1
; CHECK-HISTORY-TLS: %[[D4:[^ ]*]] = add i64 %[[D]], 8
; CHECK-HISTORY-TLS: %[[D5:[^ ]*]] = and i64 %[[D4]], %[[D3]]
; CHECK-HISTORY-TLS-SLOT: store i64 %[[D5]], i64* %[[C]]
; CHECK-HISTORY-HWASAN-TLS: store i64 %[[D5]], i64* @__hwasan_tls
; CHECK-HISTORY-LIBCALL: call void @__hwasan_add_frame_record(i64 %[[MIX]])

; CHECK-TLS:   %[[F:[^ ]*]] = or i64 %[[D]], 4294967295
; CHECK-TLS:   = add i64 %[[F]], 1

; CHECK-HISTORY-LIBCALL: %[[E:hwasan.stack.base.tag]] = xor
; CHECK-HISTORY: = xor i64 %[[E]], 0

; CHECK-NOHISTORY-NOT: store i64


entry:
  %x = alloca i32, align 4
  call void @use(i32* %x)
  ret void
}

; CHECK-HISTORY: [[MD]] = !{!"pc"}
