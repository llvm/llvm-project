; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O2 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O2 -relocation-model=pic < %s | grep "__tls_get_addr" | count 1

; This test was derived from LLVM's own
; PrettyStackTraceEntry::~PrettyStackTraceEntry().  It demonstrates an
; opportunity for CSE of calls to __tls_get_addr().

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%"class.llvm::PrettyStackTraceEntry" = type { ptr, ptr }

@_ZTVN4llvm21PrettyStackTraceEntryE = unnamed_addr constant [5 x ptr] [ptr null, ptr null, ptr @_ZN4llvm21PrettyStackTraceEntryD2Ev, ptr @_ZN4llvm21PrettyStackTraceEntryD0Ev, ptr @__cxa_pure_virtual], align 8
@_ZL20PrettyStackTraceHead = internal thread_local unnamed_addr global ptr null, align 8
@.str = private unnamed_addr constant [87 x i8] c"PrettyStackTraceHead == this && \22Pretty stack trace entry destruction is out of order\22\00", align 1
@.str1 = private unnamed_addr constant [64 x i8] c"/home/wschmidt/llvm/llvm-test2/lib/Support/PrettyStackTrace.cpp\00", align 1
@__PRETTY_FUNCTION__._ZN4llvm21PrettyStackTraceEntryD2Ev = private unnamed_addr constant [62 x i8] c"virtual llvm::PrettyStackTraceEntry::~PrettyStackTraceEntry()\00", align 1

declare void @_ZN4llvm21PrettyStackTraceEntryD2Ev(ptr %this) unnamed_addr
declare void @__cxa_pure_virtual()
declare void @__assert_fail(ptr, ptr, i32 zeroext, ptr)
declare void @_ZdlPv(ptr)

define void @_ZN4llvm21PrettyStackTraceEntryD0Ev(ptr %this) unnamed_addr align 2 {
entry:
  store ptr getelementptr inbounds ([5 x ptr], ptr @_ZTVN4llvm21PrettyStackTraceEntryE, i64 0, i64 2), ptr %this, align 8
  %0 = load ptr, ptr @_ZL20PrettyStackTraceHead, align 8
  %cmp.i = icmp eq ptr %0, %this
  br i1 %cmp.i, label %_ZN4llvm21PrettyStackTraceEntryD2Ev.exit, label %cond.false.i

cond.false.i:                                     ; preds = %entry
  tail call void @__assert_fail(ptr @.str, ptr @.str1, i32 zeroext 119, ptr @__PRETTY_FUNCTION__._ZN4llvm21PrettyStackTraceEntryD2Ev)
  unreachable

_ZN4llvm21PrettyStackTraceEntryD2Ev.exit:         ; preds = %entry
  %NextEntry.i.i = getelementptr inbounds %"class.llvm::PrettyStackTraceEntry", ptr %this, i64 0, i32 1
  %1 = load i64, ptr %NextEntry.i.i, align 8
  store i64 %1, ptr @_ZL20PrettyStackTraceHead, align 8
  tail call void @_ZdlPv(ptr %this)
  ret void
}

; CHECK-LABEL: _ZN4llvm21PrettyStackTraceEntryD0Ev:
; CHECK: addis [[REG1:[0-9]+]], 2, _ZL20PrettyStackTraceHead@got@tlsld@ha
; CHECK: addi 3, [[REG1]], _ZL20PrettyStackTraceHead@got@tlsld@l
; CHECK: bl __tls_get_addr(_ZL20PrettyStackTraceHead@tlsld)
; CHECK: addis [[REG2:[0-9]+]], 3, _ZL20PrettyStackTraceHead@dtprel@ha
; CHECK: ld {{[0-9]+}}, _ZL20PrettyStackTraceHead@dtprel@l([[REG2]])
; CHECK: std {{[0-9]+}}, _ZL20PrettyStackTraceHead@dtprel@l([[REG2]])
