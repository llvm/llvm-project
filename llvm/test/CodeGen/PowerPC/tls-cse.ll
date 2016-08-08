; RUN: llc -verify-machineinstrs -march=ppc64 -mcpu=pwr7 -O2 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -march=ppc64 -mcpu=pwr7 -O2 -relocation-model=pic < %s | grep "__tls_get_addr" | count 1

; This test was derived from LLVM's own
; PrettyStackTraceEntry::~PrettyStackTraceEntry().  It demonstrates an
; opportunity for CSE of calls to __tls_get_addr().

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%"class.llvm::PrettyStackTraceEntry" = type { i32 (...)**, %"class.llvm::PrettyStackTraceEntry"* }

@_ZTVN4llvm21PrettyStackTraceEntryE = unnamed_addr constant [5 x i8*] [i8* null, i8* null, i8* bitcast (void (%"class.llvm::PrettyStackTraceEntry"*)* @_ZN4llvm21PrettyStackTraceEntryD2Ev to i8*), i8* bitcast (void (%"class.llvm::PrettyStackTraceEntry"*)* @_ZN4llvm21PrettyStackTraceEntryD0Ev to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)], align 8
@_ZL20PrettyStackTraceHead = internal thread_local unnamed_addr global %"class.llvm::PrettyStackTraceEntry"* null, align 8
@.str = private unnamed_addr constant [87 x i8] c"PrettyStackTraceHead == this && \22Pretty stack trace entry destruction is out of order\22\00", align 1
@.str1 = private unnamed_addr constant [64 x i8] c"/home/wschmidt/llvm/llvm-test2/lib/Support/PrettyStackTrace.cpp\00", align 1
@__PRETTY_FUNCTION__._ZN4llvm21PrettyStackTraceEntryD2Ev = private unnamed_addr constant [62 x i8] c"virtual llvm::PrettyStackTraceEntry::~PrettyStackTraceEntry()\00", align 1

declare void @_ZN4llvm21PrettyStackTraceEntryD2Ev(%"class.llvm::PrettyStackTraceEntry"* %this) unnamed_addr
declare void @__cxa_pure_virtual()
declare void @__assert_fail(i8*, i8*, i32 zeroext, i8*)
declare void @_ZdlPv(i8*)

define void @_ZN4llvm21PrettyStackTraceEntryD0Ev(%"class.llvm::PrettyStackTraceEntry"* %this) unnamed_addr align 2 {
entry:
  %0 = getelementptr inbounds %"class.llvm::PrettyStackTraceEntry", %"class.llvm::PrettyStackTraceEntry"* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN4llvm21PrettyStackTraceEntryE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  %1 = load %"class.llvm::PrettyStackTraceEntry"*, %"class.llvm::PrettyStackTraceEntry"** @_ZL20PrettyStackTraceHead, align 8
  %cmp.i = icmp eq %"class.llvm::PrettyStackTraceEntry"* %1, %this
  br i1 %cmp.i, label %_ZN4llvm21PrettyStackTraceEntryD2Ev.exit, label %cond.false.i

cond.false.i:                                     ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([87 x i8], [87 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str1, i64 0, i64 0), i32 zeroext 119, i8* getelementptr inbounds ([62 x i8], [62 x i8]* @__PRETTY_FUNCTION__._ZN4llvm21PrettyStackTraceEntryD2Ev, i64 0, i64 0))
  unreachable

_ZN4llvm21PrettyStackTraceEntryD2Ev.exit:         ; preds = %entry
  %NextEntry.i.i = getelementptr inbounds %"class.llvm::PrettyStackTraceEntry", %"class.llvm::PrettyStackTraceEntry"* %this, i64 0, i32 1
  %2 = bitcast %"class.llvm::PrettyStackTraceEntry"** %NextEntry.i.i to i64*
  %3 = load i64, i64* %2, align 8
  store i64 %3, i64* bitcast (%"class.llvm::PrettyStackTraceEntry"** @_ZL20PrettyStackTraceHead to i64*), align 8
  %4 = bitcast %"class.llvm::PrettyStackTraceEntry"* %this to i8*
  tail call void @_ZdlPv(i8* %4)
  ret void
}

; CHECK-LABEL: _ZN4llvm21PrettyStackTraceEntryD0Ev:
; CHECK: addis [[REG1:[0-9]+]], 2, _ZL20PrettyStackTraceHead@got@tlsld@ha
; CHECK: addi 3, [[REG1]], _ZL20PrettyStackTraceHead@got@tlsld@l
; CHECK: bl __tls_get_addr(_ZL20PrettyStackTraceHead@tlsld)
; CHECK: addis 3, 3, _ZL20PrettyStackTraceHead@dtprel@ha
; CHECK: ld {{[0-9]+}}, _ZL20PrettyStackTraceHead@dtprel@l(3)
; CHECK: std {{[0-9]+}}, _ZL20PrettyStackTraceHead@dtprel@l(3)
