; Calls that SanitizerCoverage inserts into a funclet carry the funclet's
; "funclet" operand bundle. WinEHPrepare takes a call in a funclet without one
; for a call that does not belong there, and replaces it and the rest of its
; block with unreachable: the handler below would lose everything from the
; comparison on, the destructor call and the rethrow with it.

; The coverage -fsanitize=fuzzer asks for.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-inline-8bit-counters -sanitizer-coverage-trace-compares -S | FileCheck %s
; With the callbacks gated, the calls are in blocks split off inside the funclet.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc-guard -sanitizer-coverage-trace-compares -sanitizer-coverage-gated-trace-callbacks -S | FileCheck %s --check-prefix=GATED

; Generated from this C++ source, then renamed:
; $ clang++ --target=x86_64-pc-windows-msvc -O2 -S -emit-llvm t.cpp
; struct Element { virtual ~Element(); };
; void read(Element *element);
; void read_guarded(Element *element, int &found, int limit) {
;   try {
;     read(element);
;   } catch (...) {
;     if (found > limit)
;       found = 0;
;     delete element;
;     throw;
;   }
; }

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.33.0"

define void @"?read_guarded@@YAXPEAUElement@@AEAHH@Z"(ptr %element, ptr %found, i32 %limit) personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @"?read@@YAXPEAUElement@@@Z"(ptr %element)
          to label %return unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %catch] unwind to caller

catch:
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  %2 = load i32, ptr %found, align 4
  %cmp = icmp sgt i32 %2, %limit
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 0, ptr %found, align 4
  br label %if.end

if.end:
  %isnull = icmp eq ptr %element, null
  br i1 %isnull, label %rethrow, label %delete.notnull

delete.notnull:
  %vtable = load ptr, ptr %element, align 8
  %dtor = load ptr, ptr %vtable, align 8
  %call = call ptr %dtor(ptr %element, i32 1) [ "funclet"(token %1) ]
  br label %rethrow

rethrow:
  call void @_CxxThrowException(ptr null, ptr null) [ "funclet"(token %1) ]
  unreachable

return:
  ret void
}

; CHECK-LABEL: define void @"?read_guarded@@YAXPEAUElement@@AEAHH@Z"(
; CHECK:       catch:
; CHECK-NEXT:    %[[PAD:[0-9]+]] = catchpad within %{{[0-9]+}} [ptr null, i32 64, ptr null]
; CHECK:         call void @__sanitizer_cov_trace_cmp4(i32 %{{[0-9]+}}, i32 %limit) [ "funclet"(token %[[PAD]]) ]
; CHECK-NEXT:    %cmp = icmp sgt i32 %{{[0-9]+}}, %limit
; CHECK:       delete.notnull:
; CHECK:         call void @__sanitizer_cov_trace_pc_indir(i64 %{{[0-9]+}}) [ "funclet"(token %[[PAD]]) ]
; CHECK-NEXT:    %call = call ptr %dtor(ptr %element, i32 1) [ "funclet"(token %[[PAD]]) ]
; CHECK:       rethrow:
; CHECK:         call void @_CxxThrowException(ptr null, ptr null) [ "funclet"(token %[[PAD]]) ]

; Outside the funclet, no bundle.
; GATED-LABEL: define void @"?read_guarded@@YAXPEAUElement@@AEAHH@Z"(
; GATED:         call void @__sanitizer_cov_trace_pc_guard(ptr @__sancov_gen_) #{{[0-9]+}}{{$}}
; GATED:       catch:
; GATED-NEXT:    %[[PAD:[0-9]+]] = catchpad within %{{[0-9]+}} [ptr null, i32 64, ptr null]
; GATED:         call void @__sanitizer_cov_trace_cmp4(i32 %{{[0-9]+}}, i32 %limit) [ "funclet"(token %[[PAD]]) ]
; GATED:         call void @__sanitizer_cov_trace_pc_guard({{.*}}) #{{[0-9]+}} [ "funclet"(token %[[PAD]]) ]
; GATED:         call void @__sanitizer_cov_trace_pc_indir(i64 %{{[0-9]+}}) [ "funclet"(token %[[PAD]]) ]
; GATED-NEXT:    %call = call ptr %dtor(ptr %element, i32 1) [ "funclet"(token %[[PAD]]) ]
; GATED:       return:
; GATED:         call void @__sanitizer_cov_trace_pc_guard({{.*}}) #{{[0-9]+}}{{$}}

declare void @"?read@@YAXPEAUElement@@@Z"(ptr)
declare i32 @__CxxFrameHandler3(...)
declare void @_CxxThrowException(ptr, ptr)
