; REQUIRES: x86-registered-target

; RUN: opt < %s -S -enable-new-pm=0 -asan-instrumentation-with-call-threshold=0 -asan \
; RUN:     -asan-use-stack-safety=0  -o - | FileCheck %s --check-prefixes=NOSAFETY
; RUN: opt < %s -S -enable-new-pm=0 -asan-instrumentation-with-call-threshold=0 -asan \
; RUN:     -asan-use-stack-safety=1 -o - | FileCheck %s --check-prefixes=SAFETY
; RUN: opt < %s -S -enable-new-pm=1 -asan-instrumentation-with-call-threshold=0 \
; RUN:     -passes='asan-pipeline' -asan-use-stack-safety=0 -o - | FileCheck %s --check-prefixes=NOSAFETY
; RUN: opt < %s -S -enable-new-pm=1 -asan-instrumentation-with-call-threshold=0 \
; RUN:     -passes='asan-pipeline' -asan-use-stack-safety=1 -o - | FileCheck %s --check-prefixes=SAFETY
; NOSAFETY: call void @__asan_load1
; SAFETY-NOT: call void @__asan_load1

define i32 @stack-safety() sanitize_address {
  %buf = alloca [10 x i8], align 1
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = load i8, i8* %arrayidx, align 1
  ret i32 0
}
