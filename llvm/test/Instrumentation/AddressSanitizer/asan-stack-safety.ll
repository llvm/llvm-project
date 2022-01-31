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
; NOSAFETY: call void @__asan_store1
; NOSAFETY: call void @__asan_store1
; NOSAFETY: call void @__asan_store1
; SAFETY-NOT: call void @__asan_load1
; SAFETY-NOT: call void @__asan_store1
; SAFETY-NOT: call void @__asan_store1
; SAFETY-NOT: call void @__asan_store1

define i32 @load() sanitize_address {
  %buf = alloca [10 x i8], align 1
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = load i8, i8* %arrayidx, align 1
  ret i32 0
}

define i32 @store() sanitize_address {
  %buf = alloca [10 x i8], align 1
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  store i8 0, i8* %arrayidx
  ret i32 0
}


define void @atomicrmw() sanitize_address {
  %buf = alloca [10 x i8], align 1
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = atomicrmw add i8* %arrayidx, i8 1 seq_cst
  ret void
}

define void @cmpxchg(i8 %compare_to, i8 %new_value) sanitize_address {
  %buf = alloca [10 x i8], align 1
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = cmpxchg i8* %arrayidx, i8 %compare_to, i8 %new_value seq_cst seq_cst
  ret void
}
