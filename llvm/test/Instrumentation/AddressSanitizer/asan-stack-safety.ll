; REQUIRES: x86-registered-target

; RUN: opt < %s -S -asan-instrumentation-with-call-threshold=0 -passes='asan-pipeline' -asan-use-stack-safety=0 -o - | FileCheck %s --implicit-check-not="call {{.*}} @__asan_{{load|store|stack}}" --check-prefixes=CHECK,NOSAFETY
; RUN: opt < %s -S -asan-instrumentation-with-call-threshold=0 -passes='asan-pipeline' -asan-use-stack-safety=1 -o - | FileCheck %s --implicit-check-not="call {{.*}} @__asan_{{load|store|stack}}"

; CHECK-LABEL: define i32 @load
define i32 @load() sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = load i8, i8* %arrayidx, align 1
  ; NOSAFETY: call void @__asan_load1
  ret i32 0
}

; CHECK-LABEL: define i32 @store
define i32 @store() sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  store i8 0, i8* %arrayidx
  ; NOSAFETY: call void @__asan_store1
  ret i32 0
}

; CHECK-LABEL: define i32 @unsafe_alloca
define i32 @unsafe_alloca(i32 %i) sanitize_address {
  %buf.sroa.0 = alloca [10 x i8], align 4
  ; CHECK: call i64 @__asan_stack_malloc
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 %i, i32 0
  store volatile i8 0, i8* %ptr, align 4
  ; CHECK: call void @__asan_store1
  %ptr2 = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 0
  store volatile i8 0, i8* %ptr2, align 4
  ; NOSAFETY: call void @__asan_store1
  ret i32 0
}

; CHECK-LABEL: define void @atomicrmw
define void @atomicrmw() sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = atomicrmw add i8* %arrayidx, i8 1 seq_cst
  ; NOSAFETY: call void @__asan_store1
  ret void
}

; CHECK-LABEL: define void @cmpxchg
define void @cmpxchg(i8 %compare_to, i8 %new_value) sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  %1 = cmpxchg i8* %arrayidx, i8 %compare_to, i8 %new_value seq_cst seq_cst
  ; NOSAFETY: call void @__asan_store1
  ret void
}
