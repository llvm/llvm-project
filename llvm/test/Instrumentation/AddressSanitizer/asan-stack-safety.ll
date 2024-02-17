; REQUIRES: x86-registered-target

; RUN: opt < %s -S -asan-instrumentation-with-call-threshold=0 -passes=asan -asan-use-stack-safety=0 -o - | FileCheck %s --implicit-check-not="call {{.*}} @__asan_{{load|store|stack}}" --check-prefixes=CHECK,NOSAFETY
; RUN: opt < %s -S -asan-instrumentation-with-call-threshold=0 -passes=asan | FileCheck %s --implicit-check-not="call {{.*}} @__asan_{{load|store|stack}}"

; CHECK-LABEL: define i32 @load
define i32 @load() sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %1 = load i8, ptr %buf, align 1
  ; NOSAFETY: call void @__asan_load1
  ret i32 0
}

; CHECK-LABEL: define i32 @store
define i32 @store() sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  store i8 0, ptr %buf
  ; NOSAFETY: call void @__asan_store1
  ret i32 0
}

; CHECK-LABEL: define i32 @unsafe_alloca
define i32 @unsafe_alloca(i32 %i) sanitize_address {
  %buf.sroa.0 = alloca [10 x i8], align 4
  ; CHECK: call i64 @__asan_stack_malloc
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 %i, i32 0
  store volatile i8 0, ptr %ptr, align 4
  ; CHECK: call void @__asan_store1
  store volatile i8 0, ptr %buf.sroa.0, align 4
  ; NOSAFETY: call void @__asan_store1
  ret i32 0
}

; CHECK-LABEL: define void @atomicrmw
define void @atomicrmw() sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %1 = atomicrmw add ptr %buf, i8 1 seq_cst
  ; NOSAFETY: call void @__asan_store1
  ret void
}

; CHECK-LABEL: define void @cmpxchg
define void @cmpxchg(i8 %compare_to, i8 %new_value) sanitize_address {
  %buf = alloca [10 x i8], align 1
  ; NOSAFETY: call i64 @__asan_stack_malloc
  %1 = cmpxchg ptr %buf, i8 %compare_to, i8 %new_value seq_cst seq_cst
  ; NOSAFETY: call void @__asan_store1
  ret void
}

%struct.S = type { i32, i32 }

; CHECK-LABEL: define %struct.S @exchange(
; NOSAFETY: call i64 @__asan_stack_malloc
; CHECK:    call ptr @__asan_memcpy(
; CHECK:    call ptr @__asan_memcpy(
; NOSAFETY: call void @__asan_loadN(
define %struct.S @exchange(ptr %a, ptr %b) sanitize_address {
entry:
  %tmp = alloca %struct.S, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp, ptr align 4 %a, i64 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %b, i64 8, i1 false)
  %ret = load %struct.S, ptr %tmp
  ret %struct.S %ret
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) nounwind
