; REQUIRES: x86-registered-target
; RUN: opt -passes='default<O3>' -S %s | FileCheck %s
; RUN: opt -passes='lto<O3>' -S %s | FileCheck %s
;
; An AVX2 loop cannot inline the AVX512 helper selected on some machines.
; All helper bodies are equivalent, so a compatible body can inline and expose
; integer arithmetic to the loop vectorizer.

target triple = "x86_64-unknown-linux-gnu"
@features = external global i32
@mix = weak_odr ifunc i32 (i32), ptr @mix.resolver

define ptr @mix.resolver() {
  %f = load i32, ptr @features
  %a = and i32 %f, 1
  %b = and i32 %f, 2
  %ca = icmp ne i32 %a, 0
  %cb = icmp ne i32 %b, 0
  %lo = select i1 %ca, ptr @mix.avx2, ptr @mix.default
  %hi = select i1 %cb, ptr @mix.avx512, ptr %lo
  ret ptr %hi
}

define i32 @mix.default(i32 %x) #0 {
  %s = lshr i32 %x, 16
  %a = xor i32 %x, %s
  %m = mul i32 %a, 2146121005
  ret i32 %m
}

define i32 @mix.avx2(i32 %x) #1 {
  %s = lshr i32 %x, 16
  %a = xor i32 %x, %s
  %m = mul i32 %a, 2146121005
  ret i32 %m
}

define i32 @mix.avx512(i32 %x) #2 {
  %s = lshr i32 %x, 16
  %a = xor i32 %x, %s
  %m = mul i32 %a, 2146121005
  ret i32 %m
}

; CHECK-LABEL: define void @hash_array(
; CHECK-NOT: call
; CHECK: lshr <8 x i32>
; CHECK: mul <8 x i32>
; CHECK-NOT: call
; CHECK: ret void
define void @hash_array(ptr noalias %x, ptr noalias %y, i64 %n) #3 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr i32, ptr %x, i64 %i
  %v = load i32, ptr %p, align 4
  %h = call i32 @mix(i32 %v)
  %q = getelementptr i32, ptr %y, i64 %i
  store i32 %h, ptr %q, align 4
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
attributes #0 = { "fmv-features" "target-features"="+sse2" }
attributes #1 = { "fmv-features"="avx2" "target-features"="+avx2" }
attributes #2 = { "fmv-features"="avx512f" "target-features"="+avx512f" }
attributes #3 = { "target-features"="+avx2" }
