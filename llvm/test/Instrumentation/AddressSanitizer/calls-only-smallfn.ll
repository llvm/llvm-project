; RUN: opt < %s -passes=asan -asan-max-inline-poisoning-size=0   -asan-stack-dynamic-alloca=0 -S | FileCheck --check-prefix=OUTLINE %s
; RUN: opt < %s -passes=asan -asan-max-inline-poisoning-size=999 -asan-stack-dynamic-alloca=0 -S | FileCheck --check-prefix=INLINE  %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx13.0.0"

; Function Attrs: noinline nounwind optnone sanitize_address ssp uwtable(sync)
define void @foo() #0 {
entry:
  %array01 = alloca [1 x i8], align 1
  %array02 = alloca [2 x i8], align 1
; OUTLINE:  call void @__asan_set_shadow_f1(i64 %23, i64 4)
; OUTLINE:  call void @__asan_set_shadow_01(i64 %24, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %25, i64 1)
; OUTLINE:  call void @__asan_set_shadow_02(i64 %26, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f3(i64 %27, i64 1)
; OUTLINE:  call void @__asan_stack_free_0(i64 %7, i64 64)
; OUTLINE:  call void @__asan_set_shadow_00(i64 %55, i64 8)
; INLINE:  store i64 -935919682371587599, ptr %24, align 1
; INLINE:  store i64 -723401728380766731, ptr %52, align 1
  %arrayidx = getelementptr inbounds [1 x i8], ptr %array01, i64 0, i64 1
  store i8 1, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds [2 x i8], ptr %array02, i64 0, i64 2
  store i8 2, ptr %arrayidx1, align 1
  ret void
}
attributes #0 = { noinline nounwind optnone sanitize_address ssp uwtable(sync) "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }

