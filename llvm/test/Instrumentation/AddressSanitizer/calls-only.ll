; RUN: opt < %s -passes=asan -asan-max-inline-poisoning-size=0   -asan-stack-dynamic-alloca=0 -S | FileCheck --check-prefix=OUTLINE %s
; RUN: opt < %s -passes=asan -asan-max-inline-poisoning-size=999 -asan-stack-dynamic-alloca=0 -S | FileCheck --check-prefix=INLINE  %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx13.0.0"

; Function Attrs: noinline nounwind optnone sanitize_address ssp uwtable(sync)
define void @foo() #0 {
entry:
  %array01 = alloca [1 x i8], align 1
  %array02 = alloca [2 x i8], align 1
  %array03 = alloca [3 x i8], align 1
  %array04 = alloca [4 x i8], align 1
  %array05 = alloca [5 x i8], align 1
  %array06 = alloca [6 x i8], align 1
  %array07 = alloca [7 x i8], align 1
; OUTLINE:  call void @__asan_set_shadow_f1(i64 %{{.+}}, i64 4)
; OUTLINE:  call void @__asan_set_shadow_01(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_02(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_03(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_04(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_05(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %{{.+}}, i64 3)
; OUTLINE:  call void @__asan_set_shadow_06(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f2(i64 %{{.+}}, i64 3)
; OUTLINE:  call void @__asan_set_shadow_07(i64 %{{.+}}, i64 1)
; OUTLINE:  call void @__asan_set_shadow_f3(i64 %{{.+}}, i64 3)
; OUTLINE:  call void @__asan_stack_free_2(i64 %{{.+}}, i64 192)
; OUTLINE:  call void @__asan_set_shadow_00(i64 %{{.+}}, i64 24)
; INLINE:  store i64 -1007977276409515535, ptr %{{.+}}, align 1
; INLINE:  store i64 -940423264817843709, ptr %{{.+}}, align 1
; INLINE:  store i64 -868083087686045178, ptr %{{.+}}, align 1
  %arrayidx = getelementptr inbounds [1 x i8], ptr %array01, i64 0, i64 1
  store i8 1, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds [2 x i8], ptr %array02, i64 0, i64 2
  store i8 2, ptr %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds [3 x i8], ptr %array03, i64 0, i64 3
  store i8 3, ptr %arrayidx2, align 1
  %arrayidx3 = getelementptr inbounds [4 x i8], ptr %array04, i64 0, i64 4
  store i8 4, ptr %arrayidx3, align 1
  %arrayidx4 = getelementptr inbounds [5 x i8], ptr %array05, i64 0, i64 5
  store i8 5, ptr %arrayidx4, align 1
  %arrayidx5 = getelementptr inbounds [6 x i8], ptr %array06, i64 0, i64 6
  store i8 6, ptr %arrayidx5, align 1
  %arrayidx6 = getelementptr inbounds [7 x i8], ptr %array07, i64 0, i64 7
  store i8 7, ptr %arrayidx6, align 1
; CHECK-NOT:  store i64 -723401728380766731, ptr %{{.+}}, align 1
  ret void
}
attributes #0 = { noinline nounwind optnone sanitize_address ssp uwtable(sync) "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" }

