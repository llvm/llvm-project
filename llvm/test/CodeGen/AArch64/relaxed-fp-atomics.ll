; PR52927: Relaxed atomics can load to/store from fp regs directly
; RUN: llc < %s -mtriple=arm64-eabi -asm-verbose=false -verify-machineinstrs -mcpu=cyclone | FileCheck %s

define float @atomic_load_relaxed_f32(ptr %p, i32 %off32, i64 %off64) #0 {
; CHECK-LABEL: atomic_load_relaxed_f32:
  %ptr_unsigned = getelementptr float, ptr %p, i32 4095
  %val_unsigned = load atomic float, ptr %ptr_unsigned monotonic, align 4
; CHECK: ldr {{s[0-9]+}}, [x0, #16380]

  %ptr_regoff = getelementptr float, ptr %p, i32 %off32
  %val_regoff = load atomic float, ptr %ptr_regoff unordered, align 4
  %tot1 = fadd float %val_unsigned, %val_regoff
; CHECK: ldr {{s[0-9]+}}, [x0, w1, sxtw #2]

  %ptr_regoff64 = getelementptr float, ptr %p, i64 %off64
  %val_regoff64 = load atomic float, ptr %ptr_regoff64 monotonic, align 4
  %tot2 = fadd float %tot1, %val_regoff64
; CHECK: ldr {{s[0-9]+}}, [x0, x2, lsl #2]

  %ptr_unscaled = getelementptr float, ptr %p, i32 -64
  %val_unscaled = load atomic float, ptr %ptr_unscaled unordered, align 4
  %tot3 = fadd float %tot2, %val_unscaled
; CHECK: ldur {{s[0-9]+}}, [x0, #-256]

  ret float %tot3
}

define double @atomic_load_relaxed_f64(ptr %p, i32 %off32, i64 %off64) #0 {
; CHECK-LABEL: atomic_load_relaxed_f64:
  %ptr_unsigned = getelementptr double, ptr %p, i32 4095
  %val_unsigned = load atomic double, ptr %ptr_unsigned monotonic, align 8
; CHECK: ldr {{d[0-9]+}}, [x0, #32760]

  %ptr_regoff = getelementptr double, ptr %p, i32 %off32
  %val_regoff = load atomic double, ptr %ptr_regoff unordered, align 8
  %tot1 = fadd double %val_unsigned, %val_regoff
; CHECK: ldr {{d[0-9]+}}, [x0, w1, sxtw #3]

  %ptr_regoff64 = getelementptr double, ptr %p, i64 %off64
  %val_regoff64 = load atomic double, ptr %ptr_regoff64 monotonic, align 8
  %tot2 = fadd double %tot1, %val_regoff64
; CHECK: ldr {{d[0-9]+}}, [x0, x2, lsl #3]

  %ptr_unscaled = getelementptr double, ptr %p, i32 -32
  %val_unscaled = load atomic double, ptr %ptr_unscaled unordered, align 8
  %tot3 = fadd double %tot2, %val_unscaled
; CHECK: ldur {{d[0-9]+}}, [x0, #-256]

  ret double %tot3
}

define void @atomic_store_relaxed_f32(ptr %p, i32 %off32, i64 %off64, float %val) #0 {
; CHECK-LABEL: atomic_store_relaxed_f32:
  %ptr_unsigned = getelementptr float, ptr %p, i32 4095
  store atomic float %val, ptr %ptr_unsigned monotonic, align 4
; CHECK: str {{s[0-9]+}}, [x0, #16380]

  %ptr_regoff = getelementptr float, ptr %p, i32 %off32
  store atomic float %val, ptr %ptr_regoff unordered, align 4
; CHECK: str {{s[0-9]+}}, [x0, w1, sxtw #2]

  %ptr_regoff64 = getelementptr float, ptr %p, i64 %off64
  store atomic float %val, ptr %ptr_regoff64 monotonic, align 4
; CHECK: str {{s[0-9]+}}, [x0, x2, lsl #2]

  %ptr_unscaled = getelementptr float, ptr %p, i32 -64
  store atomic float %val, ptr %ptr_unscaled unordered, align 4
; CHECK: stur {{s[0-9]+}}, [x0, #-256]

  ret void
}

define void @atomic_store_relaxed_f64(ptr %p, i32 %off32, i64 %off64, double %val) #0 {
; CHECK-LABEL: atomic_store_relaxed_f64:
  %ptr_unsigned = getelementptr double, ptr %p, i32 4095
  store atomic double %val, ptr %ptr_unsigned monotonic, align 8
; CHECK: str {{d[0-9]+}}, [x0, #32760]

  %ptr_regoff = getelementptr double, ptr %p, i32 %off32
  store atomic double %val, ptr %ptr_regoff unordered, align 8
; CHECK: str {{d[0-9]+}}, [x0, w1, sxtw #3]

  %ptr_regoff64 = getelementptr double, ptr %p, i64 %off64
  store atomic double %val, ptr %ptr_regoff64 unordered, align 8
; CHECK: str {{d[0-9]+}}, [x0, x2, lsl #3]

  %ptr_unscaled = getelementptr double, ptr %p, i32 -32
  store atomic double %val, ptr %ptr_unscaled monotonic, align 8
; CHECK: stur {{d[0-9]+}}, [x0, #-256]

  ret void
}

define half @atomic_load_relaxed_f16(ptr %p, i32 %off32, i64 %off64) #0 {
; CHECK-LABEL: atomic_load_relaxed_f16:
  %ptr_unsigned = getelementptr half, ptr %p, i32 4095
  %val_unsigned = load atomic half, ptr %ptr_unsigned monotonic, align 4
; CHECK: ldrh {{w[0-9]+}}, [x0, #8190]

  %ptr_regoff = getelementptr half, ptr %p, i32 %off32
  %val_regoff = load atomic half, ptr %ptr_regoff unordered, align 4
  %tot1 = fadd half %val_unsigned, %val_regoff
; CHECK: ldrh {{w[0-9]+}}, [x0, w1, sxtw #1]

  %ptr_regoff64 = getelementptr half, ptr %p, i64 %off64
  %val_regoff64 = load atomic half, ptr %ptr_regoff64 monotonic, align 4
  %tot2 = fadd half %tot1, %val_regoff64
; CHECK: ldrh {{w[0-9]+}}, [x0, x2, lsl #1]

  %ptr_unscaled = getelementptr half, ptr %p, i32 -64
  %val_unscaled = load atomic half, ptr %ptr_unscaled unordered, align 4
  %tot3 = fadd half %tot2, %val_unscaled
; CHECK: ldurh {{w[0-9]+}}, [x0, #-128]

  ret half %tot3
}

define bfloat @atomic_load_relaxed_bf16(ptr %p, i32 %off32, i64 %off64) #0 {
; CHECK-LABEL: atomic_load_relaxed_bf16:
  %ptr_unsigned = getelementptr bfloat, ptr %p, i32 4095
  %val_unsigned = load atomic bfloat, ptr %ptr_unsigned monotonic, align 4
; CHECK: ldrh {{w[0-9]+}}, [x0, #8190]

  %ptr_regoff = getelementptr bfloat, ptr %p, i32 %off32
  %val_regoff = load atomic bfloat, ptr %ptr_regoff unordered, align 4
  %tot1 = fadd bfloat %val_unsigned, %val_regoff
; CHECK: ldrh {{w[0-9]+}}, [x0, w1, sxtw #1]

  %ptr_regoff64 = getelementptr bfloat, ptr %p, i64 %off64
  %val_regoff64 = load atomic bfloat, ptr %ptr_regoff64 monotonic, align 4
  %tot2 = fadd bfloat %tot1, %val_regoff64
; CHECK: ldrh {{w[0-9]+}}, [x0, x2, lsl #1]

  %ptr_unscaled = getelementptr bfloat, ptr %p, i32 -64
  %val_unscaled = load atomic bfloat, ptr %ptr_unscaled unordered, align 4
  %tot3 = fadd bfloat %tot2, %val_unscaled
; CHECK: ldurh {{w[0-9]+}}, [x0, #-128]

  ret bfloat %tot3
}

define void @atomic_store_relaxed_f16(ptr %p, i32 %off32, i64 %off64, half %val) #0 {
; CHECK-LABEL: atomic_store_relaxed_f16:
  %ptr_unsigned = getelementptr half, ptr %p, i32 4095
  store atomic half %val, ptr %ptr_unsigned monotonic, align 4
; CHECK: strh {{w[0-9]+}}, [x0, #8190]

  %ptr_regoff = getelementptr half, ptr %p, i32 %off32
  store atomic half %val, ptr %ptr_regoff unordered, align 4
; CHECK: strh {{w[0-9]+}}, [x0, w1, sxtw #1]

  %ptr_regoff64 = getelementptr half, ptr %p, i64 %off64
  store atomic half %val, ptr %ptr_regoff64 monotonic, align 4
; CHECK: strh {{w[0-9]+}}, [x0, x2, lsl #1]

  %ptr_unscaled = getelementptr half, ptr %p, i32 -64
  store atomic half %val, ptr %ptr_unscaled unordered, align 4
; CHECK: sturh {{w[0-9]+}}, [x0, #-128]

  ret void
}

define void @atomic_store_relaxed_bf16(ptr %p, i32 %off32, i64 %off64, bfloat %val) #0 {
; CHECK-LABEL: atomic_store_relaxed_bf16:
  %ptr_unsigned = getelementptr bfloat, ptr %p, i32 4095
  store atomic bfloat %val, ptr %ptr_unsigned monotonic, align 4
; CHECK: strh {{w[0-9]+}}, [x0, #8190]

  %ptr_regoff = getelementptr bfloat, ptr %p, i32 %off32
  store atomic bfloat %val, ptr %ptr_regoff unordered, align 4
; CHECK: strh {{w[0-9]+}}, [x0, w1, sxtw #1]

  %ptr_regoff64 = getelementptr bfloat, ptr %p, i64 %off64
  store atomic bfloat %val, ptr %ptr_regoff64 monotonic, align 4
; CHECK: strh {{w[0-9]+}}, [x0, x2, lsl #1]

  %ptr_unscaled = getelementptr bfloat, ptr %p, i32 -64
  store atomic bfloat %val, ptr %ptr_unscaled unordered, align 4
; CHECK: sturh {{w[0-9]+}}, [x0, #-128]

  ret void
}

attributes #0 = { nounwind }
