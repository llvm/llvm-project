; RUN: llc -mtriple=hexagon < %s | FileCheck %s

;; Small frame (< probe size): no probing loop, normal allocframe.
; CHECK-LABEL: small_frame:
; CHECK: allocframe(r29,#128):raw
; CHECK-NOT: cmp.gtu
; CHECK: dealloc_return
define void @small_frame() #0 {
entry:
  %a = alloca [128 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; Large frame (> probe size): probing loop emitted.
; CHECK-LABEL: large_frame:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,#-8192)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
define void @large_frame() #0 {
entry:
  %a = alloca [8192 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; Exact multiple of probe size: probing loop still emitted.
; CHECK-LABEL: exact_multiple:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,#-12288)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
define void @exact_multiple() #0 {
entry:
  %a = alloca [12288 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; No frame pointer path with no call: probing works without allocframe.
; CHECK-LABEL: no_fp_large:
; CHECK-NOT: allocframe
; CHECK: r28 = add(r29,#-8192)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
define void @no_fp_large() #1 {
entry:
  %a = alloca [8192 x i8], align 1
  store volatile i8 0, ptr %a
  ret void
}

;; Custom probe size of 512 bytes.
; CHECK-LABEL: custom_probe_size:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,#-8192)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-512)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
define void @custom_probe_size() #2 {
entry:
  %a = alloca [8192 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; No probe attribute: normal codegen, no probing.
; CHECK-LABEL: no_probe:
; CHECK: allocframe
; CHECK-NOT: cmp.gtu
; CHECK: dealloc_return
define void @no_probe() {
entry:
  %a = alloca [8192 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; Frame >= ALLOCFRAME_MAX (16384): allocframe(#0) + probed alloc.
; CHECK-LABEL: very_large_frame:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,#-20480)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
define void @very_large_frame() #0 {
entry:
  %a = alloca [20480 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; Frame == probe size exactly: no probing loop, normal allocframe.
; CHECK-LABEL: exact_probe_size:
; CHECK: allocframe(r29,#4096):raw
; CHECK-NOT: cmp.gtu
; CHECK: dealloc_return
define void @exact_probe_size() #0 {
entry:
  %a = alloca [4096 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; Large frame requiring constant-extended immediate (> 32767).
; CHECK-LABEL: const_extd_frame:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,##-65536)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
define void @const_extd_frame() #0 {
entry:
  %a = alloca [65536 x i8], align 1
  call void @use(ptr %a)
  ret void
}

;; Callee-saved register spills coexist with probing.
; CHECK-LABEL: callee_saved_regs:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,#-8216)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
; CHECK: memd(r29+##{{[0-9]+}}) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK: dealloc_return
define void @callee_saved_regs(ptr %p) #0 {
entry:
  %a = alloca [8192 x i8], align 1
  call void @use(ptr %a)
  call void asm sideeffect "", "~{r16},~{r17},~{r18},~{r19}"()
  call void @use(ptr %p)
  ret void
}

;; VLA (dynamic alloca): probe loop emitted for runtime-sized allocation.
; CHECK-LABEL: vla_basic:
; CHECK: allocframe
; CHECK: r[[TGT:[0-9]+]] = sub(r29,r{{[0-9]+}})
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r[[TGT]])
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r[[TGT]]
define void @vla_basic(i32 %n) #0 {
entry:
  %sz = zext i32 %n to i64
  %buf = alloca i8, i64 %sz, align 1
  call void @use(ptr %buf)
  ret void
}

;; VLA without probe attribute: plain sub(r29,Rs), no probe loop.
; CHECK-LABEL: vla_no_probe:
; CHECK: sub(r29,r
; CHECK-NOT: cmp.gtu
; CHECK: dealloc_return
define void @vla_no_probe(i32 %n) {
entry:
  %sz = zext i32 %n to i64
  %buf = alloca i8, i64 %sz, align 1
  call void @use(ptr %buf)
  ret void
}

;; Static large frame + VLA: two separate probe loops are emitted.
; CHECK-LABEL: vla_and_static:
; CHECK: allocframe(r29,#0):raw
; CHECK: r28 = add(r29,#-8200)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r28)
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r28
; CHECK: r[[TGT2:[0-9]+]] = sub(r29,r{{[0-9]+}})
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r[[TGT2]])
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r[[TGT2]]
define void @vla_and_static(i32 %n) #0 {
entry:
  %static = alloca [8192 x i8], align 1
  %sz = zext i32 %n to i64
  %dyn = alloca i8, i64 %sz, align 1
  call void @use2(ptr %static, ptr %dyn)
  ret void
}

;; VLA with alignment > 8: target pointer is aligned before probing.
; CHECK-LABEL: vla_align16:
; CHECK: allocframe
; CHECK: r[[TGT3:[0-9]+]] = sub(r29,r{{[0-9]+}})
; CHECK: r[[TGT3]] = and(r[[TGT3]],#-16)
; CHECK: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK:      {
; CHECK-NEXT:   r29 = add(r29,#-4096)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   p0 = cmp.gtu(r29,r[[TGT3]])
; CHECK-NEXT:   if (p0.new) jump:t .LBB
; CHECK-NEXT:   memw(r29+#0) = #0
; CHECK-NEXT: }
; CHECK: r29 = r[[TGT3]]
define void @vla_align16(i32 %n) #0 {
entry:
  %sz = zext i32 %n to i64
  %buf = alloca i8, i64 %sz, align 16
  call void @use(ptr %buf)
  ret void
}

declare void @use(ptr)
declare void @use2(ptr, ptr)

attributes #0 = { "probe-stack"="inline-asm" }
attributes #1 = { nounwind "frame-pointer"="none" "probe-stack"="inline-asm" }
attributes #2 = { "probe-stack"="inline-asm" "stack-probe-size"="512" }
