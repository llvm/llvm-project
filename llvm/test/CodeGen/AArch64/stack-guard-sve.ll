; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

declare dso_local void @val_fn(<vscale x 4 x float>)
declare dso_local void @ptr_fn(<vscale x 4 x float>*)

; An alloca of a scalable vector shouldn't trigger stack protection.

; CHECK-LABEL: call_value:
; CHECK-NOT: mov x19, sp
; CHECK: addvl sp, sp, #-1
; CHECK-NOT: __stack_chk_guard
; CHECK: st1w { {{z[0-9]+.s}} }, {{p[0-9]+}}, [x29, #-1, mul vl]
define void @call_value() #0 {
entry:
  %x = alloca <vscale x 4 x float>, align 16
  store <vscale x 4 x float> zeroinitializer, <vscale x 4 x float>* %x, align 16
  %0 = load <vscale x 4 x float>, <vscale x 4 x float>* %x, align 16
  call void @val_fn(<vscale x 4 x float> %0)
  ret void
}

; CHECK-LABEL: call_value_strong:
; CHECK-NOT: mov x19, sp
; CHECK: addvl sp, sp, #-1
; CHECK-NOT: __stack_chk_guard
; CHECK: st1w { {{z[0-9]+.s}} }, {{p[0-9]+}}, [x29, #-1, mul vl]
define void @call_value_strong() #1 {
entry:
  %x = alloca <vscale x 4 x float>, align 16
  store <vscale x 4 x float> zeroinitializer, <vscale x 4 x float>* %x, align 16
  %0 = load <vscale x 4 x float>, <vscale x 4 x float>* %x, align 16
  call void @val_fn(<vscale x 4 x float> %0)
  ret void
}

; Address-taking of a scalable vector should trigger stack protection only with
; sspstrong, and the scalable vector should be be placed below the stack guard.

; CHECK-LABEL: call_ptr:
; CHECK-NOT: mov x19, sp
; CHECK: addvl sp, sp, #-1
; CHECK-NOT: __stack_chk_guard
; CHECK: addvl x0, x29, #-1
; CHECK: bl ptr_fn
define void @call_ptr() #0 {
entry:
  %x = alloca <vscale x 4 x float>, align 16
  call void @ptr_fn(<vscale x 4 x float>* %x)
  ret void
}

; CHECK-LABEL: call_ptr_strong:
; CHECK: mov x29, sp
; CHECK: addvl sp, sp, #-2
; CHECK-DAG: addvl [[ADDR:x[0-9]+]], x29, #-1
; CHECK-DAG: ldr [[VAL:x[0-9]+]], [{{x[0-9]+}}, :lo12:__stack_chk_guard]
; CHECK-DAG: str [[VAL]], [[[ADDR]]]
; CHECK-DAG: addvl x0, x29, #-2
; CHECK: bl ptr_fn
define void @call_ptr_strong() #1 {
entry:
  %x = alloca <vscale x 4 x float>, align 16
  call void @ptr_fn(<vscale x 4 x float>* %x)
  ret void
}

; Check that both variables are addressed in the same way

; CHECK-LABEL: call_both:
; CHECK: mov x29, sp
; CHECK: addvl sp, sp, #-2
; CHECK-NOT: __stack_chk_guard
; CHECK: st1w { {{z[0-9]+.s}} }, {{p[0-9]+}}, [x29, #-1, mul vl]
; CHECK: bl val_fn
; CHECK: addvl x0, x29, #-2
; CHECK: bl ptr_fn
define void @call_both() #0 {
entry:
  %x = alloca <vscale x 4 x float>, align 16
  %y = alloca <vscale x 4 x float>, align 16
  store <vscale x 4 x float> zeroinitializer, <vscale x 4 x float>* %x, align 16
  %0 = load <vscale x 4 x float>, <vscale x 4 x float>* %x, align 16
  call void @val_fn(<vscale x 4 x float> %0)
  call void @ptr_fn(<vscale x 4 x float>* %y)
  ret void
}

; CHECK-LABEL: call_both_strong:
; CHECK: mov x29, sp
; CHECK: addvl sp, sp, #-3
; CHECK-DAG: addvl [[ADDR:x[0-9]+]], x29, #-1
; CHECK-DAG: ldr [[VAL:x[0-9]+]], [{{x[0-9]+}}, :lo12:__stack_chk_guard]
; CHECK-DAG: str [[VAL]], [[[ADDR]]]
; CHECK-DAG: st1w { {{z[0-9]+.s}} }, {{p[0-9]+}}, [x29, #-2, mul vl]
; CHECK: bl val_fn
; CHECK: addvl x0, x29, #-3
; CHECK: bl ptr_fn
define void @call_both_strong() #1 {
entry:
  %x = alloca <vscale x 4 x float>, align 16
  %y = alloca <vscale x 4 x float>, align 16
  store <vscale x 4 x float> zeroinitializer, <vscale x 4 x float>* %x, align 16
  %0 = load <vscale x 4 x float>, <vscale x 4 x float>* %x, align 16
  call void @val_fn(<vscale x 4 x float> %0)
  call void @ptr_fn(<vscale x 4 x float>* %y)
  ret void
}

; Pushed callee-saved regs should be above the stack guard

; CHECK-LABEL: callee_save:
; CHECK: mov x29, sp
; CHECK: addvl sp, sp, #-18
; CHECK: str {{z[0-9]+}}, [sp, #{{[0-9]+}}, mul vl]
; CHECK-NOT: mov x29, sp
; CHECK: addvl sp, sp, #-1
; CHECK-NOT: __stack_chk_guard
; CHECK: addvl [[REG:x[0-9]+]], x29, #-11
; CHECK: st1w { {{z[0-9]+.s}} }, {{p[0-9]+}}, [[[REG]], #-8, mul vl]
define void @callee_save(<vscale x 4 x float> %x) #0 {
entry:
  %x.addr = alloca <vscale x 4 x float>, align 16
  store <vscale x 4 x float> %x, <vscale x 4 x float>* %x.addr, align 16
  call void @ptr_fn(<vscale x 4 x float>* %x.addr)
  ret void
}

; CHECK-LABEL: callee_save_strong:
; CHECK: mov x29, sp
; CHECK: addvl sp, sp, #-18
; CHECK: str {{z[0-9]+}}, [sp, #{{[0-9]+}}, mul vl]
; CHECK: addvl sp, sp, #-2
; CHECK-DAG: addvl [[ADDR:x[0-9]+]], x29, #-19
; CHECK-DAG: ldr [[VAL:x[0-9]+]], [{{x[0-9]+}}, :lo12:__stack_chk_guard]
; CHECK-DAG: str [[VAL]], [[[ADDR]]]
; CHECK-DAG: addvl [[ADDR2:x[0-9]+]], x29, #-12
; CHECK-DAG: st1w { z0.s }, p0, [[[ADDR2]], #-8, mul vl]
define void @callee_save_strong(<vscale x 4 x float> %x) #1 {
entry:
  %x.addr = alloca <vscale x 4 x float>, align 16
  store <vscale x 4 x float> %x, <vscale x 4 x float>* %x.addr, align 16
  call void @ptr_fn(<vscale x 4 x float>* %x.addr)
  ret void
}

; Check that local stack allocation works correctly both when we have a stack
; guard but no vulnerable SVE objects, and when we do have such objects.

; CHECK-LABEL: local_stack_alloc:
; CHECK: mov x29, sp
; CHECK: sub sp, sp, #16, lsl #12
; CHECK: sub sp, sp, #16
; CHECK: addvl sp, sp, #-2

; Stack guard is placed below the SVE stack area (and above all fixed-width objects)
; CHECK-DAG: add [[STACK_GUARD_SPILL_PART_LOC:x[0-9]+]], sp, #8, lsl #12
; CHECK-DAG: add [[STACK_GUARD_SPILL_PART_LOC]], [[STACK_GUARD_SPILL_PART_LOC]], #16
; CHECK-DAG: ldr [[STACK_GUARD:x[0-9]+]], [{{x[0-9]+}}, :lo12:__stack_chk_guard]
; CHECK-DAG: str [[STACK_GUARD]], [[[STACK_GUARD_SPILL_PART_LOC]], #32760]

; char_arr is below the stack guard
; CHECK-DAG: add [[CHAR_ARR_LOC:x[0-9]+]], sp, #16, lsl #12
; CHECK-DAG: strb wzr, [[[CHAR_ARR_LOC]]]

; large1 is accessed via a virtual base register
; CHECK-DAG: add [[LARGE1:x[0-9]+]], sp, #8, lsl #12
; CHECK-DAG: stp x0, x0, [[[LARGE1]]]

; large2 is at the bottom of the stack
; CHECK-DAG: stp x0, x0, [sp]

; vec1 and vec2 are in the SVE stack immediately below fp
; CHECK-DAG: addvl x0, x29, #-1
; CHECK-DAG: bl ptr_fn
; CHECK-DAG: addvl x0, x29, #-2
; CHECK-DAG: bl ptr_fn
define void @local_stack_alloc(i64 %val) #0 {
entry:
  %char_arr = alloca [8 x i8], align 4
  %gep0 = getelementptr [8 x i8], [8 x i8]* %char_arr, i64 0, i64 0
  store i8 0, i8* %gep0, align 8
  %large1 = alloca [4096 x i64], align 8
  %large2 = alloca [4096 x i64], align 8
  %vec_1 = alloca <vscale x 4 x float>, align 16
  %vec_2 = alloca <vscale x 4 x float>, align 16
  %gep1 = getelementptr [4096 x i64], [4096 x i64]* %large1, i64 0, i64 0
  %gep2 = getelementptr [4096 x i64], [4096 x i64]* %large1, i64 0, i64 1
  store i64 %val, i64* %gep1, align 8
  store i64 %val, i64* %gep2, align 8
  %gep3 = getelementptr [4096 x i64], [4096 x i64]* %large2, i64 0, i64 0
  %gep4 = getelementptr [4096 x i64], [4096 x i64]* %large2, i64 0, i64 1
  store i64 %val, i64* %gep3, align 8
  store i64 %val, i64* %gep4, align 8
  call void @ptr_fn(<vscale x 4 x float>* %vec_1)
  call void @ptr_fn(<vscale x 4 x float>* %vec_2)
  ret void
}

; CHECK-LABEL: local_stack_alloc_strong:
; CHECK: mov x29, sp
; CHECK: sub sp, sp, #16, lsl #12
; CHECK: sub sp, sp, #16
; CHECK: addvl sp, sp, #-3

; Stack guard is placed at the top of the SVE stack area
; CHECK-DAG: ldr [[STACK_GUARD:x[0-9]+]], [{{x[0-9]+}}, :lo12:__stack_chk_guard]
; CHECK-DAG: addvl [[STACK_GUARD_POS:x[0-9]+]], x29, #-1
; CHECK-DAG: str [[STACK_GUARD]], [[[STACK_GUARD_POS]]]

; char_arr is below the SVE stack area
; CHECK-DAG: add [[CHAR_ARR:x[0-9]+]], sp, #15, lsl #12            // =61440
; CHECK-DAG: add [[CHAR_ARR]], [[CHAR_ARR]], #9
; CHECK-DAG: strb wzr, [[[CHAR_ARR]], #4095]

; large1 is accessed via a virtual base register
; CHECK-DAG: add [[LARGE1:x[0-9]+]], sp, #8, lsl #12
; CHECK-DAG: stp x0, x0, [[[LARGE1]], #8]

; large2 is at the bottom of the stack
; CHECK-DAG: stp x0, x0, [sp, #8]

; vec1 and vec2 are in the SVE stack area below the stack guard
; CHECK-DAG: addvl x0, x29, #-2
; CHECK-DAG: bl ptr_fn
; CHECK-DAG: addvl x0, x29, #-3
; CHECK-DAG: bl ptr_fn
define void @local_stack_alloc_strong(i64 %val) #1 {
entry:
  %char_arr = alloca [8 x i8], align 4
  %gep0 = getelementptr [8 x i8], [8 x i8]* %char_arr, i64 0, i64 0
  store i8 0, i8* %gep0, align 8
  %large1 = alloca [4096 x i64], align 8
  %large2 = alloca [4096 x i64], align 8
  %vec_1 = alloca <vscale x 4 x float>, align 16
  %vec_2 = alloca <vscale x 4 x float>, align 16
  %gep1 = getelementptr [4096 x i64], [4096 x i64]* %large1, i64 0, i64 0
  %gep2 = getelementptr [4096 x i64], [4096 x i64]* %large1, i64 0, i64 1
  store i64 %val, i64* %gep1, align 8
  store i64 %val, i64* %gep2, align 8
  %gep3 = getelementptr [4096 x i64], [4096 x i64]* %large2, i64 0, i64 0
  %gep4 = getelementptr [4096 x i64], [4096 x i64]* %large2, i64 0, i64 1
  store i64 %val, i64* %gep3, align 8
  store i64 %val, i64* %gep4, align 8
  call void @ptr_fn(<vscale x 4 x float>* %vec_1)
  call void @ptr_fn(<vscale x 4 x float>* %vec_2)
  ret void
}

; A GEP addressing into a vector of <vscale x 4 x float> is in-bounds for
; offsets up to 3, but out-of-bounds (and so triggers stack protection with
; sspstrong) after that.

; CHECK-LABEL: vector_gep_3:
; CHECK-NOT: __stack_chk_guard
define void @vector_gep_3() #0 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 3
  store float 0.0, float* %gep, align 4
  ret void
}

; CHECK-LABEL: vector_gep_4:
; CHECK-NOT: __stack_chk_guard
define void @vector_gep_4() #0 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 4
  store float 0.0, float* %gep, align 4
  ret void
}

; CHECK-LABEL: vector_gep_twice:
; CHECK-NOT: __stack_chk_guard
define void @vector_gep_twice() #0 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep1 = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 3
  store float 0.0, float* %gep1, align 4
  %gep2 = getelementptr float, float* %gep1, i64 1
  store float 0.0, float* %gep2, align 4
  ret void
}

; CHECK-LABEL: vector_gep_n:
; CHECK-NOT: __stack_chk_guard
define void @vector_gep_n(i64 %n) #0 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 %n
  store float 0.0, float* %gep, align 4
  ret void
}

; CHECK-LABEL: vector_gep_3_strong:
; CHECK-NOT: __stack_chk_guard
define void @vector_gep_3_strong() #1 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 3
  store float 0.0, float* %gep, align 4
  ret void
}

; CHECK-LABEL: vector_gep_4_strong:
; CHECK: __stack_chk_guard
define void @vector_gep_4_strong(i64 %val) #1 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 4
  store float 0.0, float* %gep, align 4
  ret void
}


; CHECK-LABEL: vector_gep_twice_strong:
; CHECK: __stack_chk_guard
define void @vector_gep_twice_strong() #1 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep1 = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 3
  store float 0.0, float* %gep1, align 4
  %gep2 = getelementptr float, float* %gep1, i64 1
  store float 0.0, float* %gep2, align 4
  ret void
}

; CHECK-LABEL: vector_gep_n_strong:
; CHECK: __stack_chk_guard
define void @vector_gep_n_strong(i64 %n) #1 {
entry:
  %vec = alloca <vscale x 4 x float>, align 16
  %gep = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %vec, i64 0, i64 %n
  store float 0.0, float* %gep, align 4
  ret void
}

attributes #0 = { ssp "frame-pointer"="non-leaf" }
attributes #1 = { sspstrong "frame-pointer"="non-leaf" }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"direct-access-external-data", i32 1}
