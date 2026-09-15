; Verifies that we materialize intrinsics across suspend points
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; See that we only spilled one value for f
; Check other variants where different levels of materialization are achieved

target datalayout = "e-m:e-p:64:64-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @f_fabs(
; CHECK: malloc(i32 24)
; CHECK: store float %n
; CHECK-NOT: store float %abs

; CHECK-LABEL: @f_fabs_optnone(
; CHECK: malloc(i32 32)
; CHECK: store float %abs

; CHECK-LABEL: @f_ctpop(
; CHECK: malloc(i32 24)
; CHECK: store i32 %n
; CHECK-NOT: store i32 %pop

; CHECK-LABEL: @f_ctpop_optnone(
; CHECK: malloc(i32 32)
; CHECK: store i32 %pop

; CHECK-LABEL: @f_floor(
; CHECK: malloc(i32 24)
; CHECK: store float %n
; CHECK-NOT: store float %res

; CHECK-LABEL: @f_floor_optnone(
; CHECK: malloc(i32 32)
; CHECK: store float %res

; CHECK-LABEL: @f_minnum(
; CHECK: malloc(i32 24)
; CHECK: store float %n
; CHECK-NOT: store float %res

; CHECK-LABEL: @f_minnum_optnone(
; CHECK: malloc(i32 32)
; CHECK: store float %res

; CHECK-LABEL: @f_fptosi_sat(
; CHECK: malloc(i32 24)
; CHECK: store float %n
; CHECK-NOT: store i32 %res

; CHECK-LABEL: @f_fptosi_sat_optnone(
; CHECK: malloc(i32 32)
; CHECK: store i32 %res

; CHECK-LABEL: @f_sadd_sat(
; CHECK: malloc(i32 24)
; CHECK: store i32 %n
; CHECK-NOT: store i32 %res

; CHECK-LABEL: @f_sadd_sat_optnone(
; CHECK: malloc(i32 32)
; CHECK: store i32 %res

; CHECK-LABEL: @f_smax(
; CHECK: malloc(i32 24)
; CHECK: store i32 %n
; CHECK-NOT: store i32 %res

; CHECK-LABEL: @f_smax_optnone(
; CHECK: malloc(i32 32)
; CHECK: store i32 %res

; CHECK-LABEL: @f_fabs.resume(
; CHECK: %n.reload{{.*}} = load float
; CHECK: call float @llvm.fabs.f32(float %n.reload{{.*}})

; CHECK-LABEL: @f_fabs_optnone.resume(
; CHECK: %abs.reload{{.*}} = load float, ptr
; CHECK: store float {{.*}}, ptr

; CHECK-LABEL: @f_ctpop.resume(
; CHECK: %n.reload{{.*}} = load i32
; CHECK: call i32 @llvm.ctpop.i32(i32 %n.reload{{.*}})

; CHECK-LABEL: @f_ctpop_optnone.resume(
; CHECK: %pop.reload{{.*}} = load i32, ptr
; CHECK: store i32 {{.*}}, ptr

; CHECK-LABEL: @f_floor.resume(
; CHECK: %n.reload{{.*}} = load float
; CHECK: call float @llvm.floor.f32(float %n.reload{{.*}})

; CHECK-LABEL: @f_floor_optnone.resume(
; CHECK: %res.reload{{.*}} = load float, ptr
; CHECK: store float {{.*}}, ptr

; CHECK-LABEL: @f_minnum.resume(
; CHECK: %n.reload{{.*}} = load float
; CHECK: call float @llvm.minnum.f32(float %n.reload{{.*}}

; CHECK-LABEL: @f_minnum_optnone.resume(
; CHECK: %res.reload{{.*}} = load float, ptr
; CHECK: store float {{.*}}, ptr

; CHECK-LABEL: @f_fptosi_sat.resume(
; CHECK: %n.reload{{.*}} = load float
; CHECK: call i32 @llvm.fptosi.sat.i32.f32(float %n.reload{{.*}})

; CHECK-LABEL: @f_fptosi_sat_optnone.resume(
; CHECK: %res.reload{{.*}} = load i32, ptr
; CHECK: store i32 {{.*}}, ptr

; CHECK-LABEL: @f_sadd_sat.resume(
; CHECK: %n.reload{{.*}} = load i32
; CHECK: call i32 @llvm.sadd.sat.i32(i32 %n.reload{{.*}}

; CHECK-LABEL: @f_sadd_sat_optnone.resume(
; CHECK: %res.reload{{.*}} = load i32, ptr
; CHECK: store i32 {{.*}}, ptr

; CHECK-LABEL: @f_smax.resume(
; CHECK: %n.reload{{.*}} = load i32
; CHECK: call i32 @llvm.smax.i32(i32 %n.reload{{.*}}

; CHECK-LABEL: @f_smax_optnone.resume(
; CHECK: %res.reload{{.*}} = load i32, ptr
; CHECK: store i32 {{.*}}, ptr

define ptr @f_fabs(float %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_fabs, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %abs = call float @llvm.fabs.f32(float %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = fadd float %abs, 1.0
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_f32(float %abs)
  call void @print_f32(float %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_fabs_optnone(float %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_fabs_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %abs = call float @llvm.fabs.f32(float %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = fadd float %abs, 1.0
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_f32(float %abs)
  call void @print_f32(float %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_ctpop(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_ctpop, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %pop = call i32 @llvm.ctpop.i32(i32 %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %pop, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %pop)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_ctpop_optnone(i32 %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_ctpop_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %pop = call i32 @llvm.ctpop.i32(i32 %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %pop, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %pop)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_floor(float %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_floor, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call float @llvm.floor.f32(float %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = fadd float %res, 1.0
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_f32(float %res)
  call void @print_f32(float %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_floor_optnone(float %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_floor_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call float @llvm.floor.f32(float %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = fadd float %res, 1.0
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_f32(float %res)
  call void @print_f32(float %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_minnum(float %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_minnum, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call float @llvm.minnum.f32(float %n, float 1.0)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = fadd float %res, 1.0
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_f32(float %res)
  call void @print_f32(float %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_minnum_optnone(float %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_minnum_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call float @llvm.minnum.f32(float %n, float 1.0)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = fadd float %res, 1.0
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_f32(float %res)
  call void @print_f32(float %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_fptosi_sat(float %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_fptosi_sat, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call i32 @llvm.fptosi.sat.i32.f32(float %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %res, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %res)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_fptosi_sat_optnone(float %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_fptosi_sat_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call i32 @llvm.fptosi.sat.i32.f32(float %n)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %res, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %res)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_sadd_sat(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_sadd_sat, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call i32 @llvm.sadd.sat.i32(i32 %n, i32 1)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %res, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %res)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_sadd_sat_optnone(i32 %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_sadd_sat_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call i32 @llvm.sadd.sat.i32(i32 %n, i32 1)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %res, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %res)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_smax(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_smax, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call i32 @llvm.smax.i32(i32 %n, i32 1)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %res, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %res)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_smax_optnone(i32 %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_smax_optnone, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %res = call i32 @llvm.smax.i32(i32 %n, i32 1)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %add = add i32 %res, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]
resume2:
  call void @print_i32(i32 %res)
  call void @print_i32(i32 %add)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)
declare float @llvm.fabs.f32(float)
declare i32 @llvm.ctpop.i32(i32)
declare float @llvm.floor.f32(float)
declare float @llvm.minnum.f32(float, float)
declare i32 @llvm.fptosi.sat.i32.f32(float)
declare i32 @llvm.sadd.sat.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare noalias ptr @malloc(i32)
declare void @print_f32(float)
declare void @print_i32(i32)
declare void @free(ptr)
