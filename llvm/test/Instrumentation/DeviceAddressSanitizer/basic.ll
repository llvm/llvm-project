; RUN: opt < %s -passes=dasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; CHECK: @__dasan_no_entry = private addrspace(1) constant i64 0
; CHECK: @__dasan_instrumented = linkonce_odr protected addrspace(1) constant i8 1

define i32 @load4(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define i32 @load4(
; CHECK-NEXT:    [[ADDR:%.*]] = ptrtoint ptr addrspace(1) %a to i64
; CHECK-NEXT:    [[TOP:%.*]] = lshr i64 [[ADDR]], 45
; CHECK-NEXT:    [[MINE:%.*]] = icmp eq i64 [[TOP]], 1
; CHECK-NEXT:    [[CLS:%.*]] = lshr i64 [[ADDR]], 40
; CHECK-NEXT:    [[CLSM:%.*]] = and i64 [[CLS]], 31
; CHECK-NEXT:    [[SHIFT:%.*]] = add i64 [[CLSM]], 6
; CHECK-NEXT:    [[INREG:%.*]] = and i64 [[ADDR]], 1099511627775
; CHECK-NEXT:    [[IDX:%.*]] = lshr i64 [[INREG]], [[SHIFT]]
; CHECK-NEXT:    [[BYTE:%.*]] = shl i64 [[IDX]], 3
; CHECK-NEXT:    [[SLOT:%.*]] = or i64 [[BYTE]], 7
; CHECK-NEXT:    [[END:%.*]] = or i64 [[ADDR]], 1099511627775
; CHECK-NEXT:    [[EA:%.*]] = sub i64 [[END]], [[SLOT]]
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[MINE]], i64 [[EA]], i64 ptrtoint (ptr addrspace(1) @__dasan_no_entry to i64)
; CHECK-NEXT:    [[EP:%.*]] = inttoptr i64 [[SEL]] to ptr addrspace(1)
; CHECK-NEXT:    [[ENTRY:%.*]] = load i64, ptr addrspace(1) [[EP]], align 8, !alias.scope ![[SCOPE:[0-9]+]], !nosanitize
; CHECK-NEXT:    [[SIZE:%.*]] = and i64 [[ENTRY]], 137438953471
; CHECK-NEXT:    [[OFFW:%.*]] = and i64 [[ENTRY]], -274877906944
; CHECK-NEXT:    [[OFF:%.*]] = ashr i64 [[OFFW]], 32
; CHECK-NEXT:    [[CBASE:%.*]] = shl i64 [[IDX]], [[SHIFT]]
; CHECK-NEXT:    [[COFF:%.*]] = sub i64 [[INREG]], [[CBASE]]
; CHECK-NEXT:    [[INOBJ:%.*]] = sub i64 [[COFF]], [[OFF]]
; CHECK:         [[START:%.*]] = sub i64 [[ADDR]], [[INOBJ]]
; CHECK:         [[LAST:%.*]] = add i64 [[INOBJ]], 3
; CHECK-NEXT:    [[OOB:%.*]] = icmp uge i64 [[LAST]], [[SIZE]]
; CHECK-NEXT:    [[BAD:%.*]] = and i1 [[MINE]], [[OOB]]
; CHECK-NEXT:    br i1 [[BAD]], label %[[FAIL:.*]], label %[[GO:.*]], !prof
; CHECK:       [[FAIL]]:
; CHECK:         notail call void @__dasan_report_load(ptr {{.*}}, i64 4, i64 [[START]], i64 [[SIZE]]) #[[SITE:[0-9]+]]
; CHECK-NEXT:    unreachable
; CHECK:       [[GO]]:
; CHECK-NEXT:    %v = load i32, ptr addrspace(1) %a, align 4, !noalias ![[SCOPE]]
  %v = load i32, ptr addrspace(1) %a, align 4
  ret i32 %v
}

define void @store4_unaligned(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define void @store4_unaligned(
; CHECK:         [[L:%.*]] = add i64 [[F:%.*]], 3
; CHECK-NEXT:    [[M:%.*]] = call i64 @llvm.umax.i64(i64 [[F]], i64 [[L]])
; CHECK-NEXT:    icmp uge i64 [[M]], %{{.*}}
; CHECK:         call void @__dasan_report_store(
  store i32 0, ptr addrspace(1) %a, align 1
  ret void
}

define void @store64_aligned(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define void @store64_aligned(
; CHECK:         [[L:%.*]] = add i64 %{{.*}}, 63
; CHECK-NEXT:    icmp uge i64 [[L]], %{{.*}}
; CHECK:         store <16 x i32>
  store <16 x i32> zeroinitializer, ptr addrspace(1) %a, align 64
  ret void
}

define void @store128_aligned(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define void @store128_aligned(
; CHECK:         [[L:%.*]] = add i64 [[F:%.*]], 127
; CHECK-NEXT:    [[M:%.*]] = call i64 @llvm.umax.i64(i64 [[F]], i64 [[L]])
; CHECK-NEXT:    icmp uge i64 [[M]], %{{.*}}
  store <32 x i32> zeroinitializer, ptr addrspace(1) %a, align 128
  ret void
}

define void @other_addrspace(ptr addrspace(3) %a) sanitize_device_address {
; CHECK-LABEL: define void @other_addrspace(
; CHECK-NEXT:    [[P:%.*]] = call{{.*}} ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[G:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[P]], i64 28
; CHECK-NEXT:    [[LIM:%.*]] = load i32, ptr addrspace(4) [[G]], align 4, !invariant.load
; CHECK:         icmp uge i32 %{{.*}}, [[LIM]]
; CHECK:         call void @__dasan_report_shared_store(
; CHECK:         store i32 0, ptr addrspace(3) %{{.*}}, align 4
  store i32 0, ptr addrspace(3) %a, align 4
  ret void
}

define void @loop(ptr addrspace(1) %a, i32 %n) sanitize_device_address {
; CHECK-LABEL: define void @loop(
; CHECK:       body:
; CHECK-NEXT:    %i = phi i32
; CHECK-NEXT:    %p = getelementptr
; CHECK-NEXT:    [[BASE:%.*]] = ptrtoint ptr addrspace(1) %a to i64
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK:         call void @__dasan_report_store(
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  store i32 1, ptr addrspace(1) %p, align 4
  br label %latch
latch:
  %next = add i32 %i, 1
  %c = icmp slt i32 %next, %n
  br i1 %c, label %body, label %out
out:
  ret void
}

; RUN: opt < %s -passes='dasan,function(loop-mssa(licm))' -S \
; RUN:   | FileCheck %s --check-prefix=LICM
; LICM-LABEL: define void @loop(
; LICM:       entry:
; LICM:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; LICM:       body:
; LICM-NOT:     load i64
; LICM:         call void @__dasan_report_store(

; RUN: opt < %s -passes='dasan,function(loop-mssa(licm))' -dasan-recover -S \
; RUN:   | FileCheck %s --check-prefix=REC-LICM
; The recover handler returns, so it is a call in the loop and the load stays.
; REC-LICM-LABEL: define void @loop(
; REC-LICM:       body:
; REC-LICM:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; REC-LICM:         call void @__dasan_report_store_noabort(

declare void @mutate()
declare void @peek() readonly nounwind

; The decode is in the header, ahead of the call, so it is free to leave the
; loop. mutate may store the entry word, so the load stays. peek only reads.
define void @loop_call(ptr addrspace(1) %a, i32 %n) sanitize_device_address {
; CHECK-LABEL: define void @loop_call(
; CHECK:       body:
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK:         call void @mutate()
; LICM-LABEL: define void @loop_call(
; LICM:       body:
; LICM:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; LICM:         call void @mutate()
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  store i32 %i, ptr addrspace(1) %a, align 4
  call void @mutate()
  %next = add i32 %i, 1
  %c = icmp slt i32 %next, %n
  br i1 %c, label %body, label %out
out:
  ret void
}

define void @loop_peek(ptr addrspace(1) %a, i32 %n) sanitize_device_address {
; CHECK-LABEL: define void @loop_peek(
; CHECK:       body:
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK:         call void @peek()
; LICM-LABEL: define void @loop_peek(
; LICM:       entry:
; LICM:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; LICM:       body:
; LICM-NOT:     load i64
; LICM:         call void @peek()
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  store i32 %i, ptr addrspace(1) %a, align 4
  call void @peek()
  %next = add i32 %i, 1
  %c = icmp slt i32 %next, %n
  br i1 %c, label %body, label %out
out:
  ret void
}

declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()

define i32 @dims() sanitize_device_address {
; CHECK-LABEL: define i32 @dims(
; CHECK-NEXT:    %p = call
; CHECK-NEXT:    %g = getelementptr
; CHECK-NEXT:    %v = load i32, ptr addrspace(4) %g
; CHECK-NEXT:    ret i32 %v
  %p = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %g = getelementptr i32, ptr addrspace(4) %p, i64 3
  %v = load i32, ptr addrspace(4) %g, align 4
  ret i32 %v
}

define void @no_attribute(ptr addrspace(1) %a) {
; CHECK-LABEL: define void @no_attribute(
; CHECK-NEXT:    store i32 0, ptr addrspace(1) %a, align 4
; CHECK-NEXT:    ret void
  store i32 0, ptr addrspace(1) %a, align 4
  ret void
}

; RUN: opt < %s -passes=dasan,gvn -S | FileCheck %s --check-prefix=GVN
define void @bump(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define void @bump(
; CHECK-COUNT-1: load i64, ptr addrspace(1)
; CHECK-NOT:     load i64, ptr addrspace(1)
; CHECK:         ret void
; GVN-LABEL: define void @bump(
; GVN:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; GVN-NOT:     load i64
; GVN:         ret void
  %v = load i32, ptr addrspace(1) %a, align 4
  %w = add i32 %v, 1
  store i32 %w, ptr addrspace(1) %a, align 4
  ret void
}

define void @notoptimized(ptr addrspace(1) %a) sanitize_device_address optnone noinline {
; CHECK-LABEL: define void @notoptimized(
; CHECK:         load i64, ptr addrspace(1)
; CHECK:         call void @__dasan_report_store
  store i32 1, ptr addrspace(1) %a, align 4
  ret void
}

@g = addrspace(1) global [4 x i32] zeroinitializer

define void @named(i64 %i) sanitize_device_address {
; CHECK-LABEL: define void @named(
; CHECK:         lshr i64 %{{.*}}, 45
; CHECK:         [[A:%.*]] = ptrtoint ptr addrspace(1) %p to i64
; CHECK:         call void @__dasan_report_store(
; CHECK:         store i32 1, ptr addrspace(1) %p
; CHECK-NEXT:    %q = alloca
; CHECK-NEXT:    %r = addrspacecast
; CHECK-NEXT:    store i32 2, ptr %r
; CHECK-NEXT:    ret void
  %p = getelementptr [4 x i32], ptr addrspace(1) @g, i64 0, i64 %i
  store i32 1, ptr addrspace(1) %p, align 4
  %q = alloca i32, align 4, addrspace(5)
  %r = addrspacecast ptr addrspace(5) %q to ptr
  store i32 2, ptr %r, align 4
  ret void
}

define void @constant_indices(i64 %i) sanitize_device_address {
; CHECK-LABEL: define void @constant_indices(
; CHECK:         %inside = getelementptr
; CHECK-NEXT:    store i32 1, ptr addrspace(1) %inside, align 4, !noalias
; CHECK:         %past = getelementptr
; CHECK:         call void @__dasan_report_store(
; CHECK:         store i32 2, ptr addrspace(1) %past
; CHECK:         %var = getelementptr
; CHECK:         call void @__dasan_report_store(
; CHECK:         store i32 3, ptr addrspace(1) %var
  %inside = getelementptr [4 x i32], ptr addrspace(1) @g, i64 0, i64 3
  store i32 1, ptr addrspace(1) %inside, align 4
  %past = getelementptr [4 x i32], ptr addrspace(1) @g, i64 0, i64 4
  store i32 2, ptr addrspace(1) %past, align 4
  %var = getelementptr [4 x i32], ptr addrspace(1) @g, i64 0, i64 %i
  store i32 3, ptr addrspace(1) %var, align 4
  ret void
}

define void @widen(ptr addrspace(1) %p, i64 %i) sanitize_device_address {
; CHECK-LABEL: define void @widen(
; CHECK:         call void @__dasan_report_store(
; CHECK:         store i32 0, ptr addrspace(1) %a
; CHECK:         store i32 1, ptr addrspace(1) %b
; CHECK:         store i32 2, ptr addrspace(1) %c
; CHECK:         store i32 3, ptr addrspace(1) %d
; CHECK-NEXT:    ret void
  %a = getelementptr i32, ptr addrspace(1) %p, i64 %i
  store i32 0, ptr addrspace(1) %a, align 4
  %j = add i64 %i, 1
  %b = getelementptr i32, ptr addrspace(1) %p, i64 %j
  store i32 1, ptr addrspace(1) %b, align 4
  %k = add i64 %i, 2
  %c = getelementptr i32, ptr addrspace(1) %p, i64 %k
  store i32 2, ptr addrspace(1) %c, align 4
  %l = add i64 %i, 3
  %d = getelementptr i32, ptr addrspace(1) %p, i64 %l
  store i32 3, ptr addrspace(1) %d, align 4
  ret void
}

define void @chased(ptr addrspace(1) %p) sanitize_device_address {
; CHECK-LABEL: define void @chased(
; CHECK:         call void @__dasan_report_store(
; CHECK:         store i32 0, ptr addrspace(1) %p
; CHECK:         call void @__dasan_report_load(
; CHECK:         %q = load ptr addrspace(1), ptr addrspace(1) %p
; CHECK:         call void @__dasan_report_store(
; CHECK:         store i32 1, ptr addrspace(1) %q
  store i32 0, ptr addrspace(1) %p, align 8
  %q = load ptr addrspace(1), ptr addrspace(1) %p, align 8
  store i32 1, ptr addrspace(1) %q, align 4
  ret void
}

define void @across_call(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define void @across_call(
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK-NOT:     !alias.scope
; CHECK:         call void @mutate()
; CHECK:         ret void
  store i32 1, ptr addrspace(1) %a, align 4
  call void @mutate()
  store i32 2, ptr addrspace(1) %a, align 4
  ret void
}

define void @across_peek(ptr addrspace(1) %a) sanitize_device_address {
; CHECK-LABEL: define void @across_peek(
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK:         call void @peek()
; CHECK-NOT:     load i64, ptr addrspace(1)
; CHECK:         ret void
  store i32 1, ptr addrspace(1) %a, align 4
  call void @peek()
  store i32 2, ptr addrspace(1) %a, align 4
  ret void
}

define void @across_raw(ptr addrspace(1) %a, ptr addrspace(1) %w) sanitize_device_address {
; CHECK-LABEL: define void @across_raw(
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK:         store i64 0, ptr addrspace(1) %w, align 8, !nosanitize !{{[0-9]+}}{{$}}
; CHECK:         ret void
  store i32 1, ptr addrspace(1) %a, align 4
  store i64 0, ptr addrspace(1) %w, align 8, !nosanitize !{}
  store i32 2, ptr addrspace(1) %a, align 4
  ret void
}

define void @only_safe() sanitize_device_address {
; CHECK-LABEL: define void @only_safe(
; CHECK-NEXT:    store i32 1, ptr addrspace(1) @g, align 4
; CHECK-NEXT:    ret void
  store i32 1, ptr addrspace(1) getelementptr inbounds ([4 x i32], ptr addrspace(1) @g, i64 0, i64 0), align 4
  ret void
}

; A provably-inbounds store is not checked, but it carries the scope so the
; decode still hoists out of the loop.
define void @loop_safe(ptr addrspace(1) %a, i32 %n) sanitize_device_address {
; CHECK-LABEL: define void @loop_safe(
; CHECK:         store i32 0, ptr addrspace(1) @g, align 4, !noalias ![[SAFE:[0-9]+]]
; CHECK:         load i64, ptr addrspace(1) {{.*}}, !alias.scope ![[SAFE]]
; LICM-LABEL: define void @loop_safe(
; LICM:       entry:
; LICM:         load i64, ptr addrspace(1) {{.*}}, !alias.scope
; LICM:       body:
; LICM-NOT:     load i64
; LICM:         ret void
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  store i32 0, ptr addrspace(1) getelementptr inbounds ([4 x i32], ptr addrspace(1) @g, i64 0, i64 0), align 4
  store i32 %i, ptr addrspace(1) %a, align 4
  %next = add i32 %i, 1
  %c = icmp slt i32 %next, %n
  br i1 %c, label %body, label %out
out:
  ret void
}

define void @two_blocks(ptr addrspace(1) %a, i1 %c) sanitize_device_address {
; CHECK-LABEL: define void @two_blocks(
; CHECK-COUNT-2: load i64, ptr addrspace(1) {{.*}}, !alias.scope
; CHECK:         ret void
entry:
  br i1 %c, label %left, label %right
left:
  store i32 1, ptr addrspace(1) %a, align 4
  ret void
right:
  store i32 2, ptr addrspace(1) %a, align 4
  ret void
}

; RUN: opt < %s -passes=dasan -dasan-recover -S | FileCheck %s --check-prefix=REC
; REC-LABEL: define i32 @load4(
; REC:         notail call void @__dasan_report_load_noabort(
; REC-NEXT:    br label %[[GO:.*]]
; REC:       [[GO]]:
; REC-NEXT:    %v = load i32, ptr addrspace(1) %a, align 4
; REC-LABEL: define void @store4_unaligned(
; REC:         call void @__dasan_report_store_noabort(
; REC-NOT:     noreturn
; REC:       attributes #{{[0-9]+}} = { cold nounwind }

; RUN: opt < %s -passes=dasan -dasan-trap-on-error -S | FileCheck %s --check-prefix=TRAP
; TRAP-LABEL: define i32 @load4(
; TRAP:         %v = load i32, ptr addrspace(1) %a, align 4
; TRAP:         call void @llvm.trap()
; TRAP-NEXT:    unreachable
; TRAP-NOT:   @__dasan_report

; RUN: opt < %s -passes=dasan -dasan-recover -dasan-trap-on-error -S \
; RUN:   | FileCheck %s --check-prefix=TRAP

; CHECK: attributes #{{[0-9]+}} = { cold noreturn nounwind }
; CHECK: attributes #[[SITE]] = { noinline nomerge }
; CHECK: [[PROF:![0-9]+]] = !{!"branch_weights", i32 1, i32 1048575}
