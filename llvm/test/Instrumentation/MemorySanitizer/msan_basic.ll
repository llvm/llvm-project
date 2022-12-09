; RUN: opt < %s -msan-check-access-address=0 -S -passes='module(msan)' 2>&1 | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=CHECK,NOORIGINS --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s --passes='module(msan)' -msan-check-access-address=0 -S | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=CHECK,NOORIGINS --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes='module(msan)' 2>&1 | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CHECK,ORIGINS %s --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -passes='module(msan)' -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CHECK,ORIGINS %s --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -passes='module(msan)' -msan-instrumentation-with-call-threshold=0 -msan-track-origins=1 -S | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CHECK-CALLS %s --implicit-check-not="call void @__msan_warning"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @llvm.used = appending global [1 x ptr] [ptr @msan.module_ctor]
; CHECK: @llvm.global_ctors {{.*}} { i32 0, ptr @msan.module_ctor, ptr null }

; Check the presence and the linkage type of __msan_track_origins and
; other interface symbols.
; CHECK-NOT: @__msan_track_origins
; ORIGINS: @__msan_track_origins = weak_odr constant i32 1
; CHECK-NOT: @__msan_keep_going = weak_odr constant i32 0
; CHECK: @__msan_retval_tls = external thread_local(initialexec) global [{{.*}}]
; CHECK: @__msan_retval_origin_tls = external thread_local(initialexec) global i32
; CHECK: @__msan_param_tls = external thread_local(initialexec) global [{{.*}}]
; CHECK: @__msan_param_origin_tls = external thread_local(initialexec) global [{{.*}}]
; CHECK: @__msan_va_arg_tls = external thread_local(initialexec) global [{{.*}}]
; CHECK: @__msan_va_arg_overflow_size_tls = external thread_local(initialexec) global i64


; Check instrumentation of stores

define void @Store(ptr nocapture %p, i32 %x) nounwind uwtable sanitize_memory {
entry:
  store i32 %x, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @Store
; CHECK: load {{.*}} @__msan_param_tls
; ORIGINS: load {{.*}} @__msan_param_origin_tls
; CHECK: store
; ORIGINS: icmp
; ORIGINS: br i1
; ORIGINS: {{^[0-9]+}}:
; ORIGINS: store
; ORIGINS: br label
; ORIGINS: {{^[0-9]+}}:
; CHECK: store
; CHECK: ret void


; Check instrumentation of aligned stores
; Shadow store has the same alignment as the original store; origin store
; does not specify explicit alignment.

define void @AlignedStore(ptr nocapture %p, i32 %x) nounwind uwtable sanitize_memory {
entry:
  store i32 %x, ptr %p, align 32
  ret void
}

; CHECK-LABEL: @AlignedStore
; CHECK: load {{.*}} @__msan_param_tls
; ORIGINS: load {{.*}} @__msan_param_origin_tls
; CHECK: store {{.*}} align 32
; ORIGINS: icmp
; ORIGINS: br i1
; ORIGINS: {{^[0-9]+}}:
; ORIGINS: store {{.*}} align 32
; ORIGINS: br label
; ORIGINS: {{^[0-9]+}}:
; CHECK: store {{.*}} align 32
; CHECK: ret void


; load followed by cmp: check that we load the shadow and call __msan_warning_with_origin.
define void @LoadAndCmp(ptr nocapture %a) nounwind uwtable sanitize_memory {
entry:
  %0 = load i32, ptr %a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...) @foo() nounwind
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

declare void @foo(...)

; CHECK-LABEL: @LoadAndCmp
; CHECK: %0 = load i32,
; CHECK: = load
; ORIGINS: %[[ORIGIN:.*]] = load
; NOORIGINS: call void @__msan_warning_noreturn()
; ORIGINS: call void @__msan_warning_with_origin_noreturn(i32 %[[ORIGIN]])
; CHECK-CONT:
; CHECK-NEXT: unreachable
; CHECK: br i1 %tobool
; CHECK: ret void

; Check that we store the shadow for the retval.
define i32 @ReturnInt() nounwind uwtable readnone sanitize_memory {
entry:
  ret i32 123
}

; CHECK-LABEL: @ReturnInt
; CHECK: store i32 0,{{.*}}__msan_retval_tls
; CHECK: ret i32

; Check that we get the shadow for the retval.
define void @CopyRetVal(ptr nocapture %a) nounwind uwtable sanitize_memory {
entry:
  %call = tail call i32 @ReturnInt() nounwind
  store i32 %call, ptr %a, align 4
  ret void
}

; CHECK-LABEL: @CopyRetVal
; CHECK: load{{.*}}__msan_retval_tls
; CHECK: store
; CHECK: store
; CHECK: ret void


; Check that we generate PHIs for shadow.
define void @FuncWithPhi(ptr nocapture %a, ptr %b, ptr nocapture %c) nounwind uwtable sanitize_memory {
entry:
  %tobool = icmp eq ptr %b, null
  br i1 %tobool, label %if.else, label %if.then

  if.then:                                          ; preds = %entry
  %0 = load i32, ptr %b, align 4
  br label %if.end

  if.else:                                          ; preds = %entry
  %1 = load i32, ptr %c, align 4
  br label %if.end

  if.end:                                           ; preds = %if.else, %if.then
  %t.0 = phi i32 [ %0, %if.then ], [ %1, %if.else ]
  store i32 %t.0, ptr %a, align 4
  ret void
}

; CHECK-LABEL: @FuncWithPhi
; NOORIGINS: call void @__msan_warning_noreturn()
; ORIGINS: call void @__msan_warning_with_origin_noreturn(i32
; CHECK: = phi
; CHECK-NEXT: = phi
; CHECK: store
; CHECK: store
; CHECK: ret void

; Compute shadow for "x << 10"
define void @ShlConst(ptr nocapture %x) nounwind uwtable sanitize_memory {
entry:
  %0 = load i32, ptr %x, align 4
  %1 = shl i32 %0, 10
  store i32 %1, ptr %x, align 4
  ret void
}

; CHECK-LABEL: @ShlConst
; CHECK: = load
; CHECK: = load
; CHECK: shl
; CHECK: shl
; CHECK: store
; CHECK: store
; CHECK: ret void

; Compute shadow for "10 << x": it should have 'sext i1'.
define void @ShlNonConst(ptr nocapture %x) nounwind uwtable sanitize_memory {
entry:
  %0 = load i32, ptr %x, align 4
  %1 = shl i32 10, %0
  store i32 %1, ptr %x, align 4
  ret void
}

; CHECK-LABEL: @ShlNonConst
; CHECK: = load
; CHECK: = load
; CHECK: = sext i1
; CHECK: store
; CHECK: store
; CHECK: ret void

; SExt
define void @SExt(ptr nocapture %a, ptr nocapture %b) nounwind uwtable sanitize_memory {
entry:
  %0 = load i16, ptr %b, align 2
  %1 = sext i16 %0 to i32
  store i32 %1, ptr %a, align 4
  ret void
}

; CHECK-LABEL: @SExt
; CHECK: = load
; CHECK: = load
; CHECK: = sext
; CHECK: = sext
; CHECK: store
; CHECK: store
; CHECK: ret void


; memset
define void @MemSet(ptr nocapture %x) nounwind uwtable sanitize_memory {
entry:
  call void @llvm.memset.p0.i64(ptr %x, i8 42, i64 10, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind

; CHECK-LABEL: @MemSet
; CHECK: call ptr @__msan_memset
; CHECK: ret void


; memcpy
define void @MemCpy(ptr nocapture %x, ptr nocapture %y) nounwind uwtable sanitize_memory {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %x, ptr %y, i64 10, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

; CHECK-LABEL: @MemCpy
; CHECK: call ptr @__msan_memcpy
; CHECK: ret void

; memset.inline
define void @MemSetInline(ptr nocapture %x) nounwind uwtable sanitize_memory {
entry:
  call void @llvm.memset.inline.p0.i64(ptr %x, i8 42, i64 10, i1 false)
  ret void
}

declare void @llvm.memset.inline.p0.i64(ptr nocapture, i8, i64, i1) nounwind

; CHECK-LABEL: @MemSetInline
; CHECK: call ptr @__msan_memset
; CHECK: ret void

; memcpy.inline
define void @MemCpyInline(ptr nocapture %x, ptr nocapture %y) nounwind uwtable sanitize_memory {
entry:
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %x, ptr %y, i64 10, i1 false)
  ret void
}

declare void @llvm.memcpy.inline.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

; CHECK-LABEL: @MemCpyInline
; CHECK: call ptr @__msan_memcpy
; CHECK: ret void

; memmove is lowered to a call
define void @MemMove(ptr nocapture %x, ptr nocapture %y) nounwind uwtable sanitize_memory {
entry:
  call void @llvm.memmove.p0.p0.i64(ptr %x, ptr %y, i64 10, i1 false)
  ret void
}

declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

; CHECK-LABEL: @MemMove
; CHECK: call ptr @__msan_memmove
; CHECK: ret void

;; ------------
;; Placeholder tests that will fail once element atomic @llvm.mem[cpy|move|set] intrinsics have
;; been added to the MemIntrinsic class hierarchy. These will act as a reminder to
;; verify that MSAN handles these intrinsics properly once they have been
;; added to that class hierarchy.
declare void @llvm.memset.element.unordered.atomic.p0.i64(ptr nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) nounwind

define void @atomic_memcpy(ptr nocapture %x, ptr nocapture %y) nounwind {
  ; CHECK-LABEL: atomic_memcpy
  ; CHECK-NEXT: call void @llvm.donothing
  ; CHECK-NEXT: call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %x, ptr align 2 %y, i64 16, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %x, ptr align 2 %y, i64 16, i32 1)
  ret void
}

define void @atomic_memmove(ptr nocapture %x, ptr nocapture %y) nounwind {
  ; CHECK-LABEL: atomic_memmove
  ; CHECK-NEXT: call void @llvm.donothing
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %x, ptr align 2 %y, i64 16, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %x, ptr align 2 %y, i64 16, i32 1)
  ret void
}

define void @atomic_memset(ptr nocapture %x) nounwind {
  ; CHECK-LABEL: atomic_memset
  ; CHECK-NEXT: call void @llvm.donothing
  ; CHECK-NEXT: call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %x, i8 88, i64 16, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %x, i8 88, i64 16, i32 1)
  ret void
}

;; ------------


; Check that we propagate shadow for "select"

define i32 @Select(i32 %a, i32 %b, i1 %c) nounwind uwtable readnone sanitize_memory {
entry:
  %cond = select i1 %c, i32 %a, i32 %b
  ret i32 %cond
}

; CHECK-LABEL: @Select
; CHECK: select i1
; CHECK-DAG: or i32
; CHECK-DAG: xor i32
; CHECK: or i32
; CHECK-DAG: select i1
; ORIGINS-DAG: select
; ORIGINS-DAG: select
; CHECK-DAG: select i1
; CHECK: store i32{{.*}}@__msan_retval_tls
; ORIGINS: store i32{{.*}}@__msan_retval_origin_tls
; CHECK: ret i32


; Check that we propagate origin for "select" with vector condition.
; Select condition is flattened to i1, which is then used to select one of the
; argument origins.

define <8 x i16> @SelectVector(<8 x i16> %a, <8 x i16> %b, <8 x i1> %c) nounwind uwtable readnone sanitize_memory {
entry:
  %cond = select <8 x i1> %c, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %cond
}

; CHECK-LABEL: @SelectVector
; CHECK: select <8 x i1>
; CHECK-DAG: or <8 x i16>
; CHECK-DAG: xor <8 x i16>
; CHECK: or <8 x i16>
; CHECK-DAG: select <8 x i1>
; ORIGINS-DAG: select
; ORIGINS-DAG: select
; CHECK-DAG: select <8 x i1>
; CHECK: store <8 x i16>{{.*}}@__msan_retval_tls
; ORIGINS: store i32{{.*}}@__msan_retval_origin_tls
; CHECK: ret <8 x i16>


; Check that we propagate origin for "select" with scalar condition and vector
; arguments. Select condition shadow is sign-extended to the vector type and
; mixed into the result shadow.

define <8 x i16> @SelectVector2(<8 x i16> %a, <8 x i16> %b, i1 %c) nounwind uwtable readnone sanitize_memory {
entry:
  %cond = select i1 %c, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %cond
}

; CHECK-LABEL: @SelectVector2
; CHECK: select i1
; CHECK-DAG: or <8 x i16>
; CHECK-DAG: xor <8 x i16>
; CHECK: or <8 x i16>
; CHECK-DAG: select i1
; ORIGINS-DAG: select i1
; ORIGINS-DAG: select i1
; CHECK-DAG: select i1
; CHECK: ret <8 x i16>


define { i64, i64 } @SelectStruct(i1 zeroext %x, { i64, i64 } %a, { i64, i64 } %b) readnone sanitize_memory {
entry:
  %c = select i1 %x, { i64, i64 } %a, { i64, i64 } %b
  ret { i64, i64 } %c
}

; CHECK-LABEL: @SelectStruct
; CHECK: select i1 {{.*}}, { i64, i64 }
; CHECK-NEXT: select i1 {{.*}}, { i64, i64 } { i64 -1, i64 -1 }, { i64, i64 }
; ORIGINS: select i1
; ORIGINS: select i1
; CHECK-NEXT: select i1 {{.*}}, { i64, i64 }
; CHECK: ret { i64, i64 }


define { ptr, double } @SelectStruct2(i1 zeroext %x, { ptr, double } %a, { ptr, double } %b) readnone sanitize_memory {
entry:
  %c = select i1 %x, { ptr, double } %a, { ptr, double } %b
  ret { ptr, double } %c
}

; CHECK-LABEL: @SelectStruct2
; CHECK: select i1 {{.*}}, { i64, i64 }
; CHECK-NEXT: select i1 {{.*}}, { i64, i64 } { i64 -1, i64 -1 }, { i64, i64 }
; ORIGINS: select i1
; ORIGINS: select i1
; CHECK-NEXT: select i1 {{.*}}, { ptr, double }
; CHECK: ret { ptr, double }


define ptr @IntToPtr(i64 %x) nounwind uwtable readnone sanitize_memory {
entry:
  %0 = inttoptr i64 %x to ptr
  ret ptr %0
}

; CHECK-LABEL: @IntToPtr
; CHECK: load i64, ptr{{.*}}__msan_param_tls
; ORIGINS-NEXT: load i32, ptr{{.*}}__msan_param_origin_tls
; CHECK-NEXT: call void @llvm.donothing
; CHECK-NEXT: inttoptr
; CHECK-NEXT: store i64{{.*}}__msan_retval_tls
; CHECK: ret ptr


define ptr @IntToPtr_ZExt(i16 %x) nounwind uwtable readnone sanitize_memory {
entry:
  %0 = inttoptr i16 %x to ptr
  ret ptr %0
}

; CHECK-LABEL: @IntToPtr_ZExt
; CHECK: load i16, ptr{{.*}}__msan_param_tls
; CHECK: zext
; CHECK-NEXT: inttoptr
; CHECK-NEXT: store i64{{.*}}__msan_retval_tls
; CHECK: ret ptr


; Check that we insert exactly one check on udiv
; (2nd arg shadow is checked, 1st arg shadow is propagated)

define i32 @Div(i32 %a, i32 %b) nounwind uwtable readnone sanitize_memory {
entry:
  %div = udiv i32 %a, %b
  ret i32 %div
}

; CHECK-LABEL: @Div
; CHECK: icmp
; NOORIGINS: call void @__msan_warning_noreturn()
; ORIGINS: call void @__msan_warning_with_origin_noreturn(i32
; CHECK-NOT: icmp
; CHECK: udiv
; CHECK-NOT: icmp
; CHECK: ret i32

; Check that fdiv, unlike udiv, simply propagates shadow.

define float @FDiv(float %a, float %b) nounwind uwtable readnone sanitize_memory {
entry:
  %c = fdiv float %a, %b
  ret float %c
}

; CHECK-LABEL: @FDiv
; CHECK: %[[SA:.*]] = load i32,{{.*}}@__msan_param_tls
; CHECK: %[[SB:.*]] = load i32,{{.*}}@__msan_param_tls
; CHECK: %[[SC:.*]] = or i32 %[[SA]], %[[SB]]
; CHECK: = fdiv float
; CHECK: store i32 %[[SC]], ptr {{.*}}@__msan_retval_tls
; CHECK: ret float

; Check that fneg simply propagates shadow.

define float @FNeg(float %a) nounwind uwtable readnone sanitize_memory {
entry:
  %c = fneg float %a
  ret float %c
}

; CHECK-LABEL: @FNeg
; CHECK: %[[SA:.*]] = load i32,{{.*}}@__msan_param_tls
; ORIGINS: %[[SB:.*]] = load i32,{{.*}}@__msan_param_origin_tls
; CHECK: = fneg float
; CHECK: store i32 %[[SA]], ptr {{.*}}@__msan_retval_tls
; ORIGINS: store i32{{.*}}@__msan_retval_origin_tls
; CHECK: ret float

; Check that we propagate shadow for x<0, x>=0, etc (i.e. sign bit tests)

define zeroext i1 @ICmpSLTZero(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp slt i32 %x, 0
  ret i1 %1
}

; CHECK-LABEL: @ICmpSLTZero
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSGEZero(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp sge i32 %x, 0
  ret i1 %1
}

; CHECK-LABEL: @ICmpSGEZero
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sge
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSGTZero(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp sgt i32 0, %x
  ret i1 %1
}

; CHECK-LABEL: @ICmpSGTZero
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sgt
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSLEZero(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp sle i32 0, %x
  ret i1 %1
}

; CHECK-LABEL: @ICmpSLEZero
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sle
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1


; Check that we propagate shadow for x<=-1, x>-1, etc (i.e. sign bit tests)

define zeroext i1 @ICmpSLTAllOnes(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp slt i32 -1, %x
  ret i1 %1
}

; CHECK-LABEL: @ICmpSLTAllOnes
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSGEAllOnes(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp sge i32 -1, %x
  ret i1 %1
}

; CHECK-LABEL: @ICmpSGEAllOnes
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sge
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSGTAllOnes(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp sgt i32 %x, -1
  ret i1 %1
}

; CHECK-LABEL: @ICmpSGTAllOnes
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sgt
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSLEAllOnes(i32 %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp sle i32 %x, -1
  ret i1 %1
}

; CHECK-LABEL: @ICmpSLEAllOnes
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sle
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1


; Check that we propagate shadow for x<0, x>=0, etc (i.e. sign bit tests)
; of the vector arguments.

define <2 x i1> @ICmpSLT_vector_Zero(<2 x ptr> %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp slt <2 x ptr> %x, zeroinitializer
  ret <2 x i1> %1
}

; CHECK-LABEL: @ICmpSLT_vector_Zero
; CHECK: icmp slt <2 x i64>
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp slt <2 x ptr>
; CHECK-NOT: call void @__msan_warning
; CHECK: ret <2 x i1>

; Check that we propagate shadow for x<=-1, x>0, etc (i.e. sign bit tests)
; of the vector arguments.

define <2 x i1> @ICmpSLT_vector_AllOnes(<2 x i32> %x) nounwind uwtable readnone sanitize_memory {
  %1 = icmp slt <2 x i32> <i32 -1, i32 -1>, %x
  ret <2 x i1> %1
}

; CHECK-LABEL: @ICmpSLT_vector_AllOnes
; CHECK: icmp slt <2 x i32>
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp slt <2 x i32>
; CHECK-NOT: call void @__msan_warning
; CHECK: ret <2 x i1>


; Check that we propagate shadow for unsigned relational comparisons with
; constants

define zeroext i1 @ICmpUGTConst(i32 %x) nounwind uwtable readnone sanitize_memory {
entry:
  %cmp = icmp ugt i32 %x, 7
  ret i1 %cmp
}

; CHECK-LABEL: @ICmpUGTConst
; CHECK: icmp ugt i32
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp ugt i32
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp ugt i32
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1


; Check that loads of shadow have the same alignment as the original loads.
; Check that loads of origin have the alignment of max(4, original alignment).

define i32 @ShadowLoadAlignmentLarge() nounwind uwtable sanitize_memory {
  %y = alloca i32, align 64
  %1 = load volatile i32, ptr %y, align 64
  ret i32 %1
}

; CHECK-LABEL: @ShadowLoadAlignmentLarge
; CHECK: load volatile i32, ptr {{.*}} align 64
; CHECK: load i32, ptr {{.*}} align 64
; CHECK: ret i32

define i32 @ShadowLoadAlignmentSmall() nounwind uwtable sanitize_memory {
  %y = alloca i32, align 2
  %1 = load volatile i32, ptr %y, align 2
  ret i32 %1
}

; CHECK-LABEL: @ShadowLoadAlignmentSmall
; CHECK: load volatile i32, ptr {{.*}} align 2
; CHECK: load i32, ptr {{.*}} align 2
; ORIGINS: load i32, ptr {{.*}} align 4
; CHECK: ret i32


; Test vector manipulation instructions.
; Check that the same bit manipulation is applied to the shadow values.
; Check that there is a zero test of the shadow of %idx argument, where present.

define i32 @ExtractElement(<4 x i32> %vec, i32 %idx) sanitize_memory {
  %x = extractelement <4 x i32> %vec, i32 %idx
  ret i32 %x
}

; CHECK-LABEL: @ExtractElement
; CHECK: extractelement
; NOORIGINS: call void @__msan_warning_noreturn()
; ORIGINS: call void @__msan_warning_with_origin_noreturn(i32
; CHECK: extractelement
; CHECK: ret i32

define <4 x i32> @InsertElement(<4 x i32> %vec, i32 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i32 %idx
  ret <4 x i32> %vec1
}

; CHECK-LABEL: @InsertElement
; CHECK: insertelement
; NOORIGINS: call void @__msan_warning_noreturn()
; ORIGINS: call void @__msan_warning_with_origin_noreturn(i32
; CHECK: insertelement
; CHECK: ret <4 x i32>

define <4 x i32> @ShuffleVector(<4 x i32> %vec, <4 x i32> %vec1) sanitize_memory {
  %vec2 = shufflevector <4 x i32> %vec, <4 x i32> %vec1,
                        <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %vec2
}

; CHECK-LABEL: @ShuffleVector
; CHECK: shufflevector
; CHECK-NOT: call void @__msan_warning
; CHECK: shufflevector
; CHECK: ret <4 x i32>


; Test bswap intrinsic instrumentation
define i32 @BSwap(i32 %x) nounwind uwtable readnone sanitize_memory {
  %y = tail call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %y
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone

; CHECK-LABEL: @BSwap
; CHECK-NOT: call void @__msan_warning
; CHECK: @llvm.bswap.i32
; CHECK-NOT: call void @__msan_warning
; CHECK: @llvm.bswap.i32
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i32

; Test handling of vectors of pointers.
; Check that shadow of such vector is a vector of integers.

define <8 x ptr> @VectorOfPointers(ptr %p) nounwind uwtable sanitize_memory {
  %x = load <8 x ptr>, ptr %p
  ret <8 x ptr> %x
}

; CHECK-LABEL: @VectorOfPointers
; CHECK: load <8 x ptr>, ptr
; CHECK: load <8 x i64>, ptr
; CHECK: store <8 x i64> {{.*}} @__msan_retval_tls
; CHECK: ret <8 x ptr>

; Test handling of va_copy.

declare void @llvm.va_copy(ptr, ptr) nounwind

define void @VACopy(ptr %p1, ptr %p2) nounwind uwtable sanitize_memory {
  call void @llvm.va_copy(ptr %p1, ptr %p2) nounwind
  ret void
}

; CHECK-LABEL: @VACopy
; CHECK: call void @llvm.memset.p0.i64({{.*}}, i8 0, i64 24, i1 false)
; CHECK: ret void


; Test that va_start instrumentation does not use va_arg_tls*.
; It should work with a local stack copy instead.

%struct.__va_list_tag = type { i32, i32, ptr, ptr }
declare void @llvm.va_start(ptr) nounwind

; Function Attrs: nounwind uwtable
define void @VAStart(i32 %x, ...) sanitize_memory {
entry:
  %x.addr = alloca i32, align 4
  %va = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.va_start(ptr %va)
  ret void
}

; CHECK-LABEL: @VAStart
; CHECK: call void @llvm.va_start
; CHECK-NOT: @__msan_va_arg_tls
; CHECK-NOT: @__msan_va_arg_overflow_size_tls
; CHECK: ret void


; Test handling of volatile stores.
; Check that MemorySanitizer does not add a check of the value being stored.

define void @VolatileStore(ptr nocapture %p, i32 %x) nounwind uwtable sanitize_memory {
entry:
  store volatile i32 %x, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @VolatileStore
; CHECK-NOT: @__msan_warning_with_origin
; CHECK: ret void


; Test that checks are omitted and returned value is always initialized if
; sanitize_memory attribute is missing.

define i32 @NoSanitizeMemory(i32 %x) uwtable {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar()
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret i32 %x
}

declare void @bar()

; CHECK-LABEL: @NoSanitizeMemory
; CHECK-NOT: @__msan_warning_with_origin
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK-NOT: @__msan_warning_with_origin
; CHECK: ret i32


; Test that stack allocations are unpoisoned in functions missing
; sanitize_memory attribute

define i32 @NoSanitizeMemoryAlloca() {
entry:
  %p = alloca i32, align 4
  %x = call i32 @NoSanitizeMemoryAllocaHelper(ptr %p)
  ret i32 %x
}

declare i32 @NoSanitizeMemoryAllocaHelper(ptr %p)

; CHECK-LABEL: @NoSanitizeMemoryAlloca
; CHECK: call void @llvm.memset.p0.i64(ptr align 4 {{.*}}, i8 0, i64 4, i1 false)
; CHECK: call i32 @NoSanitizeMemoryAllocaHelper(ptr
; CHECK: ret i32


; Test that undef is unpoisoned in functions missing
; sanitize_memory attribute

define i32 @NoSanitizeMemoryUndef() {
entry:
  %x = call i32 @NoSanitizeMemoryUndefHelper(i32 undef)
  ret i32 %x
}

declare i32 @NoSanitizeMemoryUndefHelper(i32 %x)

; CHECK-LABEL: @NoSanitizeMemoryUndef
; CHECK: store i32 0, ptr @__msan_param_tls
; CHECK: call i32 @NoSanitizeMemoryUndefHelper(i32 undef)
; CHECK: ret i32


; Test PHINode instrumentation in ignorelisted functions

define i32 @NoSanitizeMemoryPHI(i32 %x) {
entry:
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ undef, %cond.true ], [ undef, %cond.false ]
  ret i32 %cond
}

; CHECK: [[A:%.*]] = phi i32 [ undef, %cond.true ], [ undef, %cond.false ]
; CHECK: store i32 0, ptr @__msan_retval_tls
; CHECK: ret i32 [[A]]


; Test that there are no __msan_param_origin_tls stores when
; argument shadow is a compile-time zero constant (which is always the case
; in functions missing sanitize_memory attribute).

define i32 @NoSanitizeMemoryParamTLS(ptr nocapture readonly %x) {
entry:
  %0 = load i32, ptr %x, align 4
  %call = tail call i32 @NoSanitizeMemoryParamTLSHelper(i32 %0)
  ret i32 %call
}

declare i32 @NoSanitizeMemoryParamTLSHelper(i32 %x)

; CHECK-LABEL: define i32 @NoSanitizeMemoryParamTLS(
; CHECK-NOT: __msan_param_origin_tls
; CHECK: ret i32


; Test argument shadow alignment

define <2 x i64> @ArgumentShadowAlignment(i64 %a, <2 x i64> %b) sanitize_memory {
entry:
  ret <2 x i64> %b
}

; CHECK-LABEL: @ArgumentShadowAlignment
; CHECK: load <2 x i64>, ptr {{.*}} @__msan_param_tls {{.*}}, align 8
; CHECK: store <2 x i64> {{.*}}, ptr @__msan_retval_tls, align 8
; CHECK: ret <2 x i64>


; Test origin propagation for insertvalue

define { i64, i32 } @make_pair_64_32(i64 %x, i32 %y) sanitize_memory {
entry:
  %a = insertvalue { i64, i32 } undef, i64 %x, 0
  %b = insertvalue { i64, i32 } %a, i32 %y, 1
  ret { i64, i32 } %b
}

; ORIGINS: @make_pair_64_32
; First element shadow
; ORIGINS: insertvalue { i64, i32 } { i64 -1, i32 -1 }, i64 {{.*}}, 0
; First element origin
; ORIGINS: icmp ne i64
; ORIGINS: select i1
; First element app value
; ORIGINS: insertvalue { i64, i32 } undef, i64 {{.*}}, 0
; Second element shadow
; ORIGINS: insertvalue { i64, i32 } {{.*}}, i32 {{.*}}, 1
; Second element origin
; ORIGINS: icmp ne i32
; ORIGINS: select i1
; Second element app value
; ORIGINS: insertvalue { i64, i32 } {{.*}}, i32 {{.*}}, 1
; ORIGINS: ret { i64, i32 }


; Test shadow propagation for aggregates passed through ellipsis.

%struct.StructByVal = type { i32, i32, i32, i32 }

declare void @VAArgStructFn(i32 %guard, ...)

define void @VAArgStruct(ptr nocapture %s) sanitize_memory {
entry:
  %agg.tmp2 = alloca %struct.StructByVal, align 8
  %agg.tmp.sroa.0.0.copyload = load i64, ptr %s, align 4
  %agg.tmp.sroa.2.0..sroa_idx = getelementptr inbounds %struct.StructByVal, ptr %s, i64 0, i32 2
  %agg.tmp.sroa.2.0.copyload = load i64, ptr %agg.tmp.sroa.2.0..sroa_idx, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %s, i64 16, i1 false)
  call void (i32, ...) @VAArgStructFn(i32 undef, i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp.sroa.2.0.copyload, i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp.sroa.2.0.copyload, ptr byval(%struct.StructByVal) align 8 %agg.tmp2)
  ret void
}

; "undef" and the first 2 structs go to general purpose registers;
; the third struct goes to the overflow area byval

; CHECK-LABEL: @VAArgStruct
; undef not stored to __msan_va_arg_tls - it's a fixed argument
; first struct through general purpose registers
; CHECK: store i64 {{.*}}, ptr {{.*}}@__msan_va_arg_tls{{.*}}, i64 8){{.*}}, align 8
; CHECK: store i64 {{.*}}, ptr {{.*}}@__msan_va_arg_tls{{.*}}, i64 16){{.*}}, align 8
; second struct through general purpose registers
; CHECK: store i64 {{.*}}, ptr {{.*}}@__msan_va_arg_tls{{.*}}, i64 24){{.*}}, align 8
; CHECK: store i64 {{.*}}, ptr {{.*}}@__msan_va_arg_tls{{.*}}, i64 32){{.*}}, align 8
; third struct through the overflow area byval
; CHECK: ptrtoint ptr {{.*}} to i64
; CHECK: call void @llvm.memcpy.p0.p0.i64{{.*}}@__msan_va_arg_tls {{.*}}, i64 176
; CHECK: store i64 16, ptr @__msan_va_arg_overflow_size_tls
; CHECK: call void (i32, ...) @VAArgStructFn
; CHECK: ret void

; Same code compiled without SSE (see attributes below).
; The register save area is only 48 bytes instead of 176.
define void @VAArgStructNoSSE(ptr nocapture %s) sanitize_memory #0 {
entry:
  %agg.tmp2 = alloca %struct.StructByVal, align 8
  %agg.tmp.sroa.0.0.copyload = load i64, ptr %s, align 4
  %agg.tmp.sroa.2.0..sroa_idx = getelementptr inbounds %struct.StructByVal, ptr %s, i64 0, i32 2
  %agg.tmp.sroa.2.0.copyload = load i64, ptr %agg.tmp.sroa.2.0..sroa_idx, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %s, i64 16, i1 false)
  call void (i32, ...) @VAArgStructFn(i32 undef, i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp.sroa.2.0.copyload, i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp.sroa.2.0.copyload, ptr byval(%struct.StructByVal) align 8 %agg.tmp2)
  ret void
}

attributes #0 = { "target-features"="+fxsr,+x87,-sse" }

; CHECK: call void @llvm.memcpy.p0.p0.i64{{.*}}@__msan_va_arg_tls {{.*}}, i64 48

declare i32 @InnerTailCall(i32 %a)

define void @MismatchedReturnTypeTailCall(i32 %a) sanitize_memory {
  %b = tail call i32 @InnerTailCall(i32 %a)
  ret void
}

; We used to strip off the 'tail' modifier, but now that we unpoison return slot
; shadow before the call, we don't need to anymore.

; CHECK-LABEL: define void @MismatchedReturnTypeTailCall
; CHECK: tail call i32 @InnerTailCall
; CHECK: ret void


declare i32 @MustTailCall(i32 %a)

define i32 @CallMustTailCall(i32 %a) sanitize_memory {
  %b = musttail call i32 @MustTailCall(i32 %a)
  ret i32 %b
}

; For "musttail" calls we can not insert any shadow manipulating code between
; call and the return instruction. And we don't need to, because everything is
; taken care of in the callee.

; CHECK-LABEL: define i32 @CallMustTailCall
; CHECK: musttail call i32 @MustTailCall
; No instrumentation between call and ret.
; CHECK-NEXT: ret i32

declare ptr @MismatchingMustTailCall(i32 %a)

define ptr @MismatchingCallMustTailCall(i32 %a) sanitize_memory {
  %b = musttail call ptr @MismatchingMustTailCall(i32 %a)
  ret ptr %b
}

; For "musttail" calls we can not insert any shadow manipulating code between
; call and the return instruction. And we don't need to, because everything is
; taken care of in the callee.

; CHECK-LABEL: define ptr @MismatchingCallMustTailCall
; CHECK: musttail call ptr @MismatchingMustTailCall
; No instrumentation between call and ret.
; CHECK-NEXT: ret ptr


; CHECK-LABEL: define internal void @msan.module_ctor() #[[#ATTR:]] {
; CHECK: call void @__msan_init()

; CHECK-CALLS: declare void @__msan_maybe_warning_1(i8 zeroext, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_store_origin_1(i8 zeroext, ptr, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_warning_2(i16 zeroext, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_store_origin_2(i16 zeroext, ptr, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_warning_4(i32 zeroext, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_store_origin_4(i32 zeroext, ptr, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_warning_8(i64 zeroext, i32 zeroext)
; CHECK-CALLS: declare void @__msan_maybe_store_origin_8(i64 zeroext, ptr, i32 zeroext)

; CHECK:       attributes #[[#ATTR]] = { nounwind }
