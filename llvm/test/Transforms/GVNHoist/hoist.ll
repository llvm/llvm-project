; RUN: opt -gvn-hoist -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@GlobalVar = internal global float 1.000000e+00

; Check that all scalar expressions are hoisted.
;
; CHECK-LABEL: @scalarsHoisting
; CHECK: fsub
; CHECK: fsub
; CHECK: fmul
; CHECK: fmul
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @scalarsHoisting(float %d, float %min, float %max, float %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %sub = fsub float %min, %a
  %mul = fmul float %sub, %div
  %sub1 = fsub float %max, %a
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.else:                                          ; preds = %entry
  %sub3 = fsub float %max, %a
  %mul4 = fmul float %sub3, %div
  %sub5 = fsub float %min, %a
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %if.then ], [ %mul4, %if.else ]
  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that all loads and scalars depending on the loads are hoisted.
; Check that getelementptr computation gets hoisted before the load.
;
; CHECK-LABEL: @readsAndScalarsHoisting
; CHECK: load
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fsub
; CHECK: fmul
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @readsAndScalarsHoisting(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %A = getelementptr float, float* %min, i32 1
  %0 = load float, float* %A, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %B = getelementptr float, float* %min, i32 1
  %5 = load float, float* %B, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %if.then ], [ %mul4, %if.else ]
  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that we do not hoist loads after a store: the first two loads will be
; hoisted, and then the third load will not be hoisted.
;
; CHECK-LABEL: @readsAndWrites
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: store
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @readsAndWrites(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  store float %0, float* @GlobalVar
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %if.then ], [ %mul4, %if.else ]
  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that we do hoist loads when the store is above the insertion point.
;
; CHECK-LABEL: @readsAndWriteAboveInsertPt
; CHECK: load
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fsub
; CHECK: fmul
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @readsAndWriteAboveInsertPt(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  store float 0.000000e+00, float* @GlobalVar
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %if.then ], [ %mul4, %if.else ]
  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that dependent expressions are hoisted.
; CHECK-LABEL: @dependentScalarsHoisting
; CHECK: fsub
; CHECK: fadd
; CHECK: fdiv
; CHECK: fmul
; CHECK-NOT: fsub
; CHECK-NOT: fadd
; CHECK-NOT: fdiv
; CHECK-NOT: fmul
define float @dependentScalarsHoisting(float %a, float %b, i1 %c) {
entry:
  br i1 %c, label %if.then, label %if.else

if.then:
  %d = fsub float %b, %a
  %e = fadd float %d, %a
  %f = fdiv float %e, %a
  %g = fmul float %f, %a
  br label %if.end

if.else:
  %h = fsub float %b, %a
  %i = fadd float %h, %a
  %j = fdiv float %i, %a
  %k = fmul float %j, %a
  br label %if.end

if.end:
  %r = phi float [ %g, %if.then ], [ %k, %if.else ]
  ret float %r
}

; Check that all independent expressions are hoisted.
; CHECK-LABEL: @independentScalarsHoisting
; CHECK: fsub
; CHECK: fdiv
; CHECK: fmul
; CHECK: fadd
; CHECK-NOT: fsub
; CHECK-NOT: fdiv
; CHECK-NOT: fmul
define float @independentScalarsHoisting(float %a, float %b, i1 %c) {
entry:
  br i1 %c, label %if.then, label %if.else

if.then:
  %d = fadd float %b, %a
  %e = fsub float %b, %a
  %f = fdiv float %b, %a
  %g = fmul float %b, %a
  br label %if.end

if.else:
  %i = fadd float %b, %a
  %h = fsub float %b, %a
  %j = fdiv float %b, %a
  %k = fmul float %b, %a
  br label %if.end

if.end:
  %p = phi float [ %d, %if.then ], [ %i, %if.else ]
  %q = phi float [ %e, %if.then ], [ %h, %if.else ]
  %r = phi float [ %f, %if.then ], [ %j, %if.else ]
  %s = phi float [ %g, %if.then ], [ %k, %if.else ]
  %t = fadd float %p, %q
  %u = fadd float %r, %s
  %v = fadd float %t, %u
  ret float %v
}

; Check that we hoist load and scalar expressions in triangles.
; CHECK-LABEL: @triangleHoisting
; CHECK: load
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fsub
; CHECK: fmul
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @triangleHoisting(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.end:                                          ; preds = %entry
  %p1 = phi float [ %mul2, %if.then ], [ 0.000000e+00, %entry ]
  %p2 = phi float [ %mul, %if.then ], [ 0.000000e+00, %entry ]
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div

  %x = fadd float %p1, %mul6
  %y = fadd float %p2, %mul4
  %z = fadd float %x, %y
  ret float %z
}

; Check that we hoist load and scalar expressions in dominator.
; CHECK-LABEL: @dominatorHoisting
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @dominatorHoisting(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                          ; preds = %entry
  %p1 = phi float [ %mul4, %if.then ], [ 0.000000e+00, %entry ]
  %p2 = phi float [ %mul6, %if.then ], [ 0.000000e+00, %entry ]

  %x = fadd float %p1, %mul2
  %y = fadd float %p2, %mul
  %z = fadd float %x, %y
  ret float %z
}

; Check that we hoist load and scalar expressions in dominator.
; CHECK-LABEL: @domHoisting
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @domHoisting(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.else:
  %6 = load float, float* %max, align 4
  %7 = load float, float* %a, align 4
  %sub9 = fsub float %6, %7
  %mul10 = fmul float %sub9, %div
  %8 = load float, float* %min, align 4
  %sub12 = fsub float %8, %7
  %mul13 = fmul float %sub12, %div
  br label %if.end

if.end:
  %p1 = phi float [ %mul4, %if.then ], [ %mul10, %if.else ]
  %p2 = phi float [ %mul6, %if.then ], [ %mul13, %if.else ]

  %x = fadd float %p1, %mul2
  %y = fadd float %p2, %mul
  %z = fadd float %x, %y
  ret float %z
}

; Check that we do not hoist loads past stores within a same basic block.
; CHECK-LABEL: @noHoistInSingleBBWithStore
; CHECK: load
; CHECK: store
; CHECK: load
; CHECK: store
define i32 @noHoistInSingleBBWithStore() {
entry:
  %D = alloca i32, align 4
  %0 = bitcast i32* %D to i8*
  %bf = load i8, i8* %0, align 4
  %bf.clear = and i8 %bf, -3
  store i8 %bf.clear, i8* %0, align 4
  %bf1 = load i8, i8* %0, align 4
  %bf.clear1 = and i8 %bf1, 1
  store i8 %bf.clear1, i8* %0, align 4
  ret i32 0
}

; Check that we do not hoist loads past calls within a same basic block.
; CHECK-LABEL: @noHoistInSingleBBWithCall
; CHECK: load
; CHECK: call
; CHECK: load
declare void @foo()
define i32 @noHoistInSingleBBWithCall() {
entry:
  %D = alloca i32, align 4
  %0 = bitcast i32* %D to i8*
  %bf = load i8, i8* %0, align 4
  %bf.clear = and i8 %bf, -3
  call void @foo()
  %bf1 = load i8, i8* %0, align 4
  %bf.clear1 = and i8 %bf1, 1
  ret i32 0
}

; Check that we do not hoist loads past stores in any branch of a diamond.
; CHECK-LABEL: @noHoistInDiamondWithOneStore1
; CHECK: fdiv
; CHECK: fcmp
; CHECK: br
define float @noHoistInDiamondWithOneStore1(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store float 0.000000e+00, float* @GlobalVar
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.else:                                          ; preds = %entry
  ; There are no side effects on the if.else branch.
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %if.then ], [ %mul4, %if.else ]

  %6 = load float, float* %max, align 4
  %7 = load float, float* %a, align 4
  %sub6 = fsub float %6, %7
  %mul7 = fmul float %sub6, %div
  %8 = load float, float* %min, align 4
  %sub8 = fsub float %8, %7
  %mul9 = fmul float %sub8, %div

  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that we do not hoist loads past stores from half diamond.
; CHECK-LABEL: @noHoistInHalfDiamondPastStore
; CHECK: load
; CHECK-NEXT: load
; CHECK-NEXT: store
; CHECK-NEXT: br
; CHECK: load
; CHECK: load
; CHECK: load
; CHECK: br
define float @noHoistInHalfDiamondPastStore(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4

  ; Loads should not be hoisted above this store.
  store float 0.000000e+00, float* @GlobalVar

  br i1 %cmp, label %if.then, label %if.end

if.then:
  ; There are no side effects on the if.then branch.
  %2 = load float, float* %max, align 4
  %3 = load float, float* %a, align 4
  %sub3 = fsub float %2, %3
  %mul4 = fmul float %sub3, %div
  %4 = load float, float* %min, align 4
  %sub5 = fsub float %4, %3
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:
  %tmax.0 = phi float [ %mul4, %if.then ], [ %0, %entry ]
  %tmin.0 = phi float [ %mul6, %if.then ], [ %1, %entry ]

  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that we do not hoist loads past a store in any branch of a diamond.
; CHECK-LABEL: @noHoistInDiamondWithOneStore2
; CHECK: fdiv
; CHECK: fcmp
; CHECK: br
define float @noHoistInDiamondWithOneStore2(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ; There are no side effects on the if.then branch.
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %if.end

if.else:                                          ; preds = %entry
  store float 0.000000e+00, float* @GlobalVar
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %if.then ], [ %mul4, %if.else ]

  %6 = load float, float* %max, align 4
  %7 = load float, float* %a, align 4
  %sub6 = fsub float %6, %7
  %mul7 = fmul float %sub6, %div
  %8 = load float, float* %min, align 4
  %sub8 = fsub float %8, %7
  %mul9 = fmul float %sub8, %div

  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that we do not hoist loads outside a loop containing stores.
; CHECK-LABEL: @noHoistInLoopsWithStores
; CHECK: fdiv
; CHECK: fcmp
; CHECK: br
define float @noHoistInLoopsWithStores(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %do.body, label %if.else

do.body:
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4

  ; It is unsafe to hoist the loads outside the loop because of the store.
  store float 0.000000e+00, float* @GlobalVar

  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  br label %while.cond

while.cond:
  %cmp1 = fcmp oge float %mul2, 0.000000e+00
  br i1 %cmp1, label %if.end, label %do.body

if.else:
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:
  %tmax.0 = phi float [ %mul2, %while.cond ], [ %mul6, %if.else ]
  %tmin.0 = phi float [ %mul, %while.cond ], [ %mul4, %if.else ]

  %add = fadd float %tmax.0, %tmin.0
  ret float %add
}

; Check that we hoist stores: all the instructions from the then branch
; should be hoisted.
; CHECK-LABEL: @hoistStores
; CHECK: zext
; CHECK-NEXT: trunc
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: store
; CHECK-NEXT: load
; CHECK-NEXT: load
; CHECK-NEXT: zext
; CHECK-NEXT: add
; CHECK-NEXT: store
; CHECK-NEXT: br
; CHECK: if.then
; CHECK: br

%struct.foo = type { i16* }

define void @hoistStores(%struct.foo* %s, i32* %coord, i1 zeroext %delta) {
entry:
  %frombool = zext i1 %delta to i8
  %tobool = trunc i8 %frombool to i1
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %p = getelementptr inbounds %struct.foo, %struct.foo* %s, i32 0, i32 0
  %0 = load i16*, i16** %p, align 8
  %incdec.ptr = getelementptr inbounds i16, i16* %0, i32 1
  store i16* %incdec.ptr, i16** %p, align 8
  %1 = load i16, i16* %0, align 2
  %conv = zext i16 %1 to i32
  %2 = load i32, i32* %coord, align 4
  %add = add i32 %2, %conv
  store i32 %add, i32* %coord, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %p1 = getelementptr inbounds %struct.foo, %struct.foo* %s, i32 0, i32 0
  %3 = load i16*, i16** %p1, align 8
  %incdec.ptr2 = getelementptr inbounds i16, i16* %3, i32 1
  store i16* %incdec.ptr2, i16** %p1, align 8
  %4 = load i16, i16* %3, align 2
  %conv3 = zext i16 %4 to i32
  %5 = load i32, i32* %coord, align 4
  %add4 = add i32 %5, %conv3
  store i32 %add4, i32* %coord, align 4
  %6 = load i16*, i16** %p1, align 8
  %incdec.ptr6 = getelementptr inbounds i16, i16* %6, i32 1
  store i16* %incdec.ptr6, i16** %p1, align 8
  %7 = load i16, i16* %6, align 2
  %conv7 = zext i16 %7 to i32
  %shl = shl i32 %conv7, 8
  %8 = load i32, i32* %coord, align 4
  %add8 = add i32 %8, %shl
  store i32 %add8, i32* %coord, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define i32 @mergeAlignments(i1 %b, i32* %y) {
entry:
  br i1 %b, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %l1 = load i32, i32* %y, align 4
  br label %return

if.end:                                           ; preds = %entry
  %l2 = load i32, i32* %y, align 1
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %l1, %if.then ], [ %l2, %if.end ]
  ret i32 %retval.0
}
; CHECK-LABEL: define i32 @mergeAlignments(
; CHECK: %[[load:.*]] = load i32, i32* %y, align 1
; CHECK: %[[phi:.*]] = phi i32 [ %[[load]], %{{.*}} ], [ %[[load]], %{{.*}} ]
; CHECK: i32 %[[phi]]


declare i8 @pr30991_f() nounwind readonly
declare void @pr30991_f1(i8)
define i8 @pr30991(i8* %sp, i8* %word, i1 %b1, i1 %b2) {
entry:
  br i1 %b1, label %a, label %b

a:
  %r0 = load i8, i8* %word, align 1
  %incdec.ptr = getelementptr i8, i8* %sp, i32 1
  %rr0 = call i8 @pr30991_f() nounwind readonly
  call void @pr30991_f1(i8 %r0)
  ret i8 %rr0

b:
  br i1 %b2, label %c, label %x

c:
  %r1 = load i8, i8* %word, align 1
  %incdec.ptr115 = getelementptr i8, i8* %sp, i32 1
  %rr1 = call i8 @pr30991_f() nounwind readonly
  call void @pr30991_f1(i8 %r1)
  ret i8 %rr1

x:
  %r2 = load i8, i8* %word, align 1
  ret i8 %r2
}

; CHECK-LABEL: define i8 @pr30991
; CHECK:  %r0 = load i8, i8* %word, align 1
; CHECK-NEXT:  br i1 %b1, label %a, label %b
