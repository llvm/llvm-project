; RUN: opt -basic-aa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

;; memcpy.atomic formation (atomic load & store)
define void @test1(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test1(
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %Dest, i8* align 1 %Base, i64 %Size, i32 1)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load atomic i8, i8* %I.0.014 unordered, align 1
  store atomic i8 %V, i8* %DestI unordered, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation (atomic store, normal load)
define void @test2(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test2(
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %Dest, i8* align 1 %Base, i64 %Size, i32 1)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load i8, i8* %I.0.014, align 1
  store atomic i8 %V, i8* %DestI unordered, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation (atomic store, normal load w/ no align)
define void @test2b(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test2b(
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %Dest, i8* align 1 %Base, i64 %Size, i32 1)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load i8, i8* %I.0.014
  store atomic i8 %V, i8* %DestI unordered, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (atomic store, normal load w/ bad align)
define void @test2c(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test2c(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i32, i32 10000
  %Dest = alloca i32, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i32, i32* %Base, i64 %indvar
  %DestI = getelementptr i32, i32* %Dest, i64 %indvar
  %V = load i32, i32* %I.0.014, align 2
  store atomic i32 %V, i32* %DestI unordered, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (atomic store w/ bad align, normal load)
define void @test2d(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test2d(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i32, i32 10000
  %Dest = alloca i32, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i32, i32* %Base, i64 %indvar
  %DestI = getelementptr i32, i32* %Dest, i64 %indvar
  %V = load i32, i32* %I.0.014, align 4
  store atomic i32 %V, i32* %DestI unordered, align 2
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;; memcpy.atomic formation (normal store, atomic load)
define void @test3(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test3(
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %Dest, i8* align 1 %Base, i64 %Size, i32 1)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load atomic i8, i8* %I.0.014 unordered, align 1
  store i8 %V, i8* %DestI, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (normal store w/ no align, atomic load)
define void @test3b(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test3b(
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %Dest, i8* align 1 %Base, i64 %Size, i32 1)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load atomic i8, i8* %I.0.014 unordered, align 1
  store i8 %V, i8* %DestI
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (normal store, atomic load w/ bad align)
define void @test3c(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test3c(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i32, i32 10000
  %Dest = alloca i32, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i32, i32* %Base, i64 %indvar
  %DestI = getelementptr i32, i32* %Dest, i64 %indvar
  %V = load atomic i32, i32* %I.0.014 unordered, align 2
  store i32 %V, i32* %DestI, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (normal store w/ bad align, atomic load)
define void @test3d(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test3d(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i32, i32 10000
  %Dest = alloca i32, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i32, i32* %Base, i64 %indvar
  %DestI = getelementptr i32, i32* %Dest, i64 %indvar
  %V = load atomic i32, i32* %I.0.014 unordered, align 4
  store i32 %V, i32* %DestI, align 2
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;; memcpy.atomic formation rejection (atomic load, ordered-atomic store)
define void @test4(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test4(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load atomic i8, i8* %I.0.014 unordered, align 1
  store atomic i8 %V, i8* %DestI monotonic, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (ordered-atomic load, unordered-atomic store)
define void @test5(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test5(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load atomic i8, i8* %I.0.014 monotonic, align 1
  store atomic i8 %V, i8* %DestI unordered, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation (atomic load & store) -- element size 2
define void @test6(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test6(
; CHECK: [[Sz:%[0-9]+]] = shl nuw i64 %Size, 1
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 2 %Dest{{[0-9]*}}, i8* align 2 %Base{{[0-9]*}}, i64 [[Sz]], i32 2)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i16, i32 10000
  %Dest = alloca i16, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i16, i16* %Base, i64 %indvar
  %DestI = getelementptr i16, i16* %Dest, i64 %indvar
  %V = load atomic i16, i16* %I.0.014 unordered, align 2
  store atomic i16 %V, i16* %DestI unordered, align 2
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation (atomic load & store) -- element size 4
define void @test7(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test7(
; CHECK: [[Sz:%[0-9]+]] = shl nuw i64 %Size, 2
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 4 %Dest{{[0-9]*}}, i8* align 4 %Base{{[0-9]*}}, i64 [[Sz]], i32 4)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i32, i32 10000
  %Dest = alloca i32, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i32, i32* %Base, i64 %indvar
  %DestI = getelementptr i32, i32* %Dest, i64 %indvar
  %V = load atomic i32, i32* %I.0.014 unordered, align 4
  store atomic i32 %V, i32* %DestI unordered, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation (atomic load & store) -- element size 8
define void @test8(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test8(
; CHECK: [[Sz:%[0-9]+]] = shl nuw i64 %Size, 3
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 8 %Dest{{[0-9]*}}, i8* align 8 %Base{{[0-9]*}}, i64 [[Sz]], i32 8)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i64, i32 10000
  %Dest = alloca i64, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i64, i64* %Base, i64 %indvar
  %DestI = getelementptr i64, i64* %Dest, i64 %indvar
  %V = load atomic i64, i64* %I.0.014 unordered, align 8
  store atomic i64 %V, i64* %DestI unordered, align 8
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (atomic load & store) -- element size 16
define void @test9(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test9(
; CHECK: [[Sz:%[0-9]+]] = shl nuw i64 %Size, 4
; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 16 %Dest{{[0-9]*}}, i8* align 16 %Base{{[0-9]*}}, i64 [[Sz]], i32 16)
; CHECK-NOT: store
; CHECK: ret void
bb.nph:
  %Base = alloca i128, i32 10000
  %Dest = alloca i128, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i128, i128* %Base, i64 %indvar
  %DestI = getelementptr i128, i128* %Dest, i64 %indvar
  %V = load atomic i128, i128* %I.0.014 unordered, align 16
  store atomic i128 %V, i128* %DestI unordered, align 16
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;; memcpy.atomic formation rejection (atomic load & store) -- element size 32
define void @test10(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test10(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i256, i32 10000
  %Dest = alloca i256, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i256, i256* %Base, i64 %indvar
  %DestI = getelementptr i256, i256* %Dest, i64 %indvar
  %V = load atomic i256, i256* %I.0.014 unordered, align 32
  store atomic i256 %V, i256* %DestI unordered, align 32
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; Make sure that atomic memset doesn't get recognized by mistake
define void @test_nomemset(i8* %Base, i64 %Size) nounwind ssp {
; CHECK-LABEL: @test_nomemset(
; CHECK-NOT: call void @llvm.memset
; CHECK: store
; CHECK: ret void
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  store atomic i8 0, i8* %I.0.014 unordered, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Verify that unordered memset_pattern isn't recognized.
; This is a replica of test11_pattern from basic.ll
define void @test_nomemset_pattern(i32* nocapture %P) nounwind ssp {
; CHECK-LABEL: @test_nomemset_pattern(
; CHECK-NEXT: entry:
; CHECK-NOT: bitcast
; CHECK-NOT: memset_pattern
; CHECK: store atomic
; CHECK: ret void
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32, i32* %P, i64 %indvar
  store atomic i32 1, i32* %arrayidx unordered, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
