; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-accesses -force-vector-width=4 -disable-output %s 2>&1 | FileCheck %s -check-prefix=LOOP-ACCESS
; RUN: opt -passes=loop-vectorize -debug-only=vectorutils -force-vector-width=4 -disable-output %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

%struct.foo = type { ptr, ptr, ptr }
%struct.pluto = type <{ %struct.wombat, %struct.spam, %struct.wibble, [6 x i8] }>
%struct.wombat = type { %struct.barney }
%struct.barney = type { %struct.widget }
%struct.widget = type { %struct.hoge }
%struct.hoge = type { %struct.pluto.0 }
%struct.pluto.0 = type { %struct.foo }
%struct.spam = type { %struct.barney.1 }
%struct.barney.1 = type { %struct.ham }
%struct.ham = type { %struct.bar }
%struct.bar = type { %struct.barney.2 }
%struct.barney.2 = type { %struct.hoge.3 }
%struct.hoge.3 = type { ptr, ptr, ptr }
%struct.wibble = type { %struct.spam.4 }
%struct.spam.4 = type { [2 x %struct.zot] }
%struct.zot = type { %struct.bar.5 }
%struct.bar.5 = type { i8 }
%struct.baz = type { i64, %struct.pluto }

; LOOP-ACCESS: Too many dependences, stopped recording

; If no dependences are recorded because there are too many, LoopAccessAnalysis
; just conservatively returns true for any pair of instructions compared (even
; those belonging to the same store group). This tests make sure that we do not
; incorrectly release a store group which had no dependences between its
; members, even if we have no dependences recorded because there are too many. 

; CHECK: LV: Creating an interleave group with:  store ptr null, ptr %phi5, align 8
; CHECK: LV: Inserted:  store ptr %load12, ptr %getelementptr11, align 8
; CHECK:     into the interleave group with  store ptr null, ptr %phi5
; CHECK: LV: Inserted:  store ptr %load7, ptr %getelementptr, align 8
; CHECK:     into the interleave group with  store ptr null, ptr %phi5

; CHECK: LV: Creating an interleave group with:  store ptr null, ptr %getelementptr13, align 8
; CHECK: LV: Inserted:  store ptr null, ptr %phi6, align 8
; CHECK:     into the interleave group with  store ptr null, ptr %getelementptr13
; CHECK: LV: Invalidated store group due to dependence between   store ptr %load7, ptr %getelementptr, align 8 and   store ptr null, ptr %getelementptr13, align 8
; CHECK-NOT: LV: Invalidated store group due to dependence between

; Note: The (only) invalidated store group is the one containing A (store ptr %load7, ptr %getelementptr, align 8) which is:
; Group with instructions:  
;   store ptr null, ptr %phi5, align 8
;   store ptr %load7, ptr %getelementptr, align 8
;   store ptr %load12, ptr %getelementptr11, align 8
define void @test(ptr %arg, ptr %arg1) local_unnamed_addr #0 {
bb:
  br label %bb2

bb2:                                              ; preds = %bb4, %bb
  %phi = phi ptr [ %arg, %bb ], [ %phi3, %bb4 ]
  %phi3 = phi ptr [ %arg1, %bb ], [ null, %bb4 ]
  br label %bb4

bb4:                                              ; preds = %bb4, %bb2
  %phi5 = phi ptr [ %getelementptr15, %bb4 ], [ %phi, %bb2 ]
  %phi6 = phi ptr [ %getelementptr14, %bb4 ], [ %phi3, %bb2 ]
  %load = load i64, ptr %phi5, align 8
  store i64 %load, ptr %phi, align 8
  store i64 0, ptr %phi3, align 8
  %load7 = load ptr, ptr %phi6, align 8
  %load8 = load ptr, ptr %phi5, align 8
  store ptr %load8, ptr %phi6, align 8
  %getelementptr = getelementptr %struct.foo, ptr %phi5, i64 0, i32 1
  %load9 = load ptr, ptr %phi5, align 8
  store ptr %load9, ptr %phi6, align 8
  %load10 = load ptr, ptr %phi5, align 8
  store ptr %load10, ptr %phi6, align 8
  store ptr null, ptr %phi5, align 8
  store ptr %load7, ptr %getelementptr, align 8
  %getelementptr11 = getelementptr %struct.pluto, ptr %phi5, i64 0, i32 1
  %load12 = load ptr, ptr %phi6, align 8
  %getelementptr13 = getelementptr %struct.pluto, ptr %phi6, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  store ptr null, ptr %phi6, align 8
  store ptr null, ptr %getelementptr13, align 8
  store ptr %load12, ptr %getelementptr11, align 8
  store ptr null, ptr %phi5, align 8
  %getelementptr14 = getelementptr inbounds %struct.baz, ptr %phi6, i64 1
  %getelementptr15 = getelementptr %struct.baz, ptr %phi5, i64 1
  %icmp = icmp eq ptr %phi6, %phi
  br i1 %icmp, label %bb2, label %bb4
}

; Function Attrs: memory(readwrite, inaccessiblemem: none)
declare void @foo() local_unnamed_addr #0

; Function Attrs: memory(argmem: readwrite)
declare void @pluto() local_unnamed_addr #1

attributes #0 = { memory(readwrite, inaccessiblemem: none) }
attributes #1 = { memory(argmem: readwrite) }
