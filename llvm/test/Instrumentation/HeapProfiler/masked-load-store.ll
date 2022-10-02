; RUN: opt < %s -passes=memprof -memprof-use-callbacks -S \
; RUN:     | FileCheck %s -check-prefix=LOAD -check-prefix=STORE -check-prefix=ALL
; RUN: opt < %s -passes=memprof -memprof-use-callbacks -memprof-instrument-reads=0 -S \
; RUN:     | FileCheck %s -check-prefix=NOLOAD -check-prefix=STORE -check-prefix=ALL
; RUN: opt < %s -passes=memprof -memprof-use-callbacks -memprof-instrument-writes=0 -S \
; RUN:     | FileCheck %s -check-prefix=LOAD -check-prefix=NOSTORE -check-prefix=ALL
; RUN: opt < %s -passes=memprof -memprof-use-callbacks -memprof-instrument-reads=0 -memprof-instrument-writes=0 -S \
; RUN:     | FileCheck %s -check-prefix=NOLOAD -check-prefix=NOSTORE -check-prefix=ALL
; Support memory profiling instrumentation for constant-mask llvm.masked.{load,store}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@v4f32 = global <4 x float>* zeroinitializer, align 8
@v8i32 = global <8 x i32>* zeroinitializer, align 8
@v4i64 = global <4 x i32*>* zeroinitializer, align 8

;;;;;;;;;;;;;;;; STORE
declare void @llvm.masked.store.v4f32.p0v4f32(<4 x float>, <4 x float>*, i32, <4 x i1>) argmemonly nounwind
declare void @llvm.masked.store.v8i32.p0v8i32(<8 x i32>, <8 x i32>*, i32, <8 x i1>) argmemonly nounwind
declare void @llvm.masked.store.v4p0i32.p0v4p0i32(<4 x i32*>, <4 x i32*>*, i32, <4 x i1>) argmemonly nounwind

define void @store.v4f32.1110(<4 x float> %arg) {
; ALL-LABEL: @store.v4f32.1110
  %p = load <4 x float>*, <4 x float>** @v4f32, align 8
; NOSTORE-NOT: call void @__memprof_store
; STORE: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 0
; STORE: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP0]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP0]])
; STORE: [[GEP1:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 1
; STORE: [[PGEP1:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP1]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP1]])
; STORE: [[GEP2:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 2
; STORE: [[PGEP2:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP2]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP2]])
; STORE: tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 false>)
  tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @store.v8i32.10010110(<8 x i32> %arg) {
; ALL-LABEL: @store.v8i32.10010110
  %p = load <8 x i32>*, <8 x i32>** @v8i32, align 8
; NOSTORE-NOT: call void @__memprof_store
; STORE: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 0
; STORE: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP0]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP0]])
; STORE: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 3
; STORE: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP3]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP3]])
; STORE: [[GEP5:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 5
; STORE: [[PGEP5:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP5]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP5]])
; STORE: [[GEP6:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 6
; STORE: [[PGEP6:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP6]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP6]])
; STORE: tail call void @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %arg, <8 x i32>* %p, i32 8, <8 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false>)
  tail call void @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %arg, <8 x i32>* %p, i32 8, <8 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false>)
  ret void
}

define void @store.v4i64.0001(<4 x i32*> %arg) {
; ALL-LABEL: @store.v4i64.0001
  %p = load <4 x i32*>*, <4 x i32*>** @v4i64, align 8
; NOSTORE-NOT: call void @__memprof_store
; STORE: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <4 x i32*>, <4 x i32*>* %p, i64 0, i64 3
; STORE: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint i32** [[GEP3]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP3]])
; STORE: tail call void @llvm.masked.store.v4p0i32.p0v4p0i32(<4 x i32*> %arg, <4 x i32*>* %p, i32 8, <4 x i1> <i1 false, i1 false, i1 false, i1 true>)
  tail call void @llvm.masked.store.v4p0i32.p0v4p0i32(<4 x i32*> %arg, <4 x i32*>* %p, i32 8, <4 x i1> <i1 false, i1 false, i1 false, i1 true>)
  ret void
}

define void @store.v4f32.variable(<4 x float> %arg, <4 x i1> %mask) {
; ALL-LABEL: @store.v4f32.variable
  %p = load <4 x float>*, <4 x float>** @v4f32, align 8
; STORE: [[MASK0:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 0
; STORE: br i1 [[MASK0]], label %[[THEN0:[0-9A-Za-z]+]], label %[[AFTER0:[0-9A-Za-z]+]]
; STORE: [[THEN0]]:
; STORE: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 0
; STORE: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP0]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP0]])
; STORE: br label %[[AFTER0]]
; STORE: [[AFTER0]]:

; STORE: [[MASK1:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 1
; STORE: br i1 [[MASK1]], label %[[THEN1:[0-9A-Za-z]+]], label %[[AFTER1:[0-9A-Za-z]+]]
; STORE: [[THEN1]]:
; STORE: [[GEP1:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 1
; STORE: [[PGEP1:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP1]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP1]])
; STORE: br label %[[AFTER1]]
; STORE: [[AFTER1]]:

; STORE: [[MASK2:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 2
; STORE: br i1 [[MASK2]], label %[[THEN2:[0-9A-Za-z]+]], label %[[AFTER2:[0-9A-Za-z]+]]
; STORE: [[THEN2]]:
; STORE: [[GEP2:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 2
; STORE: [[PGEP2:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP2]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP2]])
; STORE: br label %[[AFTER2]]
; STORE: [[AFTER2]]:

; STORE: [[MASK3:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 3
; STORE: br i1 [[MASK3]], label %[[THEN3:[0-9A-Za-z]+]], label %[[AFTER3:[0-9A-Za-z]+]]
; STORE: [[THEN3]]:
; STORE: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 3
; STORE: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP3]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP3]])
; STORE: br label %[[AFTER3]]
; STORE: [[AFTER3]]:

; STORE: tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> %mask)
  tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> %mask)
  ret void
}

;; Store using two masked.stores, which should instrument them both.
define void @store.v4f32.1010.split(<4 x float> %arg) {
; BOTH-LABEL: @store.v4f32.1010.split
  %p = load <4 x float>*, <4 x float>** @v4f32, align 8
; STORE: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 0
; STORE: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP0]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP0]])
; STORE: tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 false, i1 false, i1 false>)
  tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 false, i1 false, i1 false>)
; STORE: [[GEP1:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 2
; STORE: [[PGEP1:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP1]] to i64
; STORE: call void @__memprof_store(i64 [[PGEP1]])
; STORE: tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> <i1 false, i1 false, i1 true, i1 false>)
  tail call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %arg, <4 x float>* %p, i32 4, <4 x i1> <i1 false, i1 false, i1 true, i1 false>)
  ret void
}

;;;;;;;;;;;;;;;; LOAD
declare <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>) argmemonly nounwind
declare <8 x i32> @llvm.masked.load.v8i32.p0v8i32(<8 x i32>*, i32, <8 x i1>, <8 x i32>) argmemonly nounwind
declare <4 x i32*> @llvm.masked.load.v4p0i32.p0v4p0i32(<4 x i32*>*, i32, <4 x i1>, <4 x i32*>) argmemonly nounwind

define <8 x i32> @load.v8i32.11100001(<8 x i32> %arg) {
; ALL-LABEL: @load.v8i32.11100001
  %p = load <8 x i32>*, <8 x i32>** @v8i32, align 8
; NOLOAD-NOT: call void @__memprof_load
; LOAD: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 0
; LOAD: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP0]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP0]])
; LOAD: [[GEP1:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 1
; LOAD: [[PGEP1:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP1]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP1]])
; LOAD: [[GEP2:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 2
; LOAD: [[PGEP2:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP2]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP2]])
; LOAD: [[GEP7:%[0-9A-Za-z]+]] = getelementptr <8 x i32>, <8 x i32>* %p, i64 0, i64 7
; LOAD: [[PGEP7:%[0-9A-Za-z]+]] = ptrtoint i32* [[GEP7]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP7]])
; LOAD: tail call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %p, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 true>, <8 x i32> %arg)
  %res = tail call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %p, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 true>, <8 x i32> %arg)
  ret <8 x i32> %res
}

define <4 x float> @load.v4f32.1001(<4 x float> %arg) {
; ALL-LABEL: @load.v4f32.1001
  %p = load <4 x float>*, <4 x float>** @v4f32, align 8
; NOLOAD-NOT: call void @__memprof_load
; LOAD: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 0
; LOAD: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP0]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP0]])
; LOAD: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 3
; LOAD: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP3]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP3]])
; LOAD: tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x float> %arg)
  %res = tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x float> %arg)
  ret <4 x float> %res
}

define <4 x i32*> @load.v4i64.0001(<4 x i32*> %arg) {
; ALL-LABEL: @load.v4i64.0001
  %p = load <4 x i32*>*, <4 x i32*>** @v4i64, align 8
; NOLOAD-NOT: call void @__memprof_load
; LOAD: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <4 x i32*>, <4 x i32*>* %p, i64 0, i64 3
; LOAD: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint i32** [[GEP3]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP3]])
; LOAD: tail call <4 x i32*> @llvm.masked.load.v4p0i32.p0v4p0i32(<4 x i32*>* %p, i32 8, <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32*> %arg)
  %res = tail call <4 x i32*> @llvm.masked.load.v4p0i32.p0v4p0i32(<4 x i32*>* %p, i32 8, <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32*> %arg)
  ret <4 x i32*> %res
}

define <4 x float> @load.v4f32.variable(<4 x float> %arg, <4 x i1> %mask) {
; ALL-LABEL: @load.v4f32.variable
  %p = load <4 x float>*, <4 x float>** @v4f32, align 8
; LOAD: [[MASK0:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 0
; LOAD: br i1 [[MASK0]], label %[[THEN0:[0-9A-Za-z]+]], label %[[AFTER0:[0-9A-Za-z]+]]
; LOAD: [[THEN0]]:
; LOAD: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 0
; LOAD: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP0]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP0]])
; LOAD: br label %[[AFTER0]]
; LOAD: [[AFTER0]]:

; LOAD: [[MASK1:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 1
; LOAD: br i1 [[MASK1]], label %[[THEN1:[0-9A-Za-z]+]], label %[[AFTER1:[0-9A-Za-z]+]]
; LOAD: [[THEN1]]:
; LOAD: [[GEP1:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 1
; LOAD: [[PGEP1:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP1]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP1]])
; LOAD: br label %[[AFTER1]]
; LOAD: [[AFTER1]]:

; LOAD: [[MASK2:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 2
; LOAD: br i1 [[MASK2]], label %[[THEN2:[0-9A-Za-z]+]], label %[[AFTER2:[0-9A-Za-z]+]]
; LOAD: [[THEN2]]:
; LOAD: [[GEP2:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 2
; LOAD: [[PGEP2:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP2]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP2]])
; LOAD: br label %[[AFTER2]]
; LOAD: [[AFTER2]]:

; LOAD: [[MASK3:%[0-9A-Za-z]+]] = extractelement <4 x i1> %mask, i64 3
; LOAD: br i1 [[MASK3]], label %[[THEN3:[0-9A-Za-z]+]], label %[[AFTER3:[0-9A-Za-z]+]]
; LOAD: [[THEN3]]:
; LOAD: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 3
; LOAD: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP3]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP3]])
; LOAD: br label %[[AFTER3]]
; LOAD: [[AFTER3]]:

; LOAD: tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> %mask, <4 x float> %arg)
  %res = tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> %mask, <4 x float> %arg)
  ret <4 x float> %res
}

;; Load using two masked.loads, which should instrument them both.
define <4 x float> @load.v4f32.1001.split(<4 x float> %arg) {
; BOTH-LABEL: @load.v4f32.1001
  %p = load <4 x float>*, <4 x float>** @v4f32, align 8
; LOAD: [[GEP0:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 0
; LOAD: [[PGEP0:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP0]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP0]])
; LOAD: %res = tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %arg)
  %res = tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %arg)
; LOAD: [[GEP3:%[0-9A-Za-z]+]] = getelementptr <4 x float>, <4 x float>* %p, i64 0, i64 3
; LOAD: [[PGEP3:%[0-9A-Za-z]+]] = ptrtoint float* [[GEP3]] to i64
; LOAD: call void @__memprof_load(i64 [[PGEP3]])
; LOAD: tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x float> %res)
  %res2 = tail call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %p, i32 4, <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x float> %res)
  ret <4 x float> %res2
}
