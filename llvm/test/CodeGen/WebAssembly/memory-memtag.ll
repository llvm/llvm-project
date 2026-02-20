; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test memory tagging builtin on wasm64

target triple = "wasm64-unknown-unknown"

; CHECK-LABEL: memory_randomtag:
; CHECK-NEXT: .functype memory_randomtag (i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: memory.randomtag	
; CHECK-NEXT: end_function
define ptr @memory_randomtag(ptr %p) {
  %1 = call ptr @llvm.wasm.memory.randomtag(ptr %p)
  ret ptr %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.randomtag(ptr)

; CHECK-LABEL: memory_copytag:
; CHECK-NEXT: .functype memory_copytag (i64, i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.copytag	
; CHECK-NEXT: end_function
define ptr @memory_copytag(ptr %p0, ptr %p1) {
  %1 = call ptr @llvm.wasm.memory.copytag(ptr %p0, ptr %p1)
  ret ptr %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.copytag(ptr, ptr)

; CHECK-LABEL: memory_subtag:
; CHECK-NEXT: .functype memory_subtag (i64, i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.subtag	
; CHECK-NEXT: end_function
define i64 @memory_subtag(ptr %p0, ptr %p1) {
  %1 = call i64 @llvm.wasm.memory.subtag(ptr %p0, ptr %p1)
  ret i64 %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare i64 @llvm.wasm.memory.subtag(ptr, ptr)

; CHECK-LABEL: memory_loadtag:
; CHECK-NEXT: .functype memory_loadtag (i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: memory.loadtag	
; CHECK-NEXT: end_function
define ptr @memory_loadtag(ptr %p0) {
  %1 = call ptr @llvm.wasm.memory.loadtag(ptr %p0)
  ret ptr %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.loadtag(ptr)

; CHECK-LABEL: memory_storetag:
; CHECK-NEXT: .functype memory_storetag (i64, i64) -> ()
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.storetag	
; CHECK-NEXT: end_function
define void @memory_storetag(ptr %p0, i64 %b16) {
  call void @llvm.wasm.memory.storetag(ptr %p0, i64 %b16)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.wasm.memory.storetag(ptr, i64)

; CHECK-LABEL: memory_storeztag:
; CHECK-NEXT: .functype memory_storeztag (i64, i64) -> ()
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.storeztag	
; CHECK-NEXT: end_function
define void @memory_storeztag(ptr %p0, i64 %b16) {
  call void @llvm.wasm.memory.storeztag(ptr %p0, i64 %b16)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.wasm.memory.storeztag(ptr, i64)

; CHECK-LABEL: memory_store1tag:
; CHECK-NEXT: .functype memory_store1tag (i64, i64) -> ()
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.store1tag	
; CHECK-NEXT: end_function
define void @memory_store1tag(ptr %p0, i64 %b16) {
  call void @llvm.wasm.memory.store1tag(ptr %p0, i64 %b16)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.wasm.memory.store1tag(ptr, i64)

; CHECK-LABEL: memory_storez1tag:
; CHECK-NEXT: .functype memory_storez1tag (i64, i64) -> ()
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.storez1tag	
; CHECK-NEXT: end_function
define void @memory_storez1tag(ptr %p0, i64 %b16) {
  call void @llvm.wasm.memory.storez1tag(ptr %p0, i64 %b16)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.wasm.memory.storez1tag(ptr, i64)

; CHECK-LABEL: memory_randomstoretag:
; CHECK-NEXT: .functype memory_randomstoretag (i64, i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.randomstoretag	
; CHECK-NEXT: local.get	0
; CHECK-NEXT: end_function
define ptr @memory_randomstoretag(ptr %p0, i64 %b16) {
  call ptr @llvm.wasm.memory.randomstoretag(ptr %p0, i64 %b16)
  ret ptr
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.randomstoretag(ptr, i64)

; CHECK-LABEL: memory_randomstoreztag:
; CHECK-NEXT: .functype memory_randomstoreztag (i64, i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.randomstoreztag	
; CHECK-NEXT: local.get	0
; CHECK-NEXT: end_function
define ptr @memory_randomstoreztag(ptr %p0, i64 %b16) {
  call ptr @llvm.wasm.memory.randomstoreztag(ptr %p0, i64 %b16)
  ret ptr
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.randomstoreztag(ptr, i64)

; CHECK-LABEL: memory_randomstore1tag:
; CHECK-NEXT: .functype memory_randomstore1tag (i64, i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.randomstore1tag	
; CHECK-NEXT: local.get	0
; CHECK-NEXT: end_function
define ptr @memory_randomstore1tag(ptr %p0, i64 %b16) {
  call ptr @llvm.wasm.memory.randomstore1tag(ptr %p0, i64 %b16)
  ret ptr
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.randomstore1tag(ptr, i64)

; CHECK-LABEL: memory_randomstorez1tag:
; CHECK-NEXT: .functype memory_randomstorez1tag (i64, i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: local.get	1
; CHECK-NEXT: memory.randomstorez1tag	
; CHECK-NEXT: local.get	0
; CHECK-NEXT: end_function
define ptr @memory_randomstorez1tag(ptr %p0, i64 %b16) {
  call ptr @llvm.wasm.memory.randomstorez1tag(ptr %p0, i64 %b16)
  ret ptr
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.randomstorez1tag(ptr, i64)
