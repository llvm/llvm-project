; RUN: opt < %s -basicaa -sink -S | FileCheck %s

declare i32 @f_load_global() nounwind readonly
declare i32 @f_load_arg(i32*) nounwind readonly argmemonly
declare void @f_store_global(i32) nounwind
declare void @f_store_arg(i32*) nounwind argmemonly
declare void @f_readonly_arg(i32* readonly, i32*) nounwind argmemonly
declare i32 @f_readnone(i32) nounwind readnone

@A = external global i32
@B = external global i32

; Sink readonly call if no stores are in the way.
;
; CHECK-LABEL: @test_sink_no_stores(
; CHECK: true:
; CHECK-NEXT: %l = call i32 @f_load_global
; CHECK-NEXT: ret i32 %l
define i32 @test_sink_no_stores(i1 %z) {
  %l = call i32 @f_load_global()
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; CHECK-LABEL: @test_sink_argmem_store(
; CHECK: true:
; CHECK-NEXT: %l = call i32 @f_load_arg
; CHECK-NEXT: ret i32 %l
define i32 @test_sink_argmem_store(i1 %z) {
  %l = call i32 @f_load_arg(i32* @A)
  store i32 0, i32* @B
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; CHECK-LABEL: @test_sink_argmem_call(
; CHECK: true:
; CHECK-NEXT: %l = call i32 @f_load_arg
; CHECK-NEXT: ret i32 %l
define i32 @test_sink_argmem_call(i1 %z) {
  %l = call i32 @f_load_arg(i32* @A)
  call void @f_store_arg(i32* @B)
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; CHECK-LABEL: @test_sink_argmem_multiple(
; CHECK: true:
; CHECK-NEXT: %l = call i32 @f_load_arg
; CHECK-NEXT: ret i32 %l
define i32 @test_sink_argmem_multiple(i1 %z) {
  %l = call i32 @f_load_arg(i32* @A)
  call void @f_readonly_arg(i32* @A, i32* @B)
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; But don't sink if there is a store.
;
; CHECK-LABEL: @test_nosink_store(
; CHECK: call i32 @f_load_global
; CHECK-NEXT: store i32
define i32 @test_nosink_store(i1 %z) {
  %l = call i32 @f_load_global()
  store i32 0, i32* @A
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; CHECK-LABEL: @test_nosink_call(
; CHECK: call i32 @f_load_global
; CHECK-NEXT: call void @f_store_global
define i32 @test_nosink_call(i1 %z) {
  %l = call i32 @f_load_global()
  call void @f_store_global(i32 0)
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; readnone calls are sunk across stores.
;
; CHECK-LABEL: @test_sink_readnone(
; CHECK: true:
; CHECK-NEXT: %l = call i32 @f_readnone(
; CHECK-NEXT: ret i32 %l
define i32 @test_sink_readnone(i1 %z) {
  %l = call i32 @f_readnone(i32 0)
  store i32 0, i32* @A
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}
