; RUN: opt -passes=inferattrs,function-attrs -S %s | FileCheck %s

declare void @decl_readonly() readonly
declare void @decl_readnone() readnone
declare void @decl_argmemonly(ptr) argmemonly
declare void @decl_unknown()

define void @test_fn_mustprogress(ptr %ptr) mustprogress {
; CHECK: Function Attrs: mustprogress
; CHECK-LABEL: @test_fn_mustprogress(
; CHECK-NOT:     call void @decl_readonly() #
; CHECK-NOT:     call void @decl_readnone() #
; CHECK-NOT:     call void @decl_unknown() #
; CHECK-NOT:     call void @decl_argmemonly(ptr [[PTR:%.*]]) #
; CHECK:         ret void
;
  call void @decl_readonly()
  call void @decl_readnone()
  call void @decl_unknown()
  call void @decl_argmemonly(ptr %ptr)
  ret void
}

define void @test_fn_willreturn(ptr %ptr) willreturn {
; CHECK: Function Attrs: mustprogress willreturn
; CHECK-LABEL: @test_fn_willreturn(
; CHECK-NOT:     call void @decl_readonly() #
; CHECK-NOT:    call void @decl_readnone() #
; CHECK-NOT:     call void @decl_unknown() #
; CHECK-NOT:     call void @decl_argmemonly(ptr [[PTR:%.*]]) #
; CHECK:         ret void
;
  call void @decl_readonly()
  call void @decl_readnone()
  call void @decl_unknown()
  call void @decl_argmemonly(ptr %ptr)
  ret void
}

define void @test_fn_mustprogress_readonly_calls(ptr %ptr) mustprogress {
; CHECK: Function Attrs: mustprogress nofree willreturn memory(read)
; CHECK-LABEL: @test_fn_mustprogress_readonly_calls(
; CHECK-NOT:     call void @decl_readonly() #
; CHECK-NOT:     call void @decl_readnone() #
; CHECK:         ret void
;
  call void @decl_readonly()
  call void @decl_readnone()
  ret void
}

define void @test_fn_mustprogress_readonly_calls_but_stores(ptr %ptr) mustprogress {
; CHECK: Function Attrs: mustprogress nofree
; CHECK-LABEL: @test_fn_mustprogress_readonly_calls_but_stores(
; CHECK-NOT:     call void @decl_readonly() #
; CHECK-NOT:     call void @decl_readnone() #
; CHECK:         store i32 0, ptr [[PTR:%.*]], align 4
; CHECK-NEXT:    ret void
;
  call void @decl_readonly()
  call void @decl_readnone()
  store i32 0, ptr %ptr
  ret void
}
