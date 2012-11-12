; RUN: opt -basicaa -functionattrs -S < %s | FileCheck %s

; Atomic load/store to local doesn't affect whether a function is
; readnone/readonly.
define i32 @test1(i32 %x) uwtable ssp {
; CHECK: define i32 @test1(i32 %x) uwtable readnone ssp {
entry:
  %x.addr = alloca i32, align 4
  store atomic i32 %x, i32* %x.addr seq_cst, align 4
  %r = load atomic i32* %x.addr seq_cst, align 4
  ret i32 %r
}

; A function with an Acquire load is not readonly.
define i32 @test2(i32* %x) uwtable ssp {
; CHECK: define i32 @test2(i32* nocapture %x) uwtable ssp {
entry:
  %r = load atomic i32* %x seq_cst, align 4
  ret i32 %r
}

