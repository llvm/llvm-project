; RUN: opt -S < %s -basicaa -memcpyopt | FileCheck %s
; <rdar://problem/8536696>

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%"class.std::auto_ptr" = type { i32* }

; CHECK-LABEL: @_Z3foov(
define void @_Z3foov(%"class.std::auto_ptr"* noalias nocapture sret %agg.result) ssp {
_ZNSt8auto_ptrIiED1Ev.exit:
  %temp.lvalue = alloca %"class.std::auto_ptr", align 8
; CHECK: call void @_Z3barv(%"class.std::auto_ptr"* sret %agg.result)
  call void @_Z3barv(%"class.std::auto_ptr"* sret %temp.lvalue)
  %tmp.i.i = getelementptr inbounds %"class.std::auto_ptr", %"class.std::auto_ptr"* %temp.lvalue, i64 0, i32 0
; CHECK-NOT: load
  %tmp2.i.i = load i32*, i32** %tmp.i.i, align 8
  %tmp.i.i4 = getelementptr inbounds %"class.std::auto_ptr", %"class.std::auto_ptr"* %agg.result, i64 0, i32 0
; CHECK-NOT: store
  store i32* %tmp2.i.i, i32** %tmp.i.i4, align 8
; CHECK: ret void
  ret void
}

declare void @_Z3barv(%"class.std::auto_ptr"* nocapture sret) nounwind
