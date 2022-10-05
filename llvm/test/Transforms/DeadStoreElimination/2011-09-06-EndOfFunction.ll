; RUN: opt -dse -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin"

%"class.std::auto_ptr" = type { ptr }

; CHECK-LABEL: @_Z3foov(
define void @_Z3foov(ptr noalias nocapture sret(%"class.std::auto_ptr") %agg.result) uwtable ssp {
_ZNSt8auto_ptrIiED1Ev.exit:
  %temp.lvalue = alloca %"class.std::auto_ptr", align 8
  call void @_Z3barv(ptr sret(%"class.std::auto_ptr") %temp.lvalue)
  %tmp.i.i = load ptr, ptr %temp.lvalue, align 8
; CHECK-NOT: store ptr null
  store ptr null, ptr %temp.lvalue, align 8
  store ptr %tmp.i.i, ptr %agg.result, align 8
; CHECK: ret void
  ret void
}

declare void @_Z3barv(ptr sret(%"class.std::auto_ptr"))
