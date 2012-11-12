; RUN: opt -dse -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin"

%"class.std::auto_ptr" = type { i32* }

; CHECK: @_Z3foov
define void @_Z3foov(%"class.std::auto_ptr"* noalias nocapture sret %agg.result) uwtable ssp {
_ZNSt8auto_ptrIiED1Ev.exit:
  %temp.lvalue = alloca %"class.std::auto_ptr", align 8
  call void @_Z3barv(%"class.std::auto_ptr"* sret %temp.lvalue)
  %_M_ptr.i.i = getelementptr inbounds %"class.std::auto_ptr"* %temp.lvalue, i64 0, i32 0
  %tmp.i.i = load i32** %_M_ptr.i.i, align 8, !tbaa !0
; CHECK-NOT: store i32* null
  store i32* null, i32** %_M_ptr.i.i, align 8, !tbaa !0
  %_M_ptr.i.i4 = getelementptr inbounds %"class.std::auto_ptr"* %agg.result, i64 0, i32 0
  store i32* %tmp.i.i, i32** %_M_ptr.i.i4, align 8, !tbaa !0
; CHECK: ret void
  ret void
}

declare void @_Z3barv(%"class.std::auto_ptr"* sret)

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
