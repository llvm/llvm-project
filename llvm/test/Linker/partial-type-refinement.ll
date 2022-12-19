; RUN: llvm-link %s %p/partial-type-refinement-link.ll -S | FileCheck %s
; PR4954

; CHECK: load ptr, ptr getelementptr inbounds (%"RegisterP<LowerArrayLength>", ptr @_ZN3mvmL1XE, i64 0, i32 0, i32 6, i32 0, i32 0, i32 0), align 16

%AnalysisResolver = type { i8, ptr }
%"DenseMap<P*,AU*>" = type { i64, ptr, i64, i64 }
%PMDataManager = type { i8, ptr, i8, i8, i8, i8, i8, i64, i8 }
%PMTopLevelManager = type { i8, i8, i8, i8, i8, i8, i8, i8, %"DenseMap<P*,AU*>" }
%P = type { i8, ptr, i64 }
%PI = type { i8, i8, i8, i8, i8, i8, %"vector<const PI*>", ptr }
%"RegisterP<LowerArrayLength>" = type { %PI }
%"_V_base<const PI*>" = type { %"_V_base<const PI*>::_V_impl" }
%"_V_base<const PI*>::_V_impl" = type { ptr, i8, i8 }
%"pair<P*,AU*>" = type opaque
%"vector<const PI*>" = type { %"_V_base<const PI*>" }

@_ZN3mvmL1XE = external global %"RegisterP<LowerArrayLength>"

define void @__tcf_0() nounwind {
entry:
  %0 = load ptr, ptr getelementptr inbounds (%"RegisterP<LowerArrayLength>", ptr @_ZN3mvmL1XE, i64 0, i32 0, i32 6, i32 0, i32 0, i32 0), align 16
  ret void
}
