; This file is used by first.ll, so it doesn't actually do anything itself
; RUN: true

%AnalysisResolver = type { i8, ptr }
%"DenseMap<P*,AU*>" = type { i64, ptr, i64, i64 }
%PMDataManager = type { i8, ptr, i8, i8, i8, i8, i8, i64, i8 }
%PMTopLevelManager = type { i8, i8, i8, i8, i8, i8, i8, i8, %"DenseMap<P*,AU*>" }
%P = type { i8, ptr, i64 }
%PI = type { i8, i8, i8, i8, i8, i8, %"vector<const PI*>", ptr }
%"SmallVImpl<const PI*>" = type { i8, ptr }
%"_V_base<const PI*>" = type { %"_V_base<const PI*>::_V_impl" }
%"_V_base<const PI*>::_V_impl" = type { ptr, i8, i8 }
%"pair<P*,AU*>" = type opaque
%"vector<const PI*>" = type { %"_V_base<const PI*>" }

define void @f(ptr %this) {
entry:
  %x = getelementptr inbounds %"SmallVImpl<const PI*>", ptr %this, i64 0, i32 1
  ret void
}
