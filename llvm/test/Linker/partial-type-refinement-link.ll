; This file is used by first.ll, so it doesn't actually do anything itself
; RUN: true

%AnalysisResolver = type { i8, %PMDataManager* }
%"DenseMap<P*,AU*>" = type { i64, %"pair<P*,AU*>"*, i64, i64 }
%PMDataManager = type { i8, %PMTopLevelManager*, i8, i8, i8, i8, i8, i64, i8 }
%PMTopLevelManager = type { i8, i8, i8, i8, i8, i8, i8, i8, %"DenseMap<P*,AU*>" }
%P = type { i8, %AnalysisResolver*, i64 }
%PI = type { i8, i8, i8, i8, i8, i8, %"vector<const PI*>", %P* }
%"SmallVImpl<const PI*>" = type { i8, %PI* }
%"_V_base<const PI*>" = type { %"_V_base<const PI*>::_V_impl" }
%"_V_base<const PI*>::_V_impl" = type { %PI*, i8, i8 }
%"pair<P*,AU*>" = type opaque
%"vector<const PI*>" = type { %"_V_base<const PI*>" }

define void @f(%"SmallVImpl<const PI*>"* %this) {
entry:
  %x = getelementptr inbounds %"SmallVImpl<const PI*>"* %this, i64 0, i32 1
  ret void
}
