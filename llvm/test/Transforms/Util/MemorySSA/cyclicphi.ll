; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s

%struct.hoge = type { i32, %struct.widget }
%struct.widget = type { i64 }

define hidden void @quux(%struct.hoge *%f) align 2 {
  %tmp = getelementptr inbounds %struct.hoge, %struct.hoge* %f, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, %struct.hoge* %f, i64 0, i32 1
  %tmp25 = bitcast %struct.widget* %tmp24 to i64**
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK:  3 = MemoryPhi({%0,liveOnEntry},{bb77,2})
; CHECK-NEXT:   br i1 undef, label %bb68, label %bb77
  br i1 undef, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK:  MemoryUse(liveOnEntry)
; CHECK-NEXT:   %tmp69 = load i64, i64* null, align 8
  %tmp69 = load i64, i64* null, align 8
; CHECK:  1 = MemoryDef(3)
; CHECK-NEXT:   store i64 %tmp69, i64* %tmp, align 8
  store i64 %tmp69, i64* %tmp, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK:  2 = MemoryPhi({bb26,3},{bb68,1})
; CHECK:  MemoryUse(2)
; CHECK-NEXT:   %tmp78 = load i64*, i64** %tmp25, align 8
  %tmp78 = load i64*, i64** %tmp25, align 8
  %tmp79 = getelementptr inbounds i64, i64* %tmp78, i64 undef
  br label %bb26
}
