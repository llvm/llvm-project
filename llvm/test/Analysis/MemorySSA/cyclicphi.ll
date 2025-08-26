; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s

%struct.hoge = type { i32, %struct.widget }
%struct.widget = type { i64 }

define hidden void @quux(ptr %f, i1 %arg) align 2 {
  %tmp = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK:  3 = MemoryPhi({%0,liveOnEntry},{bb77,2})
; CHECK-NEXT:   br i1 %arg, label %bb68, label %bb77
  br i1 %arg, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK:  MemoryUse(liveOnEntry)
; CHECK-NEXT:   %tmp69 = load i64, ptr null, align 8
  %tmp69 = load i64, ptr null, align 8
; CHECK:  1 = MemoryDef(3)
; CHECK-NEXT:   store i64 %tmp69, ptr %tmp, align 8
  store i64 %tmp69, ptr %tmp, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK:  2 = MemoryPhi({bb26,3},{bb68,1})
; CHECK:  MemoryUse(2)
; CHECK-NEXT:   %tmp78 = load ptr, ptr %tmp24, align 8
  %tmp78 = load ptr, ptr %tmp24, align 8
  %tmp79 = getelementptr inbounds i64, ptr %tmp78, i64 undef
  br label %bb26
}

define hidden void @quux_no_null_opt(ptr %f, i1 %arg) align 2 #0 {
; CHECK-LABEL: quux_no_null_opt(
  %tmp = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK:  3 = MemoryPhi({%0,liveOnEntry},{bb77,2})
; CHECK-NEXT:   br i1 %arg, label %bb68, label %bb77
  br i1 %arg, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK:  MemoryUse(3)
; CHECK-NEXT:   %tmp69 = load i64, ptr null, align 8
  %tmp69 = load i64, ptr null, align 8
; CHECK:  1 = MemoryDef(3)
; CHECK-NEXT:   store i64 %tmp69, ptr %tmp, align 8
  store i64 %tmp69, ptr %tmp, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK:  2 = MemoryPhi({bb26,3},{bb68,1})
; CHECK:  MemoryUse(2)
; CHECK-NEXT:   %tmp78 = load ptr, ptr %tmp24, align 8
  %tmp78 = load ptr, ptr %tmp24, align 8
  %tmp79 = getelementptr inbounds i64, ptr %tmp78, i64 undef
  br label %bb26
}

; CHECK-LABEL: define void @quux_skip
define void @quux_skip(ptr noalias %f, ptr noalias %g, i1 %arg) align 2 {
  %tmp = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK: 3 = MemoryPhi({%0,liveOnEntry},{bb77,2})
; CHECK-NEXT: br i1 %arg, label %bb68, label %bb77
  br i1 %arg, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK: MemoryUse(3)
; CHECK-NEXT: %tmp69 = load i64, ptr %g, align 8
  %tmp69 = load i64, ptr %g, align 8
; CHECK: 1 = MemoryDef(3)
; CHECK-NEXT: store i64 %tmp69, ptr %g, align 8
  store i64 %tmp69, ptr %g, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK: 2 = MemoryPhi({bb26,3},{bb68,1})
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %tmp78 = load ptr, ptr %tmp24, align 8
  %tmp78 = load ptr, ptr %tmp24, align 8
  br label %bb26
}

; CHECK-LABEL: define void @quux_dominated
define void @quux_dominated(ptr noalias %f, ptr noalias %g, i1 %arg) align 2 {
  %tmp = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK: 3 = MemoryPhi({%0,liveOnEntry},{bb77,2})
; CHECK: MemoryUse(3)
; CHECK-NEXT: load ptr, ptr %tmp24, align 8
  load ptr, ptr %tmp24, align 8
  br i1 %arg, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK: MemoryUse(3)
; CHECK-NEXT: %tmp69 = load i64, ptr %g, align 8
  %tmp69 = load i64, ptr %g, align 8
; CHECK: 1 = MemoryDef(3)
; CHECK-NEXT: store i64 %tmp69, ptr %g, align 8
  store i64 %tmp69, ptr %g, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK: 4 = MemoryPhi({bb26,3},{bb68,1})
; CHECK: 2 = MemoryDef(4)
; CHECK-NEXT: store ptr null, ptr %tmp24, align 8
  store ptr null, ptr %tmp24, align 8
  br label %bb26
}

; CHECK-LABEL: define void @quux_nodominate
define void @quux_nodominate(ptr noalias %f, ptr noalias %g, i1 %arg) align 2 {
  %tmp = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, ptr %f, i64 0, i32 1
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK: 3 = MemoryPhi({%0,liveOnEntry},{bb77,2})
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load ptr, ptr %tmp24, align 8
  load ptr, ptr %tmp24, align 8
  br i1 %arg, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK: MemoryUse(3)
; CHECK-NEXT: %tmp69 = load i64, ptr %g, align 8
  %tmp69 = load i64, ptr %g, align 8
; CHECK: 1 = MemoryDef(3)
; CHECK-NEXT: store i64 %tmp69, ptr %g, align 8
  store i64 %tmp69, ptr %g, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK: 2 = MemoryPhi({bb26,3},{bb68,1})
; CHECK-NEXT: br label %bb26
  br label %bb26
}

attributes #0 = { null_pointer_is_valid }
