; RUN: opt < %s -analyze -memoryssa -enable-drf-memoryssa 2>&1 | FileCheck %s
; RUN: opt -disable-output < %s -passes='print<memoryssa>' -enable-drf-memoryssa 2>&1 | FileCheck %s

@x = common local_unnamed_addr global i32 0, align 4

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %syncreg = call token @llvm.syncregion.start()
  store i32 5, i32* @x, align 4, !tbaa !2
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = load i32, i32* @x, align 4, !tbaa !2
  call void @print(i32 %0)
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  %1 = load i32, i32* @x, align 4, !tbaa !2
  call void @print(i32 %1)
  ret i32 0
}

; CHECK: entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 5, i32* @x

; CHECK: det.cont:
; CHECK: MemoryUse(1)
; CHECK-NEXT: %1 = load i32, i32* @x

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @print(i32) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }

!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
