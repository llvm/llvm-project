; RUN: opt < %s -early-cse-rhino -S -o - | FileCheck %s

@x = common local_unnamed_addr global i32 0, align 4

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %syncreg = call token @llvm.syncregion.start()
  store i32 5, i32* @x, align 4, !tbaa !2
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
; CHECK: det.achd
; CHECK-NOT: load i32
  %0 = load i32, i32* @x, align 4, !tbaa !2
  call void @print(i32 %0)
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
; CHECK: det.cont
; CHECK-NOT: load i32
  %1 = load i32, i32* @x, align 4, !tbaa !2
  call void @print(i32 %1)
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @print(i32) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (https://github.com/wsmoses/Cilk-Clang 6eb58f732f8b19addc25692083a8268ace6528fd) (git@github.com:wsmoses/Parallel-IR c0d5aa383cfc1021e28074c5defdc5ab12130123)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
