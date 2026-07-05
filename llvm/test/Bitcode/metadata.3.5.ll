; RUN: llvm-dis < %s.bc | FileCheck %s

; Check that metadata encoded in 3.5 is correctly understood going forward.
;
; Bitcode assembled by llvm-as v3.5.0.

define void @foo(i32 %v) {
; CHECK: entry:
entry:
; CHECK-NEXT: call void @llvm.bar(metadata !0)
  call void @llvm.bar(metadata !0)

; CHECK-NEXT: ret void, !baz !1
  ret void, !baz !1
}

declare void @llvm.bar(metadata)

@global = global i32 0

; CHECK: !0 = !{!1, !2, ptr @global, null}
; CHECK: !1 = !{!2, null}
; CHECK: !2 = !{}
!0 = metadata !{metadata !1, metadata !2, i32* @global, null}
!1 = metadata !{metadata !2, null}
!2 = metadata !{}
