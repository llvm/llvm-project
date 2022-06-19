; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that llvm.experimental.noalias.scope.decl is treated as not reading or writing memory.

declare void @llvm.experimental.noalias.scope.decl(metadata)

define i32 @foo(i32* %a, i32* %b, i1 %c) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, i32* %a, align 4
; CHECK-NOT: MemoryDef
; CHECK: call void @llvm.experimental.noalias.scope.decl
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
; CHECK: MemoryUse(1)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* %a, align 4
  ret i32 %1
}


!0 = !{ !1 }
!1 = distinct !{ !1, !2, !"foo: var" }
!2 = distinct !{ !2, !"foo" }
