; RUN: mlir-translate -import-llvm %s | FileCheck %s

; Test unreachable blocks are dropped.

; CHECK-LABEL: llvm.func @unreachable_block
define void @unreachable_block(float %0) {
.entry:
  ; CHECK: llvm.return
  ret void

unreachable:
  ; CHECK-NOT: llvm.fadd
  %1 = fadd float %0, %1
  br label %unreachable
}

; Test unreachable blocks with back edges are supported.

; CHECK-LABEL: llvm.func @back_edge
define i32 @back_edge(i32 %0) {
.entry:
  ; CHECK: llvm.br ^[[RET:.*]](%{{.*}})
  br label %ret
ret:
  ; CHECK: ^[[RET]](%{{.*}}: i32)
  %1 = phi i32 [ %0, %.entry ], [ %2, %unreachable ]
  ; CHECK: llvm.return %{{.*}} : i32
  ret i32 %1

unreachable:
  ; CHECK-NOT: add
  %2 = add i32 %0, %2
  %3 = icmp eq i32 %2, 42
  br i1 %3, label %ret, label %unreachable
}
