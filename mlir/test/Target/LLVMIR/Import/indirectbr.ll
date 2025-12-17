; RUN: mlir-translate --import-llvm %s -split-input-file | FileCheck %s

; CHECK: llvm.func @basic(%[[arg0:.*]]: !llvm.ptr)
define i32 @basic(ptr %dst) {
  ; CHECK:   llvm.indirectbr %[[arg0]] : !llvm.ptr, [
  ; CHECK:   ^[[bb1:.*]],
  ; CHECK:   ^[[bb2:.*]]
  ; CHECK:   ]
  indirectbr ptr %dst, [label %bb1, label %bb2]
bb1:
  ; CHECK: ^[[bb1]]:
  ; CHECK:   llvm.return
  ret i32 0
bb2:
  ; CHECK: ^[[bb2]]:
  ; CHECK:   llvm.return
  ret i32 1
}

; // -----

; CHECK: llvm.mlir.global external @addr()
@addr = global ptr null

; CHECK-LABEL:  llvm.func @test_indirectbr() {
define void @test_indirectbr() {
  ; CHECK:    %[[BA:.*]] = llvm.blockaddress <function = @test_indirectbr, tag = <id = 1>> : !llvm.ptr
  ; CHECK:    {{.*}} = llvm.mlir.addressof @addr : !llvm.ptr
  ; CHECK:    llvm.store %[[BA]], {{.*}} : !llvm.ptr, !llvm.ptr
  store ptr blockaddress(@test_indirectbr, %block), ptr @addr
  ; CHECK:    %[[TARGET_ADDR:.*]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.ptr
  %val = load ptr, ptr @addr
  ; CHECK:    llvm.indirectbr %[[TARGET_ADDR]] : !llvm.ptr, [
  ; CHECK:    ^[[TARGET_BB:.*]]
  ; CHECK:    ]
  indirectbr ptr %val, [label %block]
  ; CHECK:  ^[[TARGET_BB]]:
  ; CHECK:    llvm.blocktag <id = 1>
  ; CHECK:    llvm.return
  ; CHECK:  }
block:
  ret void
}

; // -----

; CHECK: llvm.func @callee(!llvm.ptr, i32, i32) -> i32
declare i32 @callee(ptr %a, i32 %v, i32 %p)

; CHECK: llvm.func @test_indirectbr_phi(
; CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> i32 {
define i32 @test_indirectbr_phi(ptr %address, ptr %a, ptr %b, i32 %v) {
entry:
  ; CHECK:   %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK:   %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
  %dest = select i1 poison, ptr blockaddress(@test_indirectbr_phi, %end), ptr %address
  ; CHECK:   llvm.indirectbr {{.*}} : !llvm.ptr, [
  ; CHECK:   ^[[HEAD_BB:.*]],
  ; CHECK:   ^[[TAIL_BB:.*]](%[[ONE]] : i32)
  ; CHECK:   ]
  indirectbr ptr %dest, [label %head, label %tail]

head:
  ; CHECK: ^[[HEAD_BB]]:
  ; CHECK:   llvm.indirectbr {{.*}} : !llvm.ptr, [
  ; CHECK:   ^[[TAIL_BB]](%[[TWO]] : i32),
  ; CHECK:   ^[[END_BB:.*]]
  ; CHECK:   ]
  %dest2 = select i1 poison, ptr blockaddress(@test_indirectbr_phi, %end), ptr %address
  indirectbr ptr %dest2, [label %tail, label %end]

tail:
  ; CHECK: ^[[TAIL_BB]](%[[BLOCK_ARG:.*]]: i32):
  ; CHECK:   {{.*}} = llvm.call @callee({{.*}}, %[[BLOCK_ARG]])
  ; CHECK:   llvm.return
  %p = phi i32 [1, %entry], [2, %head]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

end:
  ; CHECK: ^[[END_BB]]:
  ; CHECK:   llvm.blocktag
  ; CHECK:   llvm.return
  ; CHECK: }
  ret i32 %v
}
