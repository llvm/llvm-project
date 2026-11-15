// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @personalityFn(...) -> i32
llvm.func @foo()

// CHECK-LABEL: define void @test_cleanup()
// CHECK-SAME: personality ptr @personalityFn
llvm.func @test_cleanup() attributes { personality = @personalityFn } {
  %none = llvm.mlir.none : token
  llvm.invoke @foo() to ^normal unwind ^cleanup [] : () -> ()
^normal:
  llvm.return
^cleanup:
  // CHECK: %[[CP:.*]] = cleanuppad within none []
  %cp = llvm.cleanuppad within %none
  // CHECK: call void @foo() [ "funclet"(token %[[CP]]) ]
  llvm.call @foo() ["funclet"(%cp : token)] : () -> ()
  // CHECK: cleanupret from %[[CP]] unwind to caller
  llvm.cleanupret %cp
}

// CHECK-LABEL: define void @test_cleanup_args()
// CHECK-SAME: personality ptr @personalityFn
llvm.func @test_cleanup_args() attributes { personality = @personalityFn } {
  %none = llvm.mlir.none : token
  %arg = llvm.mlir.constant(0 : i32) : i32
  llvm.invoke @foo() to ^normal unwind ^cleanup [] : () -> ()
^normal:
  llvm.return
^cleanup:
  // CHECK: %[[CP:.*]] = cleanuppad within none [i32 0]
  %cp = llvm.cleanuppad within %none [%arg : i32]
  // CHECK: cleanupret from %[[CP]] unwind to caller
  llvm.cleanupret %cp
}

// CHECK-LABEL: define void @test_cleanup_unwind()
// CHECK-SAME: personality ptr @personalityFn
llvm.func @test_cleanup_unwind() attributes { personality = @personalityFn } {
  %none = llvm.mlir.none : token
  llvm.invoke @foo() to ^normal unwind ^cleanup [] : () -> ()
^normal:
  llvm.return
^cleanup:
  // CHECK: %[[CP:.*]] = cleanuppad within none []
  %cp = llvm.cleanuppad within %none
  // CHECK: cleanupret from %[[CP]] unwind label %[[UNWIND_BB:.*]]
  llvm.cleanupret %cp unwind ^unwind_bb
^unwind_bb:
  // CHECK: [[UNWIND_BB]]:
  // CHECK: %[[CP2:.*]] = cleanuppad within none []
  %cp2 = llvm.cleanuppad within %none
  // CHECK: cleanupret from %[[CP2]] unwind to caller
  llvm.cleanupret %cp2
}

// CHECK-LABEL: define void @test_catch()
// CHECK-SAME: personality ptr @personalityFn
llvm.func @test_catch() attributes { personality = @personalityFn } {
  %none = llvm.mlir.none : token
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.invoke @foo() to ^normal unwind ^dispatch [] : () -> ()
^normal:
  llvm.return
^dispatch:
  // CHECK: %[[CS:.*]] = catchswitch within none [label %[[BODY:.*]]] unwind to caller
  %cs = llvm.catchswitch within %none [^catch_body] unwind to caller
^catch_body:
  // CHECK: [[BODY]]:
  // CHECK: %[[PAD:.*]] = catchpad within %[[CS]] [i32 0]
  %pad = llvm.catchpad within %cs [%c0 : i32]
  // CHECK: call void @foo() [ "funclet"(token %[[PAD]]) ]
  llvm.call @foo() ["funclet"(%pad : token)] : () -> ()
  // CHECK: catchret from %[[PAD]] to label %[[EXIT:.*]]
  llvm.catchret from %pad to ^exit
^exit:
  // CHECK: [[EXIT]]:
  llvm.return
}

// CHECK-LABEL: define void @test_catch_unwind()
// CHECK-SAME: personality ptr @personalityFn
llvm.func @test_catch_unwind() attributes { personality = @personalityFn } {
  %none = llvm.mlir.none : token
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.invoke @foo() to ^normal unwind ^dispatch [] : () -> ()
^normal:
  llvm.return
^dispatch:
  // CHECK: %[[CS:.*]] = catchswitch within none [label %[[BODY:.*]]] unwind label %[[UNWIND_BB:.*]]
  %cs = llvm.catchswitch within %none [^catch_body] unwind ^unwind_bb
^catch_body:
  // CHECK: [[BODY]]:
  // CHECK: %[[PAD:.*]] = catchpad within %[[CS]] [i32 0]
  %pad = llvm.catchpad within %cs [%c0 : i32]
  // CHECK: catchret from %[[PAD]] to label %[[EXIT:.*]]
  llvm.catchret from %pad to ^exit
^unwind_bb:
  // CHECK: [[UNWIND_BB]]:
  // CHECK: %[[CP:.*]] = cleanuppad within none []
  %cp = llvm.cleanuppad within %none
  // CHECK: cleanupret from %[[CP]] unwind to caller
  llvm.cleanupret %cp
^exit:
  // CHECK: [[EXIT]]:
  llvm.return
}
