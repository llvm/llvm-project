// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Use different types to detect operands placed in the wrong segment.
llvm.func @target(i32)
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: llvm.func @invoke_segments(
// CHECK-SAME: %[[ARG:[^:]+]]: i32, %[[NORMAL:[^:]+]]: i16, %[[UNWIND:[^:]+]]: i64, %[[TAG:[^:]+]]: i8
// CHECK: llvm.invoke @target(%[[ARG]]) to ^{{[^(]+}}(%[[NORMAL]] : i16) unwind ^{{[^(]+}}(%[[UNWIND]] : i64) ["tag"(%[[TAG]] : i8)] : (i32) -> ()
llvm.func @invoke_segments(%arg: i32, %normal: i16, %unwind: i64, %tag: i8) attributes {personality = @__gxx_personality_v0} {
  llvm.invoke @target(%arg) to ^normal(%normal : i16) unwind ^unwind(%unwind : i64) ["tag"(%tag : i8)] : (i32) -> ()
^normal(%n: i16):
  llvm.return
^unwind(%u: i64):
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
}
