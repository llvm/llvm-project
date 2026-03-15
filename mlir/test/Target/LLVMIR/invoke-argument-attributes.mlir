// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @test
llvm.func @test(%arg0: i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.signext}) attributes {personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.zero : !llvm.ptr
  %1 = llvm.mlir.constant(0 : i16) : i16
  // CHECK:      invoke signext i16 @somefunc(i16 noundef signext %{{.*}})
  // CHECK-NEXT:   to label %{{.*}} unwind label %{{.*}}
  %2 = llvm.invoke @somefunc(%arg0) to ^bb2 unwind ^bb1 : (i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.signext})
^bb1:  // pred: ^bb0
  %3 = llvm.landingpad (catch %0 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
  %4 = llvm.extractvalue %3[0] : !llvm.struct<(ptr, i32)>
  %5 = llvm.call tail @__cxa_begin_catch(%4) : (!llvm.ptr) -> !llvm.ptr
  llvm.call tail @__cxa_end_catch() : () -> ()
  llvm.br ^bb3(%1 : i16)
^bb2:  // pred: ^bb0
  llvm.br ^bb3(%2 : i16)
^bb3(%6: i16):  // 2 preds: ^bb1, ^bb2
  llvm.return %6 : i16
}

llvm.func @somefunc(i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.noundef, llvm.signext})
llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @__cxa_begin_catch(!llvm.ptr) -> !llvm.ptr
llvm.func @__cxa_end_catch()
