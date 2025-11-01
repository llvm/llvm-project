// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @somefunc(i32, !llvm.ptr)

// CHECK-LABEL: define void @test_call_arg_attrs_direct
llvm.func @test_call_arg_attrs_direct(%arg0: i32, %arg1: !llvm.ptr) {
  // CHECK: call void @somefunc(i32 %{{.*}}, ptr byval(i64) %{{.*}})
  llvm.call @somefunc(%arg0, %arg1) : (i32, !llvm.ptr {llvm.byval = i64}) -> ()
  llvm.return
}

// CHECK-LABEL: define i16 @test_call_arg_attrs_indirect
llvm.func @test_call_arg_attrs_indirect(%arg0: i16, %arg1: !llvm.ptr) -> i16 {
  // CHECK: tail call signext i16 %{{.*}}(i16 noundef signext %{{.*}})
  %0 = llvm.call tail %arg1(%arg0) : !llvm.ptr, (i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.signext})
  llvm.return %0 : i16
}
