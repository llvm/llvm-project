// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s
 
module {
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("WG size of kernel = %d X %d\0A\00")

  llvm.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    omp.parallel   {
      %0 = llvm.mlir.constant(1 : index) : i64
      %1 = llvm.mlir.constant(10 : index) : i64
      %2 = llvm.mlir.constant(0 : index) : i64
      %4 = llvm.mlir.constant(0 : i32) : i32
      %12 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
      omp.wsloop (%arg2) : i64 = (%2) to (%1) step (%0)  {
        omp.parallel   {
          omp.wsloop (%arg3) : i64 = (%2) to (%0) step (%0)  {
            llvm.store %2, %12 : !llvm.ptr<i64>
            omp.yield
          }
          omp.terminator
        }
        %19 = llvm.load %12 : !llvm.ptr<i64>
        %20 = llvm.trunc %19 : i64 to i32
        %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<29 x i8>>
        %6 = llvm.getelementptr %5[%4, %4] : (!llvm.ptr<array<29 x i8>>, i32, i32) -> !llvm.ptr<i8>
        %21 = llvm.call @printf(%6, %20, %20) : (!llvm.ptr<i8>, i32, i32) -> i32
        omp.yield
      }
      omp.terminator
    }
    %a4 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %a4 : i32
  }

}

// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @1, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @[[inner1:.+]] to void (i32*, i32*, ...)*))

// CHECK: define internal void @[[inner1]]
// CHECK: %[[structArg:.+]] = alloca { i64* }
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @3, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, { i64* }*)* @[[inner2:.+]] to void (i32*, i32*, ...)*), { i64* }* %[[structArg]])
