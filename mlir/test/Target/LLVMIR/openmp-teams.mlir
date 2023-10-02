// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @foo()

// CHECK-LABEL: @omp_teams_simple
// CHECK: call void {{.*}} @__kmpc_fork_teams(ptr @{{.+}}, i32 0, ptr [[wrapperfn:.+]])
// CHECK: ret void
llvm.func @omp_teams_simple() {
    omp.teams {
        llvm.call @foo() : () -> ()
        omp.terminator
    }
    llvm.return
}

// CHECK: define internal void @[[outlinedfn:.+]]()
// CHECK:   call void @foo()
// CHECK:   ret void
// CHECK: define void [[wrapperfn]](ptr %[[global_tid:.+]], ptr %[[bound_tid:.+]])
// CHECK:   call void @[[outlinedfn]]
// CHECK:   ret void

// -----

llvm.func @foo(i32) -> ()

// CHECK-LABEL: @omp_teams_shared_simple
// CHECK-SAME: (i32 [[arg0:%.+]])
// CHECK: [[structArg:%.+]] = alloca { i32 }
// CHECK: br
// CHECK: [[gep:%.+]] = getelementptr { i32 }, ptr [[structArg]], i32 0, i32 0
// CHECK: store i32 [[arg0]], ptr [[gep]]
// CHECK: call void {{.+}} @__kmpc_fork_teams(ptr @{{.+}}, i32 1, ptr [[wrapperfn:.+]], ptr [[structArg]])
// CHECK: ret void
llvm.func @omp_teams_shared_simple(%arg0: i32) {
    omp.teams {
        llvm.call @foo(%arg0) : (i32) -> ()
        omp.terminator
    }
    llvm.return
}

// CHECK: define internal void [[outlinedfn:@.+]](ptr [[structArg:%.+]])
// CHECK:   [[gep:%.+]] = getelementptr { i32 }, ptr [[structArg]], i32 0, i32 0
// CHECK:   [[loadgep:%.+]] = load i32, ptr [[gep]]
// CHECK:   call void @foo(i32 [[loadgep]])
// CHECK:   ret void
// CHECK: define void [[wrapperfn]](ptr [[global_tid:.+]], ptr [[bound_tid:.+]], ptr [[structArg:.+]])
// CHECK:   call void [[outlinedfn]](ptr [[structArg]])
// CHECK:   ret void

// -----

llvm.func @my_alloca_fn() -> !llvm.ptr<i32>
llvm.func @foo(i32, f32, !llvm.ptr<i32>, f128, !llvm.ptr<i32>, i32) -> ()
llvm.func @bar()

// CHECK-LABEL: @omp_teams_branching_shared
// CHECK-SAME: (i1 [[condition:%.+]], i32 [[arg0:%.+]], float [[arg1:%.+]], ptr [[arg2:%.+]], fp128 [[arg3:%.+]])

// Checking that the allocation for struct argument happens in the alloca block.
// CHECK: [[structArg:%.+]] = alloca { i1, i32, float, ptr, fp128, ptr, i32 }
// CHECK: [[allocated:%.+]] = call ptr @my_alloca_fn()
// CHECK: [[loaded:%.+]] = load i32, ptr [[allocated]]
// CHECK: br label

// Checking that the shared values are stored properly in the struct arg.
// CHECK: [[conditionPtr:%.+]] = getelementptr {{.+}}, ptr [[structArg]]
// CHECK: store i1 [[condition]], ptr [[conditionPtr]]
// CHECK: [[arg0ptr:%.+]] = getelementptr {{.+}}, ptr [[structArg]], i32 0, i32 1
// CHECK: store i32 [[arg0]], ptr [[arg0ptr]]
// CHECK: [[arg1ptr:%.+]] = getelementptr {{.+}}, ptr [[structArg]], i32 0, i32 2
// CHECK: store float [[arg1]], ptr [[arg1ptr]]
// CHECK: [[arg2ptr:%.+]] = getelementptr {{.+}}, ptr [[structArg]], i32 0, i32 3
// CHECK: store ptr [[arg2]], ptr [[arg2ptr]]
// CHECK: [[arg3ptr:%.+]] = getelementptr {{.+}}, ptr [[structArg]], i32 0, i32 4
// CHECK: store fp128 [[arg3]], ptr [[arg3ptr]]
// CHECK: [[allocatedPtr:%.+]] = getelementptr {{.+}}, ptr [[structArg]], i32 0, i32 5
// CHECK: store ptr [[allocated]], ptr [[allocatedPtr]]
// CHECK: [[loadedPtr:%.+]] = getelementptr {{.+}}, ptr [[structArg]], i32 0, i32 6
// CHECK: store i32 [[loaded]], ptr [[loadedPtr]]

// Runtime call.
// CHECK: call void {{.+}} @__kmpc_fork_teams(ptr @{{.+}}, i32 1, ptr [[wrapperfn:@.+]], ptr [[structArg]])
// CHECK: br label
// CHECK: call void @bar()
// CHECK: ret void
llvm.func @omp_teams_branching_shared(%condition: i1, %arg0: i32, %arg1: f32, %arg2: !llvm.ptr<i32>, %arg3: f128) {
    %allocated = llvm.call @my_alloca_fn(): () -> !llvm.ptr<i32>
    %loaded = llvm.load %allocated : !llvm.ptr<i32>
    llvm.br ^codegenBlock
^codegenBlock:
    omp.teams {
        llvm.cond_br %condition, ^true_block, ^false_block
    ^true_block:
        llvm.call @foo(%arg0, %arg1, %arg2, %arg3, %allocated, %loaded) : (i32, f32, !llvm.ptr<i32>, f128, !llvm.ptr<i32>, i32) -> ()
        llvm.br ^exit
    ^false_block:
        llvm.br ^exit
    ^exit:
        omp.terminator
    }
    llvm.call @bar() : () -> ()
    llvm.return
}

// Check the outlined function.
// CHECK: define internal void [[outlinedfn:@.+]](ptr [[data:%.+]])
// CHECK:   [[conditionPtr:%.+]] = getelementptr {{.+}}, ptr [[data]]
// CHECK:   [[condition:%.+]] = load i1, ptr [[conditionPtr]]
// CHECK:   [[arg0ptr:%.+]] = getelementptr {{.+}}, ptr [[data]], i32 0, i32 1
// CHECK:   [[arg0:%.+]] = load i32, ptr [[arg0ptr]]
// CHECK:   [[arg1ptr:%.+]] = getelementptr {{.+}}, ptr [[data]], i32 0, i32 2
// CHECK:   [[arg1:%.+]] = load float, ptr [[arg1ptr]]
// CHECK:   [[arg2ptr:%.+]] = getelementptr {{.+}}, ptr [[data]], i32 0, i32 3
// CHECK:   [[arg2:%.+]] = load ptr, ptr [[arg2ptr]]
// CHECK:   [[arg3ptr:%.+]] = getelementptr {{.+}}, ptr [[data]], i32 0, i32 4
// CHECK:   [[arg3:%.+]] = load fp128, ptr [[arg3ptr]]
// CHECK:   [[allocatedPtr:%.+]] = getelementptr {{.+}}, ptr [[data]], i32 0, i32 5
// CHECK:   [[allocated:%.+]] = load ptr, ptr [[allocatedPtr]]
// CHECK:   [[loadedPtr:%.+]] = getelementptr {{.+}}, ptr [[data]], i32 0, i32 6
// CHECK:   [[loaded:%.+]] = load i32, ptr [[loadedPtr]]
// CHECK:   br label

// CHECK:   br i1 [[condition]], label %[[true:.+]], label %[[false:.+]]
// CHECK: [[false]]:
// CHECK-NEXT: br label
// CHECK: [[true]]:
// CHECK:   call void @foo(i32 [[arg0]], float [[arg1]], ptr [[arg2]], fp128 [[arg3]], ptr [[allocated]], i32 [[loaded]])
// CHECK-NEXT: br label
// CHECK: ret void

// Check the wrapper function
// CHECK: define void [[wrapperfn]](ptr [[globalTID:%.+]], ptr [[boundTID:%.+]], ptr [[data:%.+]])
// CHECK:   call void [[outlinedfn]](ptr [[data]])
// CHECK:   ret void
