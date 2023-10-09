// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @foo()

// CHECK-LABEL: @omp_teams_simple
// CHECK: call void {{.*}} @__kmpc_fork_teams(ptr @{{.+}}, i32 0, ptr [[OUTLINED_FN:.+]])
// CHECK: ret void
llvm.func @omp_teams_simple() {
    omp.teams {
        llvm.call @foo() : () -> ()
        omp.terminator
    }
    llvm.return
}

// CHECK: define internal void @[[OUTLINED_FN:.+]](ptr {{.+}}, ptr {{.+}})
// CHECK:   call void @foo()
// CHECK:   ret void

// -----

llvm.func @foo(i32) -> ()

// CHECK-LABEL: @omp_teams_shared_simple
// CHECK-SAME: (i32 [[ARG0:%.+]])
// CHECK: [[STRUCT_ARG:%.+]] = alloca { i32 }
// CHECK: br
// CHECK: [[GEP:%.+]] = getelementptr { i32 }, ptr [[STRUCT_ARG]], i32 0, i32 0
// CHECK: store i32 [[ARG0]], ptr [[GEP]]
// CHECK: call void {{.+}} @__kmpc_fork_teams(ptr @{{.+}}, i32 1, ptr [[OUTLINED_FN:.+]], ptr [[STRUCT_ARG]])
// CHECK: ret void
llvm.func @omp_teams_shared_simple(%arg0: i32) {
    omp.teams {
        llvm.call @foo(%arg0) : (i32) -> ()
        omp.terminator
    }
    llvm.return
}

// CHECK: define internal void [[OUTLINED_FN:@.+]](ptr {{.+}}, ptr {{.+}}, ptr [[STRUCT_ARG:%.+]])
// CHECK:   [[GEP:%.+]] = getelementptr { i32 }, ptr [[STRUCT_ARG]], i32 0, i32 0
// CHECK:   [[LOAD_GEP:%.+]] = load i32, ptr [[GEP]]
// CHECK:   call void @foo(i32 [[LOAD_GEP]])
// CHECK:   ret void

// -----

llvm.func @my_alloca_fn() -> !llvm.ptr<i32>
llvm.func @foo(i32, f32, !llvm.ptr<i32>, f128, !llvm.ptr<i32>, i32) -> ()
llvm.func @bar()

// CHECK-LABEL: @omp_teams_branching_shared
// CHECK-SAME: (i1 [[CONDITION:%.+]], i32 [[ARG0:%.+]], float [[ARG1:%.+]], ptr [[ARG2:%.+]], fp128 [[ARG3:%.+]])

// Checking that the allocation for struct argument happens in the alloca block.
// CHECK: [[STRUCT_ARG:%.+]] = alloca { i1, i32, float, ptr, fp128, ptr, i32 }
// CHECK: [[ALLOCATED:%.+]] = call ptr @my_alloca_fn()
// CHECK: [[LOADED:%.+]] = load i32, ptr [[ALLOCATED]]
// CHECK: br label

// Checking that the shared values are stored properly in the struct arg.
// CHECK: [[CONDITION_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]]
// CHECK: store i1 [[CONDITION]], ptr [[CONDITION_PTR]]
// CHECK: [[ARG0_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]], i32 0, i32 1
// CHECK: store i32 [[ARG0]], ptr [[ARG0_PTR]]
// CHECK: [[ARG1_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]], i32 0, i32 2
// CHECK: store float [[ARG1]], ptr [[ARG1_PTR]]
// CHECK: [[ARG2_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]], i32 0, i32 3
// CHECK: store ptr [[ARG2]], ptr [[ARG2_PTR]]
// CHECK: [[ARG3_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]], i32 0, i32 4
// CHECK: store fp128 [[ARG3]], ptr [[ARG3_PTR]]
// CHECK: [[ALLOCATED_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]], i32 0, i32 5
// CHECK: store ptr [[ALLOCATED]], ptr [[ALLOCATED_PTR]]
// CHECK: [[LOADED_PTR:%.+]] = getelementptr {{.+}}, ptr [[STRUCT_ARG]], i32 0, i32 6
// CHECK: store i32 [[LOADED]], ptr [[LOADED_PTR]]

// Runtime call.
// CHECK: call void {{.+}} @__kmpc_fork_teams(ptr @{{.+}}, i32 1, ptr [[OUTLINED_FN:@.+]], ptr [[STRUCT_ARG]])
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
// CHECK: define internal void [[OUTLINED_FN:@.+]](ptr {{.+}}, ptr {{.+}}, ptr [[DATA:%.+]])
// CHECK:   [[CONDITION_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]]
// CHECK:   [[CONDITION:%.+]] = load i1, ptr [[CONDITION_PTR]]
// CHECK:   [[ARG0_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]], i32 0, i32 1
// CHECK:   [[ARG0:%.+]] = load i32, ptr [[ARG0_PTR]]
// CHECK:   [[ARG1_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]], i32 0, i32 2
// CHECK:   [[ARG1:%.+]] = load float, ptr [[ARG1_PTR]]
// CHECK:   [[ARG2_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]], i32 0, i32 3
// CHECK:   [[ARG2:%.+]] = load ptr, ptr [[ARG2_PTR]]
// CHECK:   [[ARG3_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]], i32 0, i32 4
// CHECK:   [[ARG3:%.+]] = load fp128, ptr [[ARG3_PTR]]
// CHECK:   [[ALLOCATED_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]], i32 0, i32 5
// CHECK:   [[ALLOCATED:%.+]] = load ptr, ptr [[ALLOCATED_PTR]]
// CHECK:   [[LOADED_PTR:%.+]] = getelementptr {{.+}}, ptr [[DATA]], i32 0, i32 6
// CHECK:   [[LOADED:%.+]] = load i32, ptr [[LOADED_PTR]]
// CHECK:   br label

// CHECK:   br i1 [[CONDITION]], label %[[TRUE:.+]], label %[[FALSE:.+]]
// CHECK: [[FALSE]]:
// CHECK-NEXT: br label
// CHECK: [[TRUE]]:
// CHECK:   call void @foo(i32 [[ARG0]], float [[ARG1]], ptr [[ARG2]], fp128 [[ARG3]], ptr [[ALLOCATED]], i32 [[LOADED]])
// CHECK-NEXT: br label
// CHECK: ret void

