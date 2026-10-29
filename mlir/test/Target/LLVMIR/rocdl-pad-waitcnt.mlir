// RUN: mlir-translate -mlir-to-llvmir %s -o %t.ll
// RUN: FileCheck %s --check-prefix=IR < %t.ll

// IR-LABEL: @rocdl_hint_kernel(
// IR: [[A:%.*]] = load i32
// IR: br i1 {{.*}}, label %[[LOAD:[^, ]+]], label %[[JOIN:[^, ]+]]
// IR: [[LOAD]]:
// IR: [[B:%.*]] = load i32
// IR-NEXT: call void @llvm.amdgcn.pad.waitcnt(i32 0, i32 1)
// IR-NEXT: br label %[[JOIN]]
// IR: [[JOIN]]:
// IR-NEXT: [[V:%.*]] = phi i32 [ [[B]], %[[LOAD]] ], [ 0, %{{.*}} ]
// IR: store i32 [[A]]
// IR: store i32 [[V]]

module attributes {llvm.target_triple = "amdgpu9.50-amd-amdhsa"} {
  llvm.func @rocdl_hint_kernel(%data: !llvm.ptr<1>, %out: !llvm.ptr<1>,
                             %condition: i32) attributes {rocdl.kernel} {
    %index = rocdl.workitem.id.x : i32
    %wide = llvm.zext %index : i32 to i64
    %a_ptr = llvm.getelementptr %data[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %a = llvm.load %a_ptr : !llvm.ptr<1> -> i32
    %offset = llvm.mlir.constant(64 : i64) : i64
    %b_index = llvm.add %wide, %offset : i64
    %zero = llvm.mlir.constant(0 : i32) : i32
    %cond = llvm.icmp "ne" %condition, %zero : i32
    llvm.cond_br %cond, ^load, ^join(%zero : i32)
  ^load:
    %b_ptr = llvm.getelementptr %data[%b_index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %b = llvm.load %b_ptr : !llvm.ptr<1> -> i32
    rocdl.pad.waitcnt 0, 1
    llvm.br ^join(%b : i32)
  ^join(%v: i32):
    %a_out = llvm.getelementptr %out[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %a, %a_out : i32, !llvm.ptr<1>
    %v_out = llvm.getelementptr %out[%b_index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %v, %v_out : i32, !llvm.ptr<1>
    llvm.return
  }
}
