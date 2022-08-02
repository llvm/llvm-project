;; __kernel void test_64(__global int* res)
;; {
;;     long tid = get_global_id(0);
;;
;;     switch(tid)
;;     {
;;     case 0:
;;         res[tid] = 1;
;;         break;
;;     case 1:
;;         res[tid] = 2;
;;         break;
;;     case 21474836481:
;;         res[tid] = 3;
;;         break;
;;     }
;; }
;; bash$ clang -cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -include opencl.h -emit-llvm OpSwitch.cl -o test_64.ll

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpSwitch %[[#]] %[[#]] 0 0 %[[#]] 1 0 %[[#]] 1 5 %[[#]]

define spir_kernel void @test_64(i32 addrspace(1)* %res) {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 8
  %tid = alloca i64, align 8
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  store i64 %call, i64* %tid, align 8
  %0 = load i64, i64* %tid, align 8
  switch i64 %0, label %sw.epilog [
    i64 0, label %sw.bb
    i64 1, label %sw.bb1
    i64 21474836481, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  %1 = load i64, i64* %tid, align 8
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %2, i64 %1
  store i32 1, i32 addrspace(1)* %arrayidx, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %3 = load i64, i64* %tid, align 8
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %3
  store i32 2, i32 addrspace(1)* %arrayidx2, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %5 = load i64, i64* %tid, align 8
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %6, i64 %5
  store i32 3, i32 addrspace(1)* %arrayidx4, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb3, %sw.bb1, %sw.bb
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
