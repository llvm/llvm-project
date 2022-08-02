;; __kernel void test_32(__global int* res)
;; {
;;     int tid = get_global_id(0);
;;
;;     switch(tid)
;;     {
;;     case 0:
;;         res[tid] = 1;
;;         break;
;;     case 1:
;;         res[tid] = 2;
;;         break;
;;     }
;; }
;; bash$ clang -cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -include opencl.h -emit-llvm OpSwitch.cl -o test_32.ll

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpSwitch %[[#]] %[[#]] 0 %[[#]] 1 %[[#]]

define spir_kernel void @test_32(i32 addrspace(1)* %res) {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 8
  %tid = alloca i32, align 4
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %conv = trunc i64 %call to i32
  store i32 %conv, i32* %tid, align 4
  %0 = load i32, i32* %tid, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  %1 = load i32, i32* %tid, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %2, i64 %idxprom
  store i32 1, i32 addrspace(1)* %arrayidx, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %3 = load i32, i32* %tid, align 4
  %idxprom2 = sext i32 %3 to i64
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %idxprom2
  store i32 2, i32 addrspace(1)* %arrayidx3, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb1, %sw.bb
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
