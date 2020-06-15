target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* nocapture, i32, i32, i32, i1) #1

; Function Attrs: alwaysinline nounwind
define i32 @__amdgcn_atomic_inc_i32(i32* %x, i32 %v) #0 {
entry:

  %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* %x, i32 %v,
  i32 5, ; Ordering. AtomicOrdering.h: sequentially consistent
  i32 2, ; Scope. SyncScope.h:  OpenCLAllSVMDevices is 2
  i1 1 ; Volatile.  True for consistency with other atomic operations
  )
  ret i32 %ret
}

attributes #0 = { alwaysinline nounwind }
attributes #1 = { nounwind argmemonly }
