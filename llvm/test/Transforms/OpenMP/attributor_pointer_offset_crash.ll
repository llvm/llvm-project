; RUN: opt < %s -S -passes=openmp-opt

; Verify the address space cast doesn't cause a crash

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"

%"struct.(anonymous namespace)::TeamStateTy" = type { %"struct.(anonymous namespace)::ICVStateTy", i32, i8* }
%"struct.(anonymous namespace)::ICVStateTy" = type { i32, i32, i32, i32, i32, i32 }
%"struct.(anonymous namespace)::ThreadStateTy" = type { %"struct.(anonymous namespace)::ICVStateTy", %"struct.(anonymous namespace)::ThreadStateTy"* }

@_ZN12_GLOBAL__N_19TeamStateE = internal addrspace(3) global %"struct.(anonymous namespace)::TeamStateTy" undef, align 8

define internal i8* @__kmpc_alloc_shared() {
lor.lhs.false.i.i:
  br i1 false, label %_ZN4_OMP5state8lookup32ENS0_9ValueKindEb.exit.i, label %if.then.i44.i.i

if.then.i44.i.i:                                  ; preds = %lor.lhs.false.i.i
  br label %_ZN4_OMP5state8lookup32ENS0_9ValueKindEb.exit.i

_ZN4_OMP5state8lookup32ENS0_9ValueKindEb.exit.i:  ; preds = %if.then.i44.i.i, %lor.lhs.false.i.i
  %.pn.i45.i.i = phi i8* [ null, %if.then.i44.i.i ], [ addrspacecast (i8 addrspace(3)* bitcast (%"struct.(anonymous namespace)::TeamStateTy" addrspace(3)* @_ZN12_GLOBAL__N_19TeamStateE to i8 addrspace(3)*) to i8*), %lor.lhs.false.i.i ]
  %retval.0.in.i.i.i = getelementptr inbounds i8, i8* %.pn.i45.i.i, i64 4
  %retval.0.i46.i.i = bitcast i8* %retval.0.in.i.i.i to i32*
  %0 = load i32, i32* %retval.0.i46.i.i, align 4
  ret i8* null
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"openmp-device", i32 50}
