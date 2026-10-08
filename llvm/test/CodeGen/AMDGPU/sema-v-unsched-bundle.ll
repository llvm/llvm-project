; REQUIRES: asserts
; RUN: llc -mtriple=amdgpu9.0a -O1 -debug-only=machine-scheduler -filetype=null < %s 2>&1 | FileCheck --check-prefix=DEBUG %s

; DEBUG: Reverting scheduling for region 0

; Three vectors are kept live across the GWS bundle so that the region stays
; well above the occupancy limit and scheduling is reverted, which is what
; exercises the unscheduling of bundled instructions.

@G = global <32 x i8> splat (i8 1)
@G.1 = global <32 x i8> splat (i8 127)
@G.2 = global <32 x i8> splat (i8 63)

define amdgpu_kernel void @gws_sema_v_offset0(i32 %val, <32 x i1>* %inp, <32 x i1>* %inp2) {
  %LGV1 = load <32 x i8>, ptr @G.1, align 32
  %LGV = load <32 x i8>, ptr @G, align 32
  %LGV2 = load <32 x i8>, ptr @G.2, align 32
  call void @llvm.amdgcn.ds.gws.sema.v(i32 0)
  %C = icmp ne <32 x i8> %LGV, %LGV1
  %C2 = icmp ne <32 x i8> %LGV1, %LGV2
  store <32 x i1> %C, ptr %inp, align 4
  store <32 x i1> %C2, ptr %inp2, align 4
  ret void
}

declare void @llvm.amdgcn.ds.gws.sema.v(i32)
