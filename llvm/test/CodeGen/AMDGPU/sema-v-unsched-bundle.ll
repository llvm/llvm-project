; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn -O1 -mcpu=gfx90a -debug-only=machine-scheduler -filetype=null < %s 2>&1 | FileCheck --check-prefix=DEBUG %s

; DEBUG: Attempting to revert scheduling.

@G = global <32 x i8> splat (i8 1)
@G.1 = global <32 x i8> splat (i8 127)

define amdgpu_kernel void @gws_sema_v_offset0(i32 %val, <32 x i1>* %inp) {
  %LGV1 = load <32 x i8>, ptr @G.1, align 32
  %LGV = load <32 x i8>, ptr @G, align 32
  call void @llvm.amdgcn.ds.gws.sema.v(i32 0)
  %C = icmp ne <32 x i8> %LGV, %LGV1
  store <32 x i1> %C, ptr %inp, align 4
  ret void
}

declare void @llvm.amdgcn.ds.gws.sema.v(i32)
