; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: OpSConvert

define spir_kernel void @math_kernel8(<8 x i32> addrspace(1)* nocapture %out, <8 x float> addrspace(1)* nocapture readonly %in1, <8 x float> addrspace(1)* nocapture readonly %in2) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %sext = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext, 32
  %arrayidx = getelementptr inbounds <8 x float>, <8 x float> addrspace(1)* %in1, i64 %idxprom
  %0 = load <8 x float>, <8 x float> addrspace(1)* %arrayidx, align 32
  %arrayidx2 = getelementptr inbounds <8 x float>, <8 x float> addrspace(1)* %in2, i64 %idxprom
  %1 = load <8 x float>, <8 x float> addrspace(1)* %arrayidx2, align 32
  %call3 = tail call spir_func <8 x i32> @_Z7isequalDv8_fDv8_f(<8 x float> %0, <8 x float> %1)
  %arrayidx5 = getelementptr inbounds <8 x i32>, <8 x i32> addrspace(1)* %out, i64 %idxprom
  store <8 x i32> %call3, <8 x i32> addrspace(1)* %arrayidx5, align 32
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func <8 x i32> @_Z7isequalDv8_fDv8_f(<8 x float>, <8 x float>)
