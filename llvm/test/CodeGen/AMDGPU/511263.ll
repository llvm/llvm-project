; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -O1 < %s
source_filename = "i1-copy-from-loop.ll"

@G = global ptr addrspace(8) poison

define amdgpu_ps void @i1_copy_from_loop(ptr addrspace(8) inreg %rsrc, i32 %tid) {
entry:
  br label %for.body

for.body:                                         ; preds = %end.loop, %entry
  %i = phi i32 [ 0, %entry ], [ %i.inc, %end.loop ]
  %LGV = load ptr addrspace(8), ptr @G, align 8
  %cc = icmp ult i32 %i, 4
  call void @i1_copy_from_loop(ptr addrspace(8) %LGV, i32 -2147483648)
  br i1 %cc, label %mid.loop, label %for.end

mid.loop:                                         ; preds = %for.body
  %v = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 %tid, i32 %i, i32 0, i32 0)
  %cc2 = fcmp oge float %v, 0.000000e+00
  br i1 %cc2, label %end.loop, label %for.end

end.loop:                                         ; preds = %mid.loop
  %i.inc = add i32 %i, 1
  br label %for.body

for.end:                                          ; preds = %mid.loop, %for.body
  br i1 %cc, label %if, label %end

if:                                               ; preds = %for.end
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float undef, float undef, float undef, float undef, i1 true, i1 true)
  br label %end

end:                                              ; preds = %if, %for.end
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) nocapture readonly, i32, i32, i32, i32 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
