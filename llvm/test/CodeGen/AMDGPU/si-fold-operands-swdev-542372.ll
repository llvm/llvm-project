; RUN: not --crash llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a %s -o /dev/null

@lds = internal addrspace(3) global [5 x i32] poison

define amdgpu_kernel void @kernel() {
entry:
  %load.lds.0 = load <2 x i32>, ptr addrspace(3) @lds
  %vecext.i55 = extractelement <2 x i32> %load.lds.0, i64 0
  %cmp3.i57 = icmp eq i32 %vecext.i55, 2
  store i32 0, ptr addrspace(3) @lds
  br i1 %cmp3.i57, label %land.rhs49, label %land.end59

land.rhs49:                                       ; preds = %entry
  %load.lds.1 = load <2 x i32>, ptr addrspace(3) @lds
  %vecext.i67 = extractelement <2 x i32> %load.lds.1, i64 0
  %cmp3.i69 = icmp eq i32 %vecext.i67, 1
  br i1 %cmp3.i69, label %land.rhs57, label %land.end59

land.rhs57:                                       ; preds = %land.rhs49
  %rem.i.i.i = srem <2 x i32> %load.lds.0, %load.lds.1
  %ref.tmp.sroa.0.0.vec.extract.i.i = extractelement <2 x i32> %rem.i.i.i, i64 0
  store i32 %ref.tmp.sroa.0.0.vec.extract.i.i, ptr addrspace(3) @lds
  store i32 %ref.tmp.sroa.0.0.vec.extract.i.i, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @lds, i32 4)
  %load.lds.2 = load <2 x i32>, ptr addrspace(3) @lds
  %vecext.i.i.i = extractelement <2 x i32> %load.lds.2, i64 0
  %cmp3.i.i.i = icmp ne i32 %vecext.i.i.i, 0
  %vecext.1.i.i.i = extractelement <2 x i32> %load.lds.2, i64 1
  %cmp3.1.i.i.i = icmp ne i32 %vecext.1.i.i.i, 0
  %.not.i.i = select i1 %cmp3.i.i.i, i1 true, i1 %cmp3.1.i.i.i
  br i1 %.not.i.i, label %land.end59, label %if.end.i

if.end.i:                                         ; preds = %land.rhs57
  %and.i.i.i = and <2 x i32> %load.lds.2, splat (i32 1)
  %ref.tmp.sroa.0.0.vec.extract.i20.i = extractelement <2 x i32> %and.i.i.i, i64 0
  br label %land.end59

land.end59:                                       ; preds = %if.end.i, %land.rhs57, %land.rhs49, %entry
  ret void
}
