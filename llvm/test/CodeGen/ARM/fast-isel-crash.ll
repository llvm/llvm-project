; RUN: llc < %s -O0 -mtriple=thumbv7-apple-darwin

%union.anon = type { <16 x i32> }

@__md0 = external global [137 x i8]

define internal void @stretch(<4 x i8> addrspace(1)* %src, <4 x i8> addrspace(1)* %dst, i32 %width, i32 %height, i32 %iLS, i32 %oLS, <2 x float> %c, <4 x float> %param) nounwind {
entry:
  ret void
}

define internal i32 @_Z13get_global_idj(i32 %dim) nounwind ssp {
entry:
  ret i32 undef
}

define void @wrap(i8 addrspace(1)* addrspace(1)* %arglist, i32 addrspace(1)* %gtid) nounwind ssp {
entry:
  call void @stretch(<4 x i8> addrspace(1)* undef, <4 x i8> addrspace(1)* undef, i32 undef, i32 undef, i32 undef, i32 undef, <2 x float> undef, <4 x float> undef)
  ret void
}
