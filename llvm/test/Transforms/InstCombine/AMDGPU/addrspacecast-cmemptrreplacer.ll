; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=instcombine %s | FileCheck %s

; Variant of select with addrspacecast in one branch on path to root alloca (ic opt is then disabled)
define void @addrspacecast_true_path(ptr addrspace(4) align 8 byref([2 x i8]) %arg) {
; CHECK-LABEL: define void @addrspacecast_true_path(ptr addrspace(4) byref([2 x i8]) align 8 %arg) {
; CHECK-NEXT:  %coerce = alloca [2 x i8], align 8, addrspace(5)
; CHECK-NEXT:  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 8 dereferenceable(16) %coerce, ptr addrspace(4) noundef align 8 dereferenceable(16) %arg, i64 16, i1 false)
; CHECK-NEXT:  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
; CHECK-NEXT:  %cmp.i = icmp slt i32 %load.coerce, 10
; CHECK-NEXT:  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 5
; CHECK-NEXT:  %ret.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr
; CHECK-NEXT:  %out_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 6
; CHECK-NEXT:  %ret.0 = load ptr, ptr addrspace(5) %out_of_line_values.i, align 8
; CHECK-NEXT:  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(i64 8, ptr %retval.0.i)
; CHECK-NEXT:  ret void
; CHECK-NEXT:}
  %coerce = alloca [2 x i8], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %coerce, ptr addrspace(4) align 8 %arg, i64 16, i1 false)
  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
  %cmp.i = icmp slt i32 %load.coerce, 10
  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i64 5
  %ret.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr
  %out_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i64 6
  %ret.0 = load ptr, ptr addrspace(5) %out_of_line_values.i, align 8
  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
  call void @llvm.lifetime.end(i64 8, ptr addrspace(0) %retval.0.i)
  ret void
}

; Variant of select with valid addrspacecast in both branches on path to root alloca (ic opt remains enabled)
define void @addrspacecast_both_paths(ptr addrspace(4) align 8 byref([2 x i8]) %arg) {
; CHECK-LABEL: define void @addrspacecast_both_paths(ptr addrspace(4) byref([2 x i8]) align 8 %arg) {
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
  %coerce = alloca [2 x i8], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %coerce, ptr addrspace(4) align 8 %arg, i64 16, i1 false)
  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
  %cmp.i = icmp slt i32 %load.coerce, 10
  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i64 5
  %ret.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr
  %in_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i64 6
  %ret.0 = addrspacecast ptr addrspace(5) %in_of_line_values.i to ptr
  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
  call void @llvm.lifetime.start(i64 8, ptr addrspace(0) %retval.0.i)
  ret void
}

