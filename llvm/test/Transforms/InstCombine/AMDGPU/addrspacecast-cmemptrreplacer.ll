; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=instcombine %s | FileCheck %s

declare void @byval(ptr byval([2 x i8]))

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
; CHECK-NEXT:  call void @byval(ptr %retval.0.i)
; CHECK-NEXT:  ret void
; CHECK-NEXT:}
  %coerce = alloca [2 x i8], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %coerce, ptr addrspace(4) align 8 %arg, i64 16, i1 false)
  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
  %cmp.i = icmp slt i32 %load.coerce, 10
  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 5
  %ret.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr
  %out_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 6
  %ret.0 = load ptr, ptr addrspace(5) %out_of_line_values.i, align 8
  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
  call void @byval(ptr %retval.0.i)
  ret void
}

; Variant of select with valid addrspacecast in both branches on path to root alloca (ic opt remains enabled)
define void @addrspacecast_both_paths(ptr addrspace(4) align 8 byref([2 x i8]) %arg) {
; CHECK-LABEL: define void @addrspacecast_both_paths(ptr addrspace(4) byref([2 x i8]) align 8 %arg) {
; CHECK-NEXT:   %coerce = alloca [2 x i8], align 8, addrspace(5)
; CHECK-NEXT:   call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 8 dereferenceable(16) %coerce, ptr addrspace(4) noundef align 8 dereferenceable(16) %arg, i64 16, i1 false)
; CHECK-NEXT:   %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
; CHECK-NEXT:   %cmp.i = icmp sgt i32 %load.coerce, 9
; CHECK-NEXT:   %retval.0.i.v.idx = zext i1 %cmp.i to i32
; CHECK-NEXT:   %retval.0.i.v = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 %retval.0.i.v.idx
; CHECK-NEXT:   %retval.0.i = addrspacecast ptr addrspace(5) %retval.0.i.v to ptr
; CHECK-NEXT:   call void @byval(ptr %retval.0.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
  %coerce = alloca [2 x i8], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %coerce, ptr addrspace(4) align 8 %arg, i64 16, i1 false)
  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
  %cmp.i = icmp slt i32 %load.coerce, 10
  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 0
  %ret.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr
  %in_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 1
  %ret.0 = addrspacecast ptr addrspace(5) %in_of_line_values.i to ptr
  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
  call void @byval(ptr %retval.0.i)
  ret void
}

; Variant of select with multiple valid addrspacecast in both paths
define void @addrspacecast_multi_asc(ptr addrspace(4) align 8 byref([2 x i8]) %arg) {
; CHECK-LABEL: define void @addrspacecast_multi_asc(ptr addrspace(4) byref([2 x i8]) align 8 %arg) {
; CHECK-NEXT:   %coerce = alloca [2 x i8], align 8, addrspace(5)
; CHECK-NEXT:   call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 8 dereferenceable(16) %coerce, ptr addrspace(4) noundef align 8 dereferenceable(16) %arg, i64 16, i1 false)
; CHECK-NEXT:   %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
; CHECK-NEXT:   %cmp.i = icmp sgt i32 %load.coerce, 9
; CHECK-NEXT:   %retval.0.i.v.idx = zext i1 %cmp.i to i32
; CHECK-NEXT:   %retval.0.i.v = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 %retval.0.i.v.idx
; CHECK-NEXT:   %retval.0.i = addrspacecast ptr addrspace(5) %retval.0.i.v to ptr
; CHECK-NEXT:   call void @byval(ptr %retval.0.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
  %coerce = alloca [2 x i8], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %coerce, ptr addrspace(4) align 8 %arg, i64 16, i1 false)
  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
  %cmp.i = icmp slt i32 %load.coerce, 10
  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 0
  %tmp.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr addrspace(3)
  %ret.1 = addrspacecast ptr addrspace(3) %tmp.1 to ptr
  %in_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 1
  %tmp.a.0 = addrspacecast ptr addrspace(5) %in_of_line_values.i to ptr addrspace(70)
  %tmp.b.0 = addrspacecast ptr addrspace(70) %tmp.a.0 to ptr addrspace(50)
  %ret.0 = addrspacecast ptr addrspace(50) %tmp.b.0 to ptr
  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
  call void @byval(ptr %retval.0.i)
  ret void
}

; Variant of select with multiple valid addrspacecast in both paths and
; intermediate instructions on the false path
define void @addrspacecast_mid_instrs(ptr addrspace(4) align 8 byref([2 x i8]) %arg) {
; CHECK-LABEL:define void @addrspacecast_mid_instrs(ptr addrspace(4) byref([2 x i8]) align 8 %arg) {
; CHECK-NEXT:   %coerce = alloca [2 x i8], align 8, addrspace(5)
; CHECK-NEXT:   call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 8 dereferenceable(16) %coerce, ptr addrspace(4) noundef align 8 dereferenceable(16) %arg, i64 16, i1 false)
; CHECK-NEXT:   %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
; CHECK-NEXT:   %cmp.i = icmp slt i32 %load.coerce, 10
; CHECK-NEXT:   %ret.1 = addrspacecast ptr addrspace(5) %coerce to ptr
; CHECK-NEXT:   %in_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 1
; CHECK-NEXT:   %ret.0 = addrspacecast ptr addrspace(5) %in_of_line_values.i to ptr
; CHECK-NEXT:   %val = load i32, ptr %ret.0, align 8
; CHECK-NEXT:   %val.0 = add i32 %val, 1
; CHECK-NEXT:   store i32 %val.0, ptr %ret.0, align 8
; CHECK-NEXT:   %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
; CHECK-NEXT:   call void @byval(ptr %retval.0.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
  %coerce = alloca [2 x i8], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %coerce, ptr addrspace(4) align 8 %arg, i64 16, i1 false)
  %load.coerce = load i32, ptr addrspace(5) %coerce, align 8
  %cmp.i = icmp slt i32 %load.coerce, 10
  %inline_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 0
  %tmp.1 = addrspacecast ptr addrspace(5) %inline_values.i to ptr addrspace(3)
  %ret.1 = addrspacecast ptr addrspace(3) %tmp.1 to ptr
  %in_of_line_values.i = getelementptr inbounds i8, ptr addrspace(5) %coerce, i32 1
  %tmp.a.0 = addrspacecast ptr addrspace(5) %in_of_line_values.i to ptr addrspace(90)
  %tmp.b.0 = addrspacecast ptr addrspace(90) %tmp.a.0 to ptr addrspace(80)
  %ret.0 = addrspacecast ptr addrspace(80) %tmp.b.0 to ptr
  %val = load i32, ptr addrspace(0) %ret.0, align 8
  %val.0 = add i32 1, %val
  store i32 %val.0, ptr %ret.0, align 8
  %retval.0.i = select i1 %cmp.i, ptr %ret.1, ptr %ret.0
  call void @byval(ptr %retval.0.i)
  ret void
}
