; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK:     OpEntryPoint Kernel %[[#f1:]] "global_check" %[[#var0:]] %[[#var1:]] %[[#var2:]] %[[#var3:]]
; CHECK:     OpEntryPoint Kernel %[[#f2:]] "writer" %[[#var0:]] %[[#var1:]] %[[#var2:]] %[[#var3:]]
; CHECK:     OpEntryPoint Kernel %[[#f3:]] "reader" %[[#var0:]] %[[#var1:]] %[[#var2:]] %[[#var3:]]
; CHECK-DAG: OpName %[[#var0]]
; CHECK-DAG: OpName %[[#var1]]
; CHECK-DAG: OpName %[[#var2]]
; CHECK-DAG: OpName %[[#var3]]
@var = addrspace(1) global <2 x i8> zeroinitializer, align 2
@g_var = addrspace(1) global <2 x i8> zeroinitializer, align 2
@a_var = addrspace(1) global [2 x <2 x i8>] zeroinitializer, align 2
@p_var = addrspace(1) global <2 x i8> addrspace(1)* null, align 8

define spir_func <2 x i8> @from_buf(<2 x i8> %a) {
entry:
  ret <2 x i8> %a
}

define spir_func <2 x i8> @to_buf(<2 x i8> %a) {
entry:
  ret <2 x i8> %a
}

define spir_kernel void @global_check(i32 addrspace(1)* %out) {
entry:
  %0 = load <2 x i8>, <2 x i8> addrspace(1)* @var, align 2
  %cmp = icmp eq <2 x i8> %0, zeroinitializer
  %sext = select <2 x i1> %cmp, <2 x i8> <i8 -1, i8 -1>, <2 x i8> zeroinitializer
  %cast = icmp slt <2 x i8> %sext, zeroinitializer
  %i1promo = zext <2 x i1> %cast to <2 x i8>
  %call1 = call spir_func i1 @OpAll_v2i8(<2 x i8> %i1promo)
  %call = select i1 %call1, i32 1, i32 0
  %1 = and i8 1, 1
  %tobool = icmp ne i8 %1, 0
  %conv = select i1 %tobool, i32 1, i32 0
  %and = and i32 %conv, %call
  %tobool1 = icmp ne i32 %and, 0
  %frombool = select i1 %tobool1, i8 1, i8 0
  %2 = load <2 x i8>, <2 x i8> addrspace(1)* @g_var, align 2
  %cmp2 = icmp eq <2 x i8> %2, zeroinitializer
  %sext3 = select <2 x i1> %cmp2, <2 x i8> <i8 -1, i8 -1>, <2 x i8> zeroinitializer
  %cast2 = icmp slt <2 x i8> %sext3, zeroinitializer
  %i1promo1 = zext <2 x i1> %cast2 to <2 x i8>
  %call43 = call spir_func i1 @OpAll_v2i8(<2 x i8> %i1promo1)
  %call4 = select i1 %call43, i32 1, i32 0
  %3 = and i8 %frombool, 1
  %tobool5 = icmp ne i8 %3, 0
  %conv6 = select i1 %tobool5, i32 1, i32 0
  %and7 = and i32 %conv6, %call4
  %tobool8 = icmp ne i32 %and7, 0
  %frombool9 = select i1 %tobool8, i8 1, i8 0
  %4 = getelementptr inbounds [2 x <2 x i8>], [2 x <2 x i8>] addrspace(1)* @a_var, i64 0, i64 0
  %5 = load <2 x i8>, <2 x i8> addrspace(1)* %4, align 2
  %cmp10 = icmp eq <2 x i8> %5, zeroinitializer
  %sext11 = select <2 x i1> %cmp10, <2 x i8> <i8 -1, i8 -1>, <2 x i8> zeroinitializer
  %cast4 = icmp slt <2 x i8> %sext11, zeroinitializer
  %i1promo2 = zext <2 x i1> %cast4 to <2 x i8>
  %call125 = call spir_func i1 @OpAll_v2i8(<2 x i8> %i1promo2)
  %call12 = select i1 %call125, i32 1, i32 0
  %6 = and i8 %frombool9, 1
  %tobool13 = icmp ne i8 %6, 0
  %conv14 = select i1 %tobool13, i32 1, i32 0
  %and15 = and i32 %conv14, %call12
  %tobool16 = icmp ne i32 %and15, 0
  %frombool17 = select i1 %tobool16, i8 1, i8 0
  %7 = getelementptr inbounds [2 x <2 x i8>], [2 x <2 x i8>] addrspace(1)* @a_var, i64 0, i64 1
  %8 = load <2 x i8>, <2 x i8> addrspace(1)* %7, align 2
  %cmp18 = icmp eq <2 x i8> %8, zeroinitializer
  %sext19 = select <2 x i1> %cmp18, <2 x i8> <i8 -1, i8 -1>, <2 x i8> zeroinitializer
  %cast6 = icmp slt <2 x i8> %sext19, zeroinitializer
  %i1promo3 = zext <2 x i1> %cast6 to <2 x i8>
  %call207 = call spir_func i1 @OpAll_v2i8(<2 x i8> %i1promo3)
  %call20 = select i1 %call207, i32 1, i32 0
  %9 = and i8 %frombool17, 1
  %tobool21 = icmp ne i8 %9, 0
  %conv22 = select i1 %tobool21, i32 1, i32 0
  %and23 = and i32 %conv22, %call20
  %tobool24 = icmp ne i32 %and23, 0
  %frombool25 = select i1 %tobool24, i8 1, i8 0
  %10 = load <2 x i8> addrspace(1)*, <2 x i8> addrspace(1)* addrspace(1)* @p_var, align 8
  %11 = ptrtoint <2 x i8> addrspace(1)* %10 to i64
  %12 = ptrtoint <2 x i8> addrspace(1)* null to i64
  %cmp26 = icmp eq i64 %11, %12
  %conv27 = select i1 %cmp26, i32 1, i32 0
  %13 = and i8 %frombool25, 1
  %tobool28 = icmp ne i8 %13, 0
  %conv29 = select i1 %tobool28, i32 1, i32 0
  %and30 = and i32 %conv29, %conv27
  %tobool31 = icmp ne i32 %and30, 0
  %frombool32 = select i1 %tobool31, i8 1, i8 0
  %14 = and i8 %frombool32, 1
  %tobool33 = icmp ne i8 %14, 0
  %15 = select i1 %tobool33, i64 1, i64 0
  %cond = select i1 %tobool33, i32 1, i32 0
  store i32 %cond, i32 addrspace(1)* %out, align 4
  ret void
}

declare spir_func i1 @OpAll_v2i8(<2 x i8>)

define spir_kernel void @writer(<2 x i8> addrspace(1)* %src, i32 %idx) {
entry:
  %arrayidx = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %src, i64 0
  %0 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx, align 2
  %call = call spir_func <2 x i8> @from_buf(<2 x i8> %0)
  store <2 x i8> %call, <2 x i8> addrspace(1)* @var, align 2
  %arrayidx1 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %src, i64 1
  %1 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx1, align 2
  %call2 = call spir_func <2 x i8> @from_buf(<2 x i8> %1)
  store <2 x i8> %call2, <2 x i8> addrspace(1)* @g_var, align 2
  %arrayidx3 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %src, i64 2
  %2 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx3, align 2
  %call4 = call spir_func <2 x i8> @from_buf(<2 x i8> %2)
  %3 = getelementptr inbounds [2 x <2 x i8>], [2 x <2 x i8>] addrspace(1)* @a_var, i64 0, i64 0
  store <2 x i8> %call4, <2 x i8> addrspace(1)* %3, align 2
  %arrayidx5 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %src, i64 3
  %4 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx5, align 2
  %call6 = call spir_func <2 x i8> @from_buf(<2 x i8> %4)
  %5 = getelementptr inbounds [2 x <2 x i8>], [2 x <2 x i8>] addrspace(1)* @a_var, i64 0, i64 1
  store <2 x i8> %call6, <2 x i8> addrspace(1)* %5, align 2
  %idx.ext = zext i32 %idx to i64
  %add.ptr = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %3, i64 %idx.ext
  store <2 x i8> addrspace(1)* %add.ptr, <2 x i8> addrspace(1)* addrspace(1)* @p_var, align 8
  ret void
}

define spir_kernel void @reader(<2 x i8> addrspace(1)* %dest, <2 x i8> %ptr_write_val) {
entry:
  %call = call spir_func <2 x i8> @from_buf(<2 x i8> %ptr_write_val)
  %0 = load <2 x i8> addrspace(1)*, <2 x i8> addrspace(1)* addrspace(1)* @p_var, align 8
  store <2 x i8> %call, <2 x i8> addrspace(1)* %0, align 2
  %1 = load <2 x i8>, <2 x i8> addrspace(1)* @var, align 2
  %call1 = call spir_func <2 x i8> @to_buf(<2 x i8> %1)
  %arrayidx = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %dest, i64 0
  store <2 x i8> %call1, <2 x i8> addrspace(1)* %arrayidx, align 2
  %2 = load <2 x i8>, <2 x i8> addrspace(1)* @g_var, align 2
  %call2 = call spir_func <2 x i8> @to_buf(<2 x i8> %2)
  %arrayidx3 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %dest, i64 1
  store <2 x i8> %call2, <2 x i8> addrspace(1)* %arrayidx3, align 2
  %3 = getelementptr inbounds [2 x <2 x i8>], [2 x <2 x i8>] addrspace(1)* @a_var, i64 0, i64 0
  %4 = load <2 x i8>, <2 x i8> addrspace(1)* %3, align 2
  %call4 = call spir_func <2 x i8> @to_buf(<2 x i8> %4)
  %arrayidx5 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %dest, i64 2
  store <2 x i8> %call4, <2 x i8> addrspace(1)* %arrayidx5, align 2
  %5 = getelementptr inbounds [2 x <2 x i8>], [2 x <2 x i8>] addrspace(1)* @a_var, i64 0, i64 1
  %6 = load <2 x i8>, <2 x i8> addrspace(1)* %5, align 2
  %call6 = call spir_func <2 x i8> @to_buf(<2 x i8> %6)
  %arrayidx7 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %dest, i64 3
  store <2 x i8> %call6, <2 x i8> addrspace(1)* %arrayidx7, align 2
  ret void
}
