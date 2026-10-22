; CUDA samples matrixMul (16 x 16 threadgroup tiles in threadgroup memory, unrolled k loop), the PTX nvcc
; emits for it compiled by ZLUDA (zoc --arch agx-g14c, the .opt.ll output).
; agx-cost: --uniform 6=64 --uniform 7=64 --wg 16,16
target triple = "agx-apple-macosx"

@As = external addrspace(3) global [1024 x i8], align 4
@Bs = external addrspace(3) global [1024 x i8], align 4

; Function Attrs: norecurse nounwind denormal_fpenv(preservesign, float: ieee)
define agx_kernel void @matrixMul(ptr addrspace(4) nofree readonly byref(i64) captures(none) %"315", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"316", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"317", ptr addrspace(4) nofree readonly byref(i32) captures(none) %"318", ptr addrspace(4) nofree readonly byref(i32) captures(none) %"319") local_unnamed_addr #0 {
"312":
  %0 = load i64, ptr addrspace(4) %"315", align 8
  %1 = load i32, ptr addrspace(4) %"318", align 4
  %2 = load i32, ptr addrspace(4) %"319", align 4
  %vx.i102 = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %vy.i105 = tail call i32 @llvm.agx.threadgroup.position.in.grid.y()
  %vx.i = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %vy.i101 = tail call i32 @llvm.agx.thread.position.in.threadgroup.y()
  %3 = shl i32 %1, 4
  %"675" = mul i32 %3, %vy.i105
  %"678" = add i32 %1, -1
  %"681" = add i32 %"678", %"675"
  %4 = shl i32 %vx.i102, 4
  %5 = shl i32 %2, 4
  %6 = icmp sgt i32 %"675", %"681"
  br i1 %6, label %"312.71_crit_edge", label %"178"

"312.71_crit_edge":                               ; preds = %"312"
  %.pre = mul i32 %2, %vy.i101
  br label %"71"

"178":                                            ; preds = %"312"
  %7 = load i64, ptr addrspace(4) %"317", align 8
  %8 = load i64, ptr addrspace(4) %"316", align 8
  %9 = mul i32 %1, %vy.i101
  %"685" = add i32 %9, %vx.i
  %10 = mul i32 %2, %vy.i101
  %"689" = add i32 %10, %vx.i
  %11 = shl i32 %vy.i101, 6
  %12 = shl i32 %vx.i, 2
  %"695" = add i32 %11, ptrtoint (ptr addrspace(3) @As to i32)
  %"698" = add i32 %"695", %12
  %"703" = add i32 %12, ptrtoint (ptr addrspace(3) @Bs to i32)
  %"706" = add i32 %"703", %11
  %13 = inttoptr i64 %8 to ptr
  %14 = addrspacecast ptr %13 to ptr addrspace(1)
  %"477" = ptrtoint ptr addrspace(1) %14 to i64
  %15 = inttoptr i64 %7 to ptr
  %16 = addrspacecast ptr %15 to ptr addrspace(1)
  %"479" = ptrtoint ptr addrspace(1) %16 to i64
  %17 = zext i32 %"698" to i64
  %"722" = inttoptr i64 %17 to ptr addrspace(3)
  %18 = zext i32 %"706" to i64
  %"732" = inttoptr i64 %18 to ptr addrspace(3)
  %19 = zext i32 %"695" to i64
  %"733" = inttoptr i64 %19 to ptr addrspace(3)
  %20 = zext i32 %"703" to i64
  %"734" = inttoptr i64 %20 to ptr addrspace(3)
  %"250" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 4
  %"252" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 64
  %"254" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 8
  %"256" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 128
  %"258" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 12
  %"260" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 192
  %"262" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 16
  %"264" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 256
  %"266" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 20
  %"268" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 320
  %"270" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 24
  %"272" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 384
  %"274" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 28
  %"276" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 448
  %"278" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 32
  %"280" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 512
  %"282" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 36
  %"284" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 576
  %"286" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 40
  %"288" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 640
  %"290" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 44
  %"292" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 704
  %"294" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 48
  %"296" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 768
  %"298" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 52
  %"300" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 832
  %"302" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 56
  %"304" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 896
  %"306" = getelementptr inbounds nuw i8, ptr addrspace(3) %"733", i64 60
  %"308" = getelementptr inbounds nuw i8, ptr addrspace(3) %"734", i64 960
  br label %"70"

"70":                                             ; preds = %"70", %"178"
  %"386.0" = phi i32 [ %"675", %"178" ], [ %"768", %"70" ]
  %"387.0" = phi i32 [ %4, %"178" ], [ %"765", %"70" ]
  %"324.0" = phi float [ 0.000000e+00, %"178" ], [ %"633", %"70" ]
  %"713" = add i32 %"685", %"386.0"
  %21 = sext i32 %"713" to i64
  %"716" = shl nsw i64 %21, 2
  %"718" = add i64 %"716", %"477"
  %"721" = inttoptr i64 %"718" to ptr addrspace(1)
  %22 = load float, ptr addrspace(1) %"721", align 4
  store float %22, ptr addrspace(3) %"722", align 4
  %"723" = add i32 %"689", %"387.0"
  %23 = sext i32 %"723" to i64
  %"726" = shl nsw i64 %23, 2
  %"728" = add i64 %"726", %"479"
  %"731" = inttoptr i64 %"728" to ptr addrspace(1)
  %24 = load float, ptr addrspace(1) %"731", align 4
  store float %24, ptr addrspace(3) %"732", align 4
  fence syncscope("workgroup") seq_cst
  tail call void @llvm.agx.threadgroup.barrier()
  %25 = load float, ptr addrspace(3) %"733", align 4
  %26 = load float, ptr addrspace(3) %"734", align 4
  %"513" = tail call float @llvm.fma.f32(float %25, float %26, float %"324.0")
  %27 = load float, ptr addrspace(3) %"250", align 4
  %28 = load float, ptr addrspace(3) %"252", align 4
  %"521" = tail call float @llvm.fma.f32(float %27, float %28, float %"513")
  %29 = load float, ptr addrspace(3) %"254", align 4
  %30 = load float, ptr addrspace(3) %"256", align 4
  %"529" = tail call float @llvm.fma.f32(float %29, float %30, float %"521")
  %31 = load float, ptr addrspace(3) %"258", align 4
  %32 = load float, ptr addrspace(3) %"260", align 4
  %"537" = tail call float @llvm.fma.f32(float %31, float %32, float %"529")
  %33 = load float, ptr addrspace(3) %"262", align 4
  %34 = load float, ptr addrspace(3) %"264", align 4
  %"545" = tail call float @llvm.fma.f32(float %33, float %34, float %"537")
  %35 = load float, ptr addrspace(3) %"266", align 4
  %36 = load float, ptr addrspace(3) %"268", align 4
  %"553" = tail call float @llvm.fma.f32(float %35, float %36, float %"545")
  %37 = load float, ptr addrspace(3) %"270", align 4
  %38 = load float, ptr addrspace(3) %"272", align 4
  %"561" = tail call float @llvm.fma.f32(float %37, float %38, float %"553")
  %39 = load float, ptr addrspace(3) %"274", align 4
  %40 = load float, ptr addrspace(3) %"276", align 4
  %"569" = tail call float @llvm.fma.f32(float %39, float %40, float %"561")
  %41 = load float, ptr addrspace(3) %"278", align 4
  %42 = load float, ptr addrspace(3) %"280", align 4
  %"577" = tail call float @llvm.fma.f32(float %41, float %42, float %"569")
  %43 = load float, ptr addrspace(3) %"282", align 4
  %44 = load float, ptr addrspace(3) %"284", align 4
  %"585" = tail call float @llvm.fma.f32(float %43, float %44, float %"577")
  %45 = load float, ptr addrspace(3) %"286", align 4
  %46 = load float, ptr addrspace(3) %"288", align 4
  %"593" = tail call float @llvm.fma.f32(float %45, float %46, float %"585")
  %47 = load float, ptr addrspace(3) %"290", align 4
  %48 = load float, ptr addrspace(3) %"292", align 4
  %"601" = tail call float @llvm.fma.f32(float %47, float %48, float %"593")
  %49 = load float, ptr addrspace(3) %"294", align 4
  %50 = load float, ptr addrspace(3) %"296", align 4
  %"609" = tail call float @llvm.fma.f32(float %49, float %50, float %"601")
  %51 = load float, ptr addrspace(3) %"298", align 4
  %52 = load float, ptr addrspace(3) %"300", align 4
  %"617" = tail call float @llvm.fma.f32(float %51, float %52, float %"609")
  %53 = load float, ptr addrspace(3) %"302", align 4
  %54 = load float, ptr addrspace(3) %"304", align 4
  %"625" = tail call float @llvm.fma.f32(float %53, float %54, float %"617")
  %55 = load float, ptr addrspace(3) %"306", align 4
  %56 = load float, ptr addrspace(3) %"308", align 4
  %"633" = tail call float @llvm.fma.f32(float %55, float %56, float %"625")
  fence syncscope("workgroup") seq_cst
  tail call void @llvm.agx.threadgroup.barrier()
  %"765" = add i32 %"387.0", %5
  %"768" = add i32 %"386.0", 16
  %.not = icmp sgt i32 %"768", %"681"
  br i1 %.not, label %"71", label %"70"

"71":                                             ; preds = %"70", %"312.71_crit_edge"
  %.pre-phi = phi i32 [ %.pre, %"312.71_crit_edge" ], [ %10, %"70" ]
  %"324.1" = phi float [ 0.000000e+00, %"312.71_crit_edge" ], [ %"633", %"70" ]
  %57 = mul i32 %5, %vy.i105
  %"772" = add i32 %57, %4
  %"776" = add i32 %"772", %vx.i
  %"780" = add i32 %"776", %.pre-phi
  %58 = inttoptr i64 %0 to ptr
  %59 = addrspacecast ptr %58 to ptr addrspace(1)
  %"657" = ptrtoint ptr addrspace(1) %59 to i64
  %60 = sext i32 %"780" to i64
  %"783" = shl nsw i64 %60, 2
  %"785" = add i64 %"783", %"657"
  %"788" = inttoptr i64 %"785" to ptr addrspace(1)
  store float %"324.1", ptr addrspace(1) %"788", align 4
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.y() #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.agx.threadgroup.barrier() #3

attributes #0 = { norecurse nounwind denormal_fpenv(preservesign, float: ieee) "no-trapping-math"="true" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nofree nounwind willreturn }
