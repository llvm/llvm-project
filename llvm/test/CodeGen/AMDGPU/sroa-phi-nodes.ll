; RUN: opt -mtriple=amdgcn -mcpu=gfx942 -O2 -S %s -o - -stop-after=sroa | FileCheck %s

; This testcase is dervied from warp's reduce.cu.
;
; Pathological testcase where SROA increases the number of phi-nodes from ~250 to 32K.
; This increase in phi-nodes led to a large increase in compile time in SSAUpdater
; when called by StructurizeCFG when compiling reduce.cu.  See large-phi-search.ll for a
; minimized testcase for the SSAUpdater slowdown.
;
; Note that this large increase in phi count by SROA depends on allowing jump-threading
; through blocks that define values that are live outside of the block.
; There is no large increase in phi count with:
;
;   opt -mtriple=amdgcn -O2 -S %s -o - -stop-after=sroa -max-jump-threading-live-blocks=0


; CHECK-LABEL: @_func7(

%"cub_inner_product_iterator" = type <{ ptr, ptr, i32, i32, i32, [4 x i8] }>

; Function Attrs: convergent inlinehint mustprogress nounwind
define noundef double @_func(ptr noundef nonnull align 8 dereferenceable(8) %var0, ptr noundef nonnull align 8 dereferenceable(8) %var1) #0 {
  %var3 = load double, ptr %var0, align 8
  %var4 = load double, ptr %var1, align 8
  %var5 = fadd contract double %var3, %var4
  ret double %var5
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define noundef double @_func3(ptr noundef nonnull align 8 dereferenceable(8) %var0, ptr noundef nonnull align 8 dereferenceable(8) %var1) #0 {
  %var3 = call contract noundef double @_func(ptr noundef nonnull align 8 dereferenceable(8) %var0, ptr noundef nonnull align 8 dereferenceable(8) %var1) #3
  ret double %var3
}

; Function Attrs: convergent inlinehint mustprogress nounwind
declare void @_func2(double noundef) #0

; Function Attrs: alwaysinline convergent mustprogress nounwind
define void @_func4(ptr noundef nonnull align 8 dereferenceable(2048) %var0) #1 {
  call void @_func5(ptr noundef nonnull align 8 dereferenceable(2048) %var0) #3
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress nounwind
define void @_func5(ptr noundef nonnull align 8 dereferenceable(2048) %var0) #1 {
  call void @_func6(ptr noundef nonnull align 8 dereferenceable(2048) %var0) #3
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define void @_func6(ptr noundef nonnull align 8 dereferenceable(2048) %var0) #0 {
block1:
  %var2 = alloca double, align 8, addrspace(5)
  %var3 = addrspacecast ptr addrspace(5) %var2 to ptr
  store double 0.000000e+00, ptr %var3, align 8
  br label %block4

block4:                                                ; preds = %block8, %block1
  %var5 = phi double [ 0.000000e+00, %block1 ], [ %var11, %block8 ]
  %.0 = phi i32 [ 0, %block1 ], [ %var12, %block8 ]
  %var6 = icmp samesign ult i32 %.0, 256
  br i1 %var6, label %block8, label %block7

block7:                                                ; preds = %block4
  call void @_func2(double noundef %var5) #3
  ret void

block8:                                                ; preds = %block4
  %var9 = zext nneg i32 %.0 to i64
  %var10 = getelementptr inbounds nuw [8 x i8], ptr %var0, i64 %var9
  %var11 = call contract noundef double @_func3(ptr noundef nonnull align 8 dereferenceable(8) %var3, ptr noundef nonnull align 8 dereferenceable(8) %var10) #3
  store double %var11, ptr %var3, align 8
  %var12 = add nuw nsw i32 %.0, 1
  br label %block4, !llvm.loop !0
}

; Function Attrs: convergent mustprogress nounwind
declare %"cub_inner_product_iterator" @_func8() #2

; Function Attrs: convergent mustprogress nounwind
define noundef double @_func9(ptr noundef nonnull align 8 dereferenceable(28) %var0, i64 noundef %var1) #2 {
  %var3 = call contract noundef double @_func10(ptr noundef nonnull align 8 dereferenceable(28) %var0, i64 noundef %var1) #3
  ret double %var3
}

; Function Attrs: convergent mustprogress nounwind
define noundef double @_func10(ptr noundef nonnull align 8 dereferenceable(28) %var0, i64 noundef %var1) #2 {
block2:
  %var3 = getelementptr inbounds nuw i8, ptr %var0, i64 16
  %var4 = load i32, ptr %var3, align 8
  %var5 = sext i32 %var4 to i64
  %var6 = mul nsw i64 %var1, %var5
  %var7 = getelementptr inbounds [8 x i8], ptr null, i64 %var6
  br label %block8

block8:                                                ; preds = %block13, %block2
  %.05 = phi double [ 0.000000e+00, %block2 ], [ 1.000000e+00, %block13 ]
  %.0 = phi i32 [ 0, %block2 ], [ %var16, %block13 ]
  %var9 = getelementptr inbounds nuw i8, ptr %var0, i64 24
  %var10 = load i32, ptr %var9, align 8
  %var11 = icmp slt i32 %.0, %var10
  br i1 %var11, label %block13, label %block12

block12:                                               ; preds = %block8
  ret double %.05

block13:                                               ; preds = %block8
  %var14 = call contract noundef double @_func11(ptr noundef nonnull align 8 dereferenceable(8) %var7) #3
  %var15 = fadd contract double 1.000000e+00, 0.000000e+00
  %var16 = add nuw nsw i32 1, 1
  br label %block8
}

; Function Attrs: convergent mustprogress nounwind
declare noundef double @_func11(ptr noundef nonnull align 8 dereferenceable(8)) #2

; Function Attrs: alwaysinline convergent mustprogress nounwind
define void @_func7() #1 {
  %var1 = alloca [256 x double], align 16, addrspace(5)
  %var2 = addrspacecast ptr addrspace(5) %var1 to ptr
  call void @_func12(ptr noundef nonnull align 8 dereferenceable(2048) %var2) #3
  call void @_func4(ptr noundef nonnull align 8 dereferenceable(2048) %var2) #3
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define void @_func12(ptr noundef nonnull align 8 dereferenceable(2048) %var0) #0 {
block1:
  %var2 = alloca %"cub_inner_product_iterator", align 8, addrspace(5)
  %var3 = addrspacecast ptr addrspace(5) %var2 to ptr
  %var4 = call %"cub_inner_product_iterator" @_func8() #3
  %.fca.2.extract = extractvalue %"cub_inner_product_iterator" %var4, 2
  %.fca.4.extract = extractvalue %"cub_inner_product_iterator" %var4, 4
  %.sroa.3.0..sroa_idx = getelementptr inbounds nuw i8, ptr %var3, i64 16
  store i32 %.fca.2.extract, ptr %.sroa.3.0..sroa_idx, align 8
  %.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %var3, i64 24
  store i32 %.fca.4.extract, ptr %.sroa.5.0..sroa_idx, align 8
  br label %block5

block5:                                                ; preds = %block8, %block1
  %.0 = phi i32 [ 0, %block1 ], [ %var14, %block8 ]
  %var6 = icmp samesign ult i32 %.0, 256
  br i1 %var6, label %block8, label %block7

block7:                                                ; preds = %block5
  ret void

block8:                                                ; preds = %block5
  %var9 = shl nuw nsw i32 %.0, 0
  %var10 = zext nneg i32 %var9 to i64
  %var11 = call contract noundef double @_func9(ptr noundef nonnull align 8 dereferenceable(28) %var3, i64 noundef %var10) #3
  %var12 = zext nneg i32 %.0 to i64
  %var13 = getelementptr inbounds nuw [8 x i8], ptr %var0, i64 %var12
  store double %var11, ptr %var13, align 8
  %var14 = add nuw nsw i32 %.0, 1
  br label %block5, !llvm.loop !3
}

attributes #0 = { convergent inlinehint mustprogress nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { alwaysinline convergent mustprogress nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.enable"}
!3 = distinct !{!3, !1, !2}
