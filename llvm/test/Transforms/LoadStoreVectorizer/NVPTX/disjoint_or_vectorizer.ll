; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @copy_fusion_1(i32 %workgroup_id_x, i32 %workitem_id_x, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg0, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg1, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg2, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg3, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg4, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg5, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg6, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg7, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg8, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg9, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg10, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg11, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg12, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg13, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg14, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg15, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg16, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg17, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg18, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg19, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg20, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg21, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg22, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg23, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg24, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg25, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg26, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg27, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg28, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg29, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg30, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg31, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg32, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg33, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg34, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg35, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg36, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg37, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg38, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg39, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg40, ptr noalias nocapture readonly align 16 dereferenceable(23592960) %arg41, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg42, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg43, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg44, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg45, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg46, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg47, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg48, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg49, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg50, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg51, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg52, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg53, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg54, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg55, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg56, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg57, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg58, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg59, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg60, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg61, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg62, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg63, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg64, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg65, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg66, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg67, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg68, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg69, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg70, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg71, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg72, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg73, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg74, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg75, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg76, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg77, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg78, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg79, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg80, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg81, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg82, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg83, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg84, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg85, ptr noalias nocapture writeonly align 128 dereferenceable(23592960) %arg86) local_unnamed_addr #5 {
entry:
  %arg86180 = addrspacecast ptr %arg86 to ptr addrspace(1)
  %arg85178 = addrspacecast ptr %arg85 to ptr addrspace(1)
  %arg84176 = addrspacecast ptr %arg84 to ptr addrspace(1)
  %arg83174 = addrspacecast ptr %arg83 to ptr addrspace(1)
  %arg82172 = addrspacecast ptr %arg82 to ptr addrspace(1)
  %arg81170 = addrspacecast ptr %arg81 to ptr addrspace(1)
  %arg80168 = addrspacecast ptr %arg80 to ptr addrspace(1)
  %arg79166 = addrspacecast ptr %arg79 to ptr addrspace(1)
  %arg78164 = addrspacecast ptr %arg78 to ptr addrspace(1)
  %arg77162 = addrspacecast ptr %arg77 to ptr addrspace(1)
  %arg76160 = addrspacecast ptr %arg76 to ptr addrspace(1)
  %arg75158 = addrspacecast ptr %arg75 to ptr addrspace(1)
  %arg74156 = addrspacecast ptr %arg74 to ptr addrspace(1)
  %arg73154 = addrspacecast ptr %arg73 to ptr addrspace(1)
  %arg72152 = addrspacecast ptr %arg72 to ptr addrspace(1)
  %arg71150 = addrspacecast ptr %arg71 to ptr addrspace(1)
  %arg70148 = addrspacecast ptr %arg70 to ptr addrspace(1)
  %arg69146 = addrspacecast ptr %arg69 to ptr addrspace(1)
  %arg68144 = addrspacecast ptr %arg68 to ptr addrspace(1)
  %arg67142 = addrspacecast ptr %arg67 to ptr addrspace(1)
  %arg66140 = addrspacecast ptr %arg66 to ptr addrspace(1)
  %arg65138 = addrspacecast ptr %arg65 to ptr addrspace(1)
  %arg64136 = addrspacecast ptr %arg64 to ptr addrspace(1)
  %arg63134 = addrspacecast ptr %arg63 to ptr addrspace(1)
  %arg62132 = addrspacecast ptr %arg62 to ptr addrspace(1)
  %arg61130 = addrspacecast ptr %arg61 to ptr addrspace(1)
  %arg60128 = addrspacecast ptr %arg60 to ptr addrspace(1)
  %arg59126 = addrspacecast ptr %arg59 to ptr addrspace(1)
  %arg58124 = addrspacecast ptr %arg58 to ptr addrspace(1)
  %arg57122 = addrspacecast ptr %arg57 to ptr addrspace(1)
  %arg56120 = addrspacecast ptr %arg56 to ptr addrspace(1)
  %arg55118 = addrspacecast ptr %arg55 to ptr addrspace(1)
  %arg54116 = addrspacecast ptr %arg54 to ptr addrspace(1)
  %arg53114 = addrspacecast ptr %arg53 to ptr addrspace(1)
  %arg52112 = addrspacecast ptr %arg52 to ptr addrspace(1)
  %arg51110 = addrspacecast ptr %arg51 to ptr addrspace(1)
  %arg50108 = addrspacecast ptr %arg50 to ptr addrspace(1)
  %arg49106 = addrspacecast ptr %arg49 to ptr addrspace(1)
  %arg48104 = addrspacecast ptr %arg48 to ptr addrspace(1)
  %arg47102 = addrspacecast ptr %arg47 to ptr addrspace(1)
  %arg46100 = addrspacecast ptr %arg46 to ptr addrspace(1)
  %arg4598 = addrspacecast ptr %arg45 to ptr addrspace(1)
  %arg4496 = addrspacecast ptr %arg44 to ptr addrspace(1)
  %arg4394 = addrspacecast ptr %arg43 to ptr addrspace(1)
  %arg4292 = addrspacecast ptr %arg42 to ptr addrspace(1)
  %arg4190 = addrspacecast ptr %arg41 to ptr addrspace(1)
  %arg4088 = addrspacecast ptr %arg40 to ptr addrspace(1)
  %arg3986 = addrspacecast ptr %arg39 to ptr addrspace(1)
  %arg3884 = addrspacecast ptr %arg38 to ptr addrspace(1)
  %arg3782 = addrspacecast ptr %arg37 to ptr addrspace(1)
  %arg3680 = addrspacecast ptr %arg36 to ptr addrspace(1)
  %arg3578 = addrspacecast ptr %arg35 to ptr addrspace(1)
  %arg3476 = addrspacecast ptr %arg34 to ptr addrspace(1)
  %arg3374 = addrspacecast ptr %arg33 to ptr addrspace(1)
  %arg3272 = addrspacecast ptr %arg32 to ptr addrspace(1)
  %arg3170 = addrspacecast ptr %arg31 to ptr addrspace(1)
  %arg3068 = addrspacecast ptr %arg30 to ptr addrspace(1)
  %arg2966 = addrspacecast ptr %arg29 to ptr addrspace(1)
  %arg2864 = addrspacecast ptr %arg28 to ptr addrspace(1)
  %arg2762 = addrspacecast ptr %arg27 to ptr addrspace(1)
  %arg2660 = addrspacecast ptr %arg26 to ptr addrspace(1)
  %arg2558 = addrspacecast ptr %arg25 to ptr addrspace(1)
  %arg2456 = addrspacecast ptr %arg24 to ptr addrspace(1)
  %arg2354 = addrspacecast ptr %arg23 to ptr addrspace(1)
  %arg2252 = addrspacecast ptr %arg22 to ptr addrspace(1)
  %arg2150 = addrspacecast ptr %arg21 to ptr addrspace(1)
  %arg2048 = addrspacecast ptr %arg20 to ptr addrspace(1)
  %arg1946 = addrspacecast ptr %arg19 to ptr addrspace(1)
  %arg1844 = addrspacecast ptr %arg18 to ptr addrspace(1)
  %arg1742 = addrspacecast ptr %arg17 to ptr addrspace(1)
  %arg1640 = addrspacecast ptr %arg16 to ptr addrspace(1)
  %arg1538 = addrspacecast ptr %arg15 to ptr addrspace(1)
  %arg1436 = addrspacecast ptr %arg14 to ptr addrspace(1)
  %arg1334 = addrspacecast ptr %arg13 to ptr addrspace(1)
  %arg1232 = addrspacecast ptr %arg12 to ptr addrspace(1)
  %arg1130 = addrspacecast ptr %arg11 to ptr addrspace(1)
  %arg1028 = addrspacecast ptr %arg10 to ptr addrspace(1)
  %arg926 = addrspacecast ptr %arg9 to ptr addrspace(1)
  %arg824 = addrspacecast ptr %arg8 to ptr addrspace(1)
  %arg722 = addrspacecast ptr %arg7 to ptr addrspace(1)
  %arg620 = addrspacecast ptr %arg6 to ptr addrspace(1)
  %arg518 = addrspacecast ptr %arg5 to ptr addrspace(1)
  %arg416 = addrspacecast ptr %arg4 to ptr addrspace(1)
  %arg314 = addrspacecast ptr %arg3 to ptr addrspace(1)
  %arg212 = addrspacecast ptr %arg2 to ptr addrspace(1)
  %arg110 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %arg02 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %2 = shl i32 %0, 9
  %3 = shl i32 %1, 2
  %4 = and i32 %3, 60
  %5 = zext i32 %4 to i64
  %6 = add i32 %2, %3
  br label %loop.loop_body

loop.loop_body:                                   ; preds = %entry, %copy_fusion_1.in_bounds-after
  %lsr.iv = phi i32 [ -884736, %entry ], [ %lsr.iv.next, %copy_fusion_1.in_bounds-after ]
  %7 = add i32 %6, %lsr.iv
  %8 = add i32 %7, 884736
  %linear_index_plus_base.fr = freeze i32 %8
  %9 = icmp ult i32 %linear_index_plus_base.fr, 11796480
  br i1 %9, label %copy_fusion_1.in_bounds-true, label %copy_fusion_1.in_bounds-after

copy_fusion_1.in_bounds-after:                    ; preds = %copy_fusion_1.in_bounds-true, %loop.loop_body
  %lsr.iv.next = add nsw i32 %lsr.iv, 884736
  %10 = icmp ugt i32 %lsr.iv.next, 10911743
  br i1 %10, label %loop.loop_exit, label %loop.loop_body

loop.loop_exit:                                   ; preds = %copy_fusion_1.in_bounds-after
  ret void

copy_fusion_1.in_bounds-true:                     ; preds = %loop.loop_body
  %linear_index3 = or i32 %linear_index_plus_base.fr, 3
  %11 = and i32 %linear_index3, 63
  %linear_index2 = or i32 %linear_index_plus_base.fr, 2
  %12 = and i32 %linear_index2, 62
  %linear_index1 = or i32 %linear_index_plus_base.fr, 1
  %13 = and i32 %linear_index1, 61
  %14 = lshr i32 %linear_index_plus_base.fr, 6
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg02, i64 0, i64 0, i64 %15, i64 %5
  %17 = load i16, ptr addrspace(1) %16, align 8
  %18 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg110, i64 0, i64 0, i64 %15, i64 %5
  %19 = load i16, ptr addrspace(1) %18, align 8
  %20 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg212, i64 0, i64 0, i64 %15, i64 %5
  %21 = load i16, ptr addrspace(1) %20, align 8
  %22 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg314, i64 0, i64 0, i64 %15, i64 %5
  %23 = load i16, ptr addrspace(1) %22, align 8
  %24 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg416, i64 0, i64 0, i64 %15, i64 %5
  %25 = load i16, ptr addrspace(1) %24, align 8
  %26 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg518, i64 0, i64 0, i64 %15, i64 %5
  %27 = load i16, ptr addrspace(1) %26, align 8
  %28 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg620, i64 0, i64 0, i64 %15, i64 %5
  %29 = load i16, ptr addrspace(1) %28, align 8
  %30 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg722, i64 0, i64 0, i64 %15, i64 %5
  %31 = load i16, ptr addrspace(1) %30, align 8
  %32 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg824, i64 0, i64 0, i64 %15, i64 %5
  %33 = load i16, ptr addrspace(1) %32, align 8
  %34 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg926, i64 0, i64 0, i64 %15, i64 %5
  %35 = load i16, ptr addrspace(1) %34, align 8
  %36 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1028, i64 0, i64 0, i64 %15, i64 %5
  %37 = load i16, ptr addrspace(1) %36, align 8
  %38 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1130, i64 0, i64 0, i64 %15, i64 %5
  %39 = load i16, ptr addrspace(1) %38, align 8
  %40 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1232, i64 0, i64 0, i64 %15, i64 %5
  %41 = load i16, ptr addrspace(1) %40, align 8
  %42 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1334, i64 0, i64 0, i64 %15, i64 %5
  %43 = load i16, ptr addrspace(1) %42, align 8
  %44 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1436, i64 0, i64 0, i64 %15, i64 %5
  %45 = load i16, ptr addrspace(1) %44, align 8
  %46 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1538, i64 0, i64 0, i64 %15, i64 %5
  %47 = load i16, ptr addrspace(1) %46, align 8
  %48 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1640, i64 0, i64 0, i64 %15, i64 %5
  %49 = load i16, ptr addrspace(1) %48, align 8
  %50 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1742, i64 0, i64 0, i64 %15, i64 %5
  %51 = load i16, ptr addrspace(1) %50, align 8
  %52 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1844, i64 0, i64 0, i64 %15, i64 %5
  %53 = load i16, ptr addrspace(1) %52, align 8
  %54 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1946, i64 0, i64 0, i64 %15, i64 %5
  %55 = load i16, ptr addrspace(1) %54, align 8
  %56 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2048, i64 0, i64 0, i64 %15, i64 %5
  %57 = load i16, ptr addrspace(1) %56, align 8
  %58 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2150, i64 0, i64 0, i64 %15, i64 %5
  %59 = load i16, ptr addrspace(1) %58, align 8
  %60 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2252, i64 0, i64 0, i64 %15, i64 %5
  %61 = load i16, ptr addrspace(1) %60, align 8
  %62 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2354, i64 0, i64 0, i64 %15, i64 %5
  %63 = load i16, ptr addrspace(1) %62, align 8
  %64 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2456, i64 0, i64 0, i64 %15, i64 %5
  %65 = load i16, ptr addrspace(1) %64, align 8
  %66 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2558, i64 0, i64 0, i64 %15, i64 %5
  %67 = load i16, ptr addrspace(1) %66, align 8
  %68 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2660, i64 0, i64 0, i64 %15, i64 %5
  %69 = load i16, ptr addrspace(1) %68, align 8
  %70 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2762, i64 0, i64 0, i64 %15, i64 %5
  %71 = load i16, ptr addrspace(1) %70, align 8
  %72 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2864, i64 0, i64 0, i64 %15, i64 %5
  %73 = load i16, ptr addrspace(1) %72, align 8
  %74 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2966, i64 0, i64 0, i64 %15, i64 %5
  %75 = load i16, ptr addrspace(1) %74, align 8
  %76 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3068, i64 0, i64 0, i64 %15, i64 %5
  %77 = load i16, ptr addrspace(1) %76, align 8
  %78 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3170, i64 0, i64 0, i64 %15, i64 %5
  %79 = load i16, ptr addrspace(1) %78, align 8
  %80 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3272, i64 0, i64 0, i64 %15, i64 %5
  %81 = load i16, ptr addrspace(1) %80, align 8
  %82 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3374, i64 0, i64 0, i64 %15, i64 %5
  %83 = load i16, ptr addrspace(1) %82, align 8
  %84 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3476, i64 0, i64 0, i64 %15, i64 %5
  %85 = load i16, ptr addrspace(1) %84, align 8
  %86 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3578, i64 0, i64 0, i64 %15, i64 %5
  %87 = load i16, ptr addrspace(1) %86, align 8
  %88 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3680, i64 0, i64 0, i64 %15, i64 %5
  %89 = load i16, ptr addrspace(1) %88, align 8
  %90 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3782, i64 0, i64 0, i64 %15, i64 %5
  %91 = load i16, ptr addrspace(1) %90, align 8
  %92 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3884, i64 0, i64 0, i64 %15, i64 %5
  %93 = load i16, ptr addrspace(1) %92, align 8
  %94 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3986, i64 0, i64 0, i64 %15, i64 %5
  %95 = load i16, ptr addrspace(1) %94, align 8
  %96 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4088, i64 0, i64 0, i64 %15, i64 %5
  %97 = load i16, ptr addrspace(1) %96, align 8
  %98 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4190, i64 0, i64 0, i64 %15, i64 %5
  %99 = load i16, ptr addrspace(1) %98, align 8
  %100 = zext i32 %linear_index_plus_base.fr to i64
  %101 = getelementptr inbounds i16, ptr addrspace(1) %arg4292, i64 %100
  store i16 %17, ptr addrspace(1) %101, align 8
  %102 = getelementptr inbounds i16, ptr addrspace(1) %arg4394, i64 %100
  store i16 %19, ptr addrspace(1) %102, align 8
  %103 = getelementptr inbounds i16, ptr addrspace(1) %arg4496, i64 %100
  store i16 %21, ptr addrspace(1) %103, align 8
  %104 = getelementptr inbounds i16, ptr addrspace(1) %arg4598, i64 %100
  store i16 %23, ptr addrspace(1) %104, align 8
  %105 = getelementptr inbounds i16, ptr addrspace(1) %arg46100, i64 %100
  store i16 %25, ptr addrspace(1) %105, align 8
  %106 = getelementptr inbounds i16, ptr addrspace(1) %arg47102, i64 %100
  store i16 %27, ptr addrspace(1) %106, align 8
  %107 = getelementptr inbounds i16, ptr addrspace(1) %arg48104, i64 %100
  store i16 %29, ptr addrspace(1) %107, align 8
  %108 = getelementptr inbounds i16, ptr addrspace(1) %arg49106, i64 %100
  store i16 %31, ptr addrspace(1) %108, align 8
  %109 = getelementptr inbounds i16, ptr addrspace(1) %arg50108, i64 %100
  store i16 %33, ptr addrspace(1) %109, align 8
  %110 = getelementptr inbounds i16, ptr addrspace(1) %arg51110, i64 %100
  store i16 %35, ptr addrspace(1) %110, align 8
  %111 = getelementptr inbounds i16, ptr addrspace(1) %arg52112, i64 %100
  store i16 %37, ptr addrspace(1) %111, align 8
  %112 = getelementptr inbounds i16, ptr addrspace(1) %arg53114, i64 %100
  store i16 %39, ptr addrspace(1) %112, align 8
  %113 = getelementptr inbounds i16, ptr addrspace(1) %arg54116, i64 %100
  store i16 %41, ptr addrspace(1) %113, align 8
  %114 = getelementptr inbounds i16, ptr addrspace(1) %arg55118, i64 %100
  store i16 %43, ptr addrspace(1) %114, align 8
  %115 = getelementptr inbounds i16, ptr addrspace(1) %arg56120, i64 %100
  store i16 %45, ptr addrspace(1) %115, align 8
  %116 = getelementptr inbounds i16, ptr addrspace(1) %arg57122, i64 %100
  store i16 %47, ptr addrspace(1) %116, align 8
  %117 = getelementptr inbounds i16, ptr addrspace(1) %arg58124, i64 %100
  store i16 %49, ptr addrspace(1) %117, align 8
  %118 = getelementptr inbounds i16, ptr addrspace(1) %arg59126, i64 %100
  store i16 %51, ptr addrspace(1) %118, align 8
  %119 = getelementptr inbounds i16, ptr addrspace(1) %arg60128, i64 %100
  store i16 %47, ptr addrspace(1) %119, align 8
  %120 = getelementptr inbounds i16, ptr addrspace(1) %arg61130, i64 %100
  store i16 %49, ptr addrspace(1) %120, align 8
  %121 = getelementptr inbounds i16, ptr addrspace(1) %arg62132, i64 %100
  store i16 %51, ptr addrspace(1) %121, align 8
  %122 = getelementptr inbounds i16, ptr addrspace(1) %arg63134, i64 %100
  store i16 %53, ptr addrspace(1) %122, align 8
  %123 = getelementptr inbounds i16, ptr addrspace(1) %arg64136, i64 %100
  store i16 %55, ptr addrspace(1) %123, align 8
  %124 = getelementptr inbounds i16, ptr addrspace(1) %arg65138, i64 %100
  store i16 %57, ptr addrspace(1) %124, align 8
  %125 = getelementptr inbounds i16, ptr addrspace(1) %arg66140, i64 %100
  store i16 %59, ptr addrspace(1) %125, align 8
  %126 = getelementptr inbounds i16, ptr addrspace(1) %arg67142, i64 %100
  store i16 %61, ptr addrspace(1) %126, align 8
  %127 = getelementptr inbounds i16, ptr addrspace(1) %arg68144, i64 %100
  store i16 %63, ptr addrspace(1) %127, align 8
  %128 = getelementptr inbounds i16, ptr addrspace(1) %arg69146, i64 %100
  store i16 %65, ptr addrspace(1) %128, align 8
  %129 = getelementptr inbounds i16, ptr addrspace(1) %arg70148, i64 %100
  store i16 %67, ptr addrspace(1) %129, align 8
  %130 = getelementptr inbounds i16, ptr addrspace(1) %arg71150, i64 %100
  store i16 %69, ptr addrspace(1) %130, align 8
  %131 = getelementptr inbounds i16, ptr addrspace(1) %arg72152, i64 %100
  store i16 %71, ptr addrspace(1) %131, align 8
  %132 = getelementptr inbounds i16, ptr addrspace(1) %arg73154, i64 %100
  store i16 %73, ptr addrspace(1) %132, align 8
  %133 = getelementptr inbounds i16, ptr addrspace(1) %arg74156, i64 %100
  store i16 %75, ptr addrspace(1) %133, align 8
  %134 = getelementptr inbounds i16, ptr addrspace(1) %arg75158, i64 %100
  store i16 %77, ptr addrspace(1) %134, align 8
  %135 = getelementptr inbounds i16, ptr addrspace(1) %arg76160, i64 %100
  store i16 %79, ptr addrspace(1) %135, align 8
  %136 = getelementptr inbounds i16, ptr addrspace(1) %arg77162, i64 %100
  store i16 %81, ptr addrspace(1) %136, align 8
  %137 = getelementptr inbounds i16, ptr addrspace(1) %arg78164, i64 %100
  store i16 %83, ptr addrspace(1) %137, align 8
  %138 = getelementptr inbounds i16, ptr addrspace(1) %arg79166, i64 %100
  store i16 %85, ptr addrspace(1) %138, align 8
  %139 = getelementptr inbounds i16, ptr addrspace(1) %arg80168, i64 %100
  store i16 %87, ptr addrspace(1) %139, align 8
  %140 = getelementptr inbounds i16, ptr addrspace(1) %arg81170, i64 %100
  store i16 %89, ptr addrspace(1) %140, align 8
  %141 = getelementptr inbounds i16, ptr addrspace(1) %arg82172, i64 %100
  store i16 %91, ptr addrspace(1) %141, align 8
  %142 = getelementptr inbounds i16, ptr addrspace(1) %arg83174, i64 %100
  store i16 %93, ptr addrspace(1) %142, align 8
  %143 = getelementptr inbounds i16, ptr addrspace(1) %arg84176, i64 %100
  store i16 %95, ptr addrspace(1) %143, align 8
  %144 = getelementptr inbounds i16, ptr addrspace(1) %arg85178, i64 %100
  store i16 %97, ptr addrspace(1) %144, align 8
  %145 = getelementptr inbounds i16, ptr addrspace(1) %arg86180, i64 %100
  store i16 %99, ptr addrspace(1) %145, align 8
  %146 = zext i32 %13 to i64
  %147 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg02, i64 0, i64 0, i64 %15, i64 %146
  %148 = load i16, ptr addrspace(1) %147, align 2
  %149 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg110, i64 0, i64 0, i64 %15, i64 %146
  %150 = load i16, ptr addrspace(1) %149, align 2
  %151 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg212, i64 0, i64 0, i64 %15, i64 %146
  %152 = load i16, ptr addrspace(1) %151, align 2
  %153 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg314, i64 0, i64 0, i64 %15, i64 %146
  %154 = load i16, ptr addrspace(1) %153, align 2
  %155 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg416, i64 0, i64 0, i64 %15, i64 %146
  %156 = load i16, ptr addrspace(1) %155, align 2
  %157 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg518, i64 0, i64 0, i64 %15, i64 %146
  %158 = load i16, ptr addrspace(1) %157, align 2
  %159 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg620, i64 0, i64 0, i64 %15, i64 %146
  %160 = load i16, ptr addrspace(1) %159, align 2
  %161 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg722, i64 0, i64 0, i64 %15, i64 %146
  %162 = load i16, ptr addrspace(1) %161, align 2
  %163 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg824, i64 0, i64 0, i64 %15, i64 %146
  %164 = load i16, ptr addrspace(1) %163, align 2
  %165 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg926, i64 0, i64 0, i64 %15, i64 %146
  %166 = load i16, ptr addrspace(1) %165, align 2
  %167 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1028, i64 0, i64 0, i64 %15, i64 %146
  %168 = load i16, ptr addrspace(1) %167, align 2
  %169 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1130, i64 0, i64 0, i64 %15, i64 %146
  %170 = load i16, ptr addrspace(1) %169, align 2
  %171 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1232, i64 0, i64 0, i64 %15, i64 %146
  %172 = load i16, ptr addrspace(1) %171, align 2
  %173 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1334, i64 0, i64 0, i64 %15, i64 %146
  %174 = load i16, ptr addrspace(1) %173, align 2
  %175 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1436, i64 0, i64 0, i64 %15, i64 %146
  %176 = load i16, ptr addrspace(1) %175, align 2
  %177 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1538, i64 0, i64 0, i64 %15, i64 %146
  %178 = load i16, ptr addrspace(1) %177, align 2
  %179 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1640, i64 0, i64 0, i64 %15, i64 %146
  %180 = load i16, ptr addrspace(1) %179, align 2
  %181 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1742, i64 0, i64 0, i64 %15, i64 %146
  %182 = load i16, ptr addrspace(1) %181, align 2
  %183 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1844, i64 0, i64 0, i64 %15, i64 %146
  %184 = load i16, ptr addrspace(1) %183, align 2
  %185 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1946, i64 0, i64 0, i64 %15, i64 %146
  %186 = load i16, ptr addrspace(1) %185, align 2
  %187 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2048, i64 0, i64 0, i64 %15, i64 %146
  %188 = load i16, ptr addrspace(1) %187, align 2
  %189 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2150, i64 0, i64 0, i64 %15, i64 %146
  %190 = load i16, ptr addrspace(1) %189, align 2
  %191 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2252, i64 0, i64 0, i64 %15, i64 %146
  %192 = load i16, ptr addrspace(1) %191, align 2
  %193 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2354, i64 0, i64 0, i64 %15, i64 %146
  %194 = load i16, ptr addrspace(1) %193, align 2
  %195 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2456, i64 0, i64 0, i64 %15, i64 %146
  %196 = load i16, ptr addrspace(1) %195, align 2
  %197 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2558, i64 0, i64 0, i64 %15, i64 %146
  %198 = load i16, ptr addrspace(1) %197, align 2
  %199 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2660, i64 0, i64 0, i64 %15, i64 %146
  %200 = load i16, ptr addrspace(1) %199, align 2
  %201 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2762, i64 0, i64 0, i64 %15, i64 %146
  %202 = load i16, ptr addrspace(1) %201, align 2
  %203 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2864, i64 0, i64 0, i64 %15, i64 %146
  %204 = load i16, ptr addrspace(1) %203, align 2
  %205 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2966, i64 0, i64 0, i64 %15, i64 %146
  %206 = load i16, ptr addrspace(1) %205, align 2
  %207 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3068, i64 0, i64 0, i64 %15, i64 %146
  %208 = load i16, ptr addrspace(1) %207, align 2
  %209 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3170, i64 0, i64 0, i64 %15, i64 %146
  %210 = load i16, ptr addrspace(1) %209, align 2
  %211 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3272, i64 0, i64 0, i64 %15, i64 %146
  %212 = load i16, ptr addrspace(1) %211, align 2
  %213 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3374, i64 0, i64 0, i64 %15, i64 %146
  %214 = load i16, ptr addrspace(1) %213, align 2
  %215 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3476, i64 0, i64 0, i64 %15, i64 %146
  %216 = load i16, ptr addrspace(1) %215, align 2
  %217 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3578, i64 0, i64 0, i64 %15, i64 %146
  %218 = load i16, ptr addrspace(1) %217, align 2
  %219 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3680, i64 0, i64 0, i64 %15, i64 %146
  %220 = load i16, ptr addrspace(1) %219, align 2
  %221 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3782, i64 0, i64 0, i64 %15, i64 %146
  %222 = load i16, ptr addrspace(1) %221, align 2
  %223 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3884, i64 0, i64 0, i64 %15, i64 %146
  %224 = load i16, ptr addrspace(1) %223, align 2
  %225 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3986, i64 0, i64 0, i64 %15, i64 %146
  %226 = load i16, ptr addrspace(1) %225, align 2
  %227 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4088, i64 0, i64 0, i64 %15, i64 %146
  %228 = load i16, ptr addrspace(1) %227, align 2
  %229 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4190, i64 0, i64 0, i64 %15, i64 %146
  %230 = load i16, ptr addrspace(1) %229, align 2
  %231 = zext i32 %linear_index1 to i64
  %232 = getelementptr inbounds i16, ptr addrspace(1) %arg4292, i64 %231
  store i16 %148, ptr addrspace(1) %232, align 2
  %233 = getelementptr inbounds i16, ptr addrspace(1) %arg4394, i64 %231
  store i16 %150, ptr addrspace(1) %233, align 2
  %234 = getelementptr inbounds i16, ptr addrspace(1) %arg4496, i64 %231
  store i16 %152, ptr addrspace(1) %234, align 2
  %235 = getelementptr inbounds i16, ptr addrspace(1) %arg4598, i64 %231
  store i16 %154, ptr addrspace(1) %235, align 2
  %236 = getelementptr inbounds i16, ptr addrspace(1) %arg46100, i64 %231
  store i16 %156, ptr addrspace(1) %236, align 2
  %237 = getelementptr inbounds i16, ptr addrspace(1) %arg47102, i64 %231
  store i16 %158, ptr addrspace(1) %237, align 2
  %238 = getelementptr inbounds i16, ptr addrspace(1) %arg48104, i64 %231
  store i16 %160, ptr addrspace(1) %238, align 2
  %239 = getelementptr inbounds i16, ptr addrspace(1) %arg49106, i64 %231
  store i16 %162, ptr addrspace(1) %239, align 2
  %240 = getelementptr inbounds i16, ptr addrspace(1) %arg50108, i64 %231
  store i16 %164, ptr addrspace(1) %240, align 2
  %241 = getelementptr inbounds i16, ptr addrspace(1) %arg51110, i64 %231
  store i16 %166, ptr addrspace(1) %241, align 2
  %242 = getelementptr inbounds i16, ptr addrspace(1) %arg52112, i64 %231
  store i16 %168, ptr addrspace(1) %242, align 2
  %243 = getelementptr inbounds i16, ptr addrspace(1) %arg53114, i64 %231
  store i16 %170, ptr addrspace(1) %243, align 2
  %244 = getelementptr inbounds i16, ptr addrspace(1) %arg54116, i64 %231
  store i16 %172, ptr addrspace(1) %244, align 2
  %245 = getelementptr inbounds i16, ptr addrspace(1) %arg55118, i64 %231
  store i16 %174, ptr addrspace(1) %245, align 2
  %246 = getelementptr inbounds i16, ptr addrspace(1) %arg56120, i64 %231
  store i16 %176, ptr addrspace(1) %246, align 2
  %247 = getelementptr inbounds i16, ptr addrspace(1) %arg57122, i64 %231
  store i16 %178, ptr addrspace(1) %247, align 2
  %248 = getelementptr inbounds i16, ptr addrspace(1) %arg58124, i64 %231
  store i16 %180, ptr addrspace(1) %248, align 2
  %249 = getelementptr inbounds i16, ptr addrspace(1) %arg59126, i64 %231
  store i16 %182, ptr addrspace(1) %249, align 2
  %250 = getelementptr inbounds i16, ptr addrspace(1) %arg60128, i64 %231
  store i16 %178, ptr addrspace(1) %250, align 2
  %251 = getelementptr inbounds i16, ptr addrspace(1) %arg61130, i64 %231
  store i16 %180, ptr addrspace(1) %251, align 2
  %252 = getelementptr inbounds i16, ptr addrspace(1) %arg62132, i64 %231
  store i16 %182, ptr addrspace(1) %252, align 2
  %253 = getelementptr inbounds i16, ptr addrspace(1) %arg63134, i64 %231
  store i16 %184, ptr addrspace(1) %253, align 2
  %254 = getelementptr inbounds i16, ptr addrspace(1) %arg64136, i64 %231
  store i16 %186, ptr addrspace(1) %254, align 2
  %255 = getelementptr inbounds i16, ptr addrspace(1) %arg65138, i64 %231
  store i16 %188, ptr addrspace(1) %255, align 2
  %256 = getelementptr inbounds i16, ptr addrspace(1) %arg66140, i64 %231
  store i16 %190, ptr addrspace(1) %256, align 2
  %257 = getelementptr inbounds i16, ptr addrspace(1) %arg67142, i64 %231
  store i16 %192, ptr addrspace(1) %257, align 2
  %258 = getelementptr inbounds i16, ptr addrspace(1) %arg68144, i64 %231
  store i16 %194, ptr addrspace(1) %258, align 2
  %259 = getelementptr inbounds i16, ptr addrspace(1) %arg69146, i64 %231
  store i16 %196, ptr addrspace(1) %259, align 2
  %260 = getelementptr inbounds i16, ptr addrspace(1) %arg70148, i64 %231
  store i16 %198, ptr addrspace(1) %260, align 2
  %261 = getelementptr inbounds i16, ptr addrspace(1) %arg71150, i64 %231
  store i16 %200, ptr addrspace(1) %261, align 2
  %262 = getelementptr inbounds i16, ptr addrspace(1) %arg72152, i64 %231
  store i16 %202, ptr addrspace(1) %262, align 2
  %263 = getelementptr inbounds i16, ptr addrspace(1) %arg73154, i64 %231
  store i16 %204, ptr addrspace(1) %263, align 2
  %264 = getelementptr inbounds i16, ptr addrspace(1) %arg74156, i64 %231
  store i16 %206, ptr addrspace(1) %264, align 2
  %265 = getelementptr inbounds i16, ptr addrspace(1) %arg75158, i64 %231
  store i16 %208, ptr addrspace(1) %265, align 2
  %266 = getelementptr inbounds i16, ptr addrspace(1) %arg76160, i64 %231
  store i16 %210, ptr addrspace(1) %266, align 2
  %267 = getelementptr inbounds i16, ptr addrspace(1) %arg77162, i64 %231
  store i16 %212, ptr addrspace(1) %267, align 2
  %268 = getelementptr inbounds i16, ptr addrspace(1) %arg78164, i64 %231
  store i16 %214, ptr addrspace(1) %268, align 2
  %269 = getelementptr inbounds i16, ptr addrspace(1) %arg79166, i64 %231
  store i16 %216, ptr addrspace(1) %269, align 2
  %270 = getelementptr inbounds i16, ptr addrspace(1) %arg80168, i64 %231
  store i16 %218, ptr addrspace(1) %270, align 2
  %271 = getelementptr inbounds i16, ptr addrspace(1) %arg81170, i64 %231
  store i16 %220, ptr addrspace(1) %271, align 2
  %272 = getelementptr inbounds i16, ptr addrspace(1) %arg82172, i64 %231
  store i16 %222, ptr addrspace(1) %272, align 2
  %273 = getelementptr inbounds i16, ptr addrspace(1) %arg83174, i64 %231
  store i16 %224, ptr addrspace(1) %273, align 2
  %274 = getelementptr inbounds i16, ptr addrspace(1) %arg84176, i64 %231
  store i16 %226, ptr addrspace(1) %274, align 2
  %275 = getelementptr inbounds i16, ptr addrspace(1) %arg85178, i64 %231
  store i16 %228, ptr addrspace(1) %275, align 2
  %276 = getelementptr inbounds i16, ptr addrspace(1) %arg86180, i64 %231
  store i16 %230, ptr addrspace(1) %276, align 2
  %277 = zext i32 %12 to i64
  %278 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg02, i64 0, i64 0, i64 %15, i64 %277
  %279 = load i16, ptr addrspace(1) %278, align 4
  %280 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg110, i64 0, i64 0, i64 %15, i64 %277
  %281 = load i16, ptr addrspace(1) %280, align 4
  %282 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg212, i64 0, i64 0, i64 %15, i64 %277
  %283 = load i16, ptr addrspace(1) %282, align 4
  %284 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg314, i64 0, i64 0, i64 %15, i64 %277
  %285 = load i16, ptr addrspace(1) %284, align 4
  %286 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg416, i64 0, i64 0, i64 %15, i64 %277
  %287 = load i16, ptr addrspace(1) %286, align 4
  %288 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg518, i64 0, i64 0, i64 %15, i64 %277
  %289 = load i16, ptr addrspace(1) %288, align 4
  %290 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg620, i64 0, i64 0, i64 %15, i64 %277
  %291 = load i16, ptr addrspace(1) %290, align 4
  %292 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg722, i64 0, i64 0, i64 %15, i64 %277
  %293 = load i16, ptr addrspace(1) %292, align 4
  %294 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg824, i64 0, i64 0, i64 %15, i64 %277
  %295 = load i16, ptr addrspace(1) %294, align 4
  %296 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg926, i64 0, i64 0, i64 %15, i64 %277
  %297 = load i16, ptr addrspace(1) %296, align 4
  %298 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1028, i64 0, i64 0, i64 %15, i64 %277
  %299 = load i16, ptr addrspace(1) %298, align 4
  %300 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1130, i64 0, i64 0, i64 %15, i64 %277
  %301 = load i16, ptr addrspace(1) %300, align 4
  %302 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1232, i64 0, i64 0, i64 %15, i64 %277
  %303 = load i16, ptr addrspace(1) %302, align 4
  %304 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1334, i64 0, i64 0, i64 %15, i64 %277
  %305 = load i16, ptr addrspace(1) %304, align 4
  %306 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1436, i64 0, i64 0, i64 %15, i64 %277
  %307 = load i16, ptr addrspace(1) %306, align 4
  %308 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1538, i64 0, i64 0, i64 %15, i64 %277
  %309 = load i16, ptr addrspace(1) %308, align 4
  %310 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1640, i64 0, i64 0, i64 %15, i64 %277
  %311 = load i16, ptr addrspace(1) %310, align 4
  %312 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1742, i64 0, i64 0, i64 %15, i64 %277
  %313 = load i16, ptr addrspace(1) %312, align 4
  %314 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1844, i64 0, i64 0, i64 %15, i64 %277
  %315 = load i16, ptr addrspace(1) %314, align 4
  %316 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1946, i64 0, i64 0, i64 %15, i64 %277
  %317 = load i16, ptr addrspace(1) %316, align 4
  %318 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2048, i64 0, i64 0, i64 %15, i64 %277
  %319 = load i16, ptr addrspace(1) %318, align 4
  %320 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2150, i64 0, i64 0, i64 %15, i64 %277
  %321 = load i16, ptr addrspace(1) %320, align 4
  %322 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2252, i64 0, i64 0, i64 %15, i64 %277
  %323 = load i16, ptr addrspace(1) %322, align 4
  %324 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2354, i64 0, i64 0, i64 %15, i64 %277
  %325 = load i16, ptr addrspace(1) %324, align 4
  %326 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2456, i64 0, i64 0, i64 %15, i64 %277
  %327 = load i16, ptr addrspace(1) %326, align 4
  %328 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2558, i64 0, i64 0, i64 %15, i64 %277
  %329 = load i16, ptr addrspace(1) %328, align 4
  %330 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2660, i64 0, i64 0, i64 %15, i64 %277
  %331 = load i16, ptr addrspace(1) %330, align 4
  %332 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2762, i64 0, i64 0, i64 %15, i64 %277
  %333 = load i16, ptr addrspace(1) %332, align 4
  %334 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2864, i64 0, i64 0, i64 %15, i64 %277
  %335 = load i16, ptr addrspace(1) %334, align 4
  %336 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2966, i64 0, i64 0, i64 %15, i64 %277
  %337 = load i16, ptr addrspace(1) %336, align 4
  %338 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3068, i64 0, i64 0, i64 %15, i64 %277
  %339 = load i16, ptr addrspace(1) %338, align 4
  %340 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3170, i64 0, i64 0, i64 %15, i64 %277
  %341 = load i16, ptr addrspace(1) %340, align 4
  %342 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3272, i64 0, i64 0, i64 %15, i64 %277
  %343 = load i16, ptr addrspace(1) %342, align 4
  %344 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3374, i64 0, i64 0, i64 %15, i64 %277
  %345 = load i16, ptr addrspace(1) %344, align 4
  %346 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3476, i64 0, i64 0, i64 %15, i64 %277
  %347 = load i16, ptr addrspace(1) %346, align 4
  %348 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3578, i64 0, i64 0, i64 %15, i64 %277
  %349 = load i16, ptr addrspace(1) %348, align 4
  %350 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3680, i64 0, i64 0, i64 %15, i64 %277
  %351 = load i16, ptr addrspace(1) %350, align 4
  %352 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3782, i64 0, i64 0, i64 %15, i64 %277
  %353 = load i16, ptr addrspace(1) %352, align 4
  %354 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3884, i64 0, i64 0, i64 %15, i64 %277
  %355 = load i16, ptr addrspace(1) %354, align 4
  %356 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3986, i64 0, i64 0, i64 %15, i64 %277
  %357 = load i16, ptr addrspace(1) %356, align 4
  %358 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4088, i64 0, i64 0, i64 %15, i64 %277
  %359 = load i16, ptr addrspace(1) %358, align 4
  %360 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4190, i64 0, i64 0, i64 %15, i64 %277
  %361 = load i16, ptr addrspace(1) %360, align 4
  %362 = zext i32 %linear_index2 to i64
  %363 = getelementptr inbounds i16, ptr addrspace(1) %arg4292, i64 %362
  store i16 %279, ptr addrspace(1) %363, align 4
  %364 = getelementptr inbounds i16, ptr addrspace(1) %arg4394, i64 %362
  store i16 %281, ptr addrspace(1) %364, align 4
  %365 = getelementptr inbounds i16, ptr addrspace(1) %arg4496, i64 %362
  store i16 %283, ptr addrspace(1) %365, align 4
  %366 = getelementptr inbounds i16, ptr addrspace(1) %arg4598, i64 %362
  store i16 %285, ptr addrspace(1) %366, align 4
  %367 = getelementptr inbounds i16, ptr addrspace(1) %arg46100, i64 %362
  store i16 %287, ptr addrspace(1) %367, align 4
  %368 = getelementptr inbounds i16, ptr addrspace(1) %arg47102, i64 %362
  store i16 %289, ptr addrspace(1) %368, align 4
  %369 = getelementptr inbounds i16, ptr addrspace(1) %arg48104, i64 %362
  store i16 %291, ptr addrspace(1) %369, align 4
  %370 = getelementptr inbounds i16, ptr addrspace(1) %arg49106, i64 %362
  store i16 %293, ptr addrspace(1) %370, align 4
  %371 = getelementptr inbounds i16, ptr addrspace(1) %arg50108, i64 %362
  store i16 %295, ptr addrspace(1) %371, align 4
  %372 = getelementptr inbounds i16, ptr addrspace(1) %arg51110, i64 %362
  store i16 %297, ptr addrspace(1) %372, align 4
  %373 = getelementptr inbounds i16, ptr addrspace(1) %arg52112, i64 %362
  store i16 %299, ptr addrspace(1) %373, align 4
  %374 = getelementptr inbounds i16, ptr addrspace(1) %arg53114, i64 %362
  store i16 %301, ptr addrspace(1) %374, align 4
  %375 = getelementptr inbounds i16, ptr addrspace(1) %arg54116, i64 %362
  store i16 %303, ptr addrspace(1) %375, align 4
  %376 = getelementptr inbounds i16, ptr addrspace(1) %arg55118, i64 %362
  store i16 %305, ptr addrspace(1) %376, align 4
  %377 = getelementptr inbounds i16, ptr addrspace(1) %arg56120, i64 %362
  store i16 %307, ptr addrspace(1) %377, align 4
  %378 = getelementptr inbounds i16, ptr addrspace(1) %arg57122, i64 %362
  store i16 %309, ptr addrspace(1) %378, align 4
  %379 = getelementptr inbounds i16, ptr addrspace(1) %arg58124, i64 %362
  store i16 %311, ptr addrspace(1) %379, align 4
  %380 = getelementptr inbounds i16, ptr addrspace(1) %arg59126, i64 %362
  store i16 %313, ptr addrspace(1) %380, align 4
  %381 = getelementptr inbounds i16, ptr addrspace(1) %arg60128, i64 %362
  store i16 %309, ptr addrspace(1) %381, align 4
  %382 = getelementptr inbounds i16, ptr addrspace(1) %arg61130, i64 %362
  store i16 %311, ptr addrspace(1) %382, align 4
  %383 = getelementptr inbounds i16, ptr addrspace(1) %arg62132, i64 %362
  store i16 %313, ptr addrspace(1) %383, align 4
  %384 = getelementptr inbounds i16, ptr addrspace(1) %arg63134, i64 %362
  store i16 %315, ptr addrspace(1) %384, align 4
  %385 = getelementptr inbounds i16, ptr addrspace(1) %arg64136, i64 %362
  store i16 %317, ptr addrspace(1) %385, align 4
  %386 = getelementptr inbounds i16, ptr addrspace(1) %arg65138, i64 %362
  store i16 %319, ptr addrspace(1) %386, align 4
  %387 = getelementptr inbounds i16, ptr addrspace(1) %arg66140, i64 %362
  store i16 %321, ptr addrspace(1) %387, align 4
  %388 = getelementptr inbounds i16, ptr addrspace(1) %arg67142, i64 %362
  store i16 %323, ptr addrspace(1) %388, align 4
  %389 = getelementptr inbounds i16, ptr addrspace(1) %arg68144, i64 %362
  store i16 %325, ptr addrspace(1) %389, align 4
  %390 = getelementptr inbounds i16, ptr addrspace(1) %arg69146, i64 %362
  store i16 %327, ptr addrspace(1) %390, align 4
  %391 = getelementptr inbounds i16, ptr addrspace(1) %arg70148, i64 %362
  store i16 %329, ptr addrspace(1) %391, align 4
  %392 = getelementptr inbounds i16, ptr addrspace(1) %arg71150, i64 %362
  store i16 %331, ptr addrspace(1) %392, align 4
  %393 = getelementptr inbounds i16, ptr addrspace(1) %arg72152, i64 %362
  store i16 %333, ptr addrspace(1) %393, align 4
  %394 = getelementptr inbounds i16, ptr addrspace(1) %arg73154, i64 %362
  store i16 %335, ptr addrspace(1) %394, align 4
  %395 = getelementptr inbounds i16, ptr addrspace(1) %arg74156, i64 %362
  store i16 %337, ptr addrspace(1) %395, align 4
  %396 = getelementptr inbounds i16, ptr addrspace(1) %arg75158, i64 %362
  store i16 %339, ptr addrspace(1) %396, align 4
  %397 = getelementptr inbounds i16, ptr addrspace(1) %arg76160, i64 %362
  store i16 %341, ptr addrspace(1) %397, align 4
  %398 = getelementptr inbounds i16, ptr addrspace(1) %arg77162, i64 %362
  store i16 %343, ptr addrspace(1) %398, align 4
  %399 = getelementptr inbounds i16, ptr addrspace(1) %arg78164, i64 %362
  store i16 %345, ptr addrspace(1) %399, align 4
  %400 = getelementptr inbounds i16, ptr addrspace(1) %arg79166, i64 %362
  store i16 %347, ptr addrspace(1) %400, align 4
  %401 = getelementptr inbounds i16, ptr addrspace(1) %arg80168, i64 %362
  store i16 %349, ptr addrspace(1) %401, align 4
  %402 = getelementptr inbounds i16, ptr addrspace(1) %arg81170, i64 %362
  store i16 %351, ptr addrspace(1) %402, align 4
  %403 = getelementptr inbounds i16, ptr addrspace(1) %arg82172, i64 %362
  store i16 %353, ptr addrspace(1) %403, align 4
  %404 = getelementptr inbounds i16, ptr addrspace(1) %arg83174, i64 %362
  store i16 %355, ptr addrspace(1) %404, align 4
  %405 = getelementptr inbounds i16, ptr addrspace(1) %arg84176, i64 %362
  store i16 %357, ptr addrspace(1) %405, align 4
  %406 = getelementptr inbounds i16, ptr addrspace(1) %arg85178, i64 %362
  store i16 %359, ptr addrspace(1) %406, align 4
  %407 = getelementptr inbounds i16, ptr addrspace(1) %arg86180, i64 %362
  store i16 %361, ptr addrspace(1) %407, align 4
  %408 = zext i32 %11 to i64
  %409 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg02, i64 0, i64 0, i64 %15, i64 %408
  %410 = load i16, ptr addrspace(1) %409, align 2
  %411 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg110, i64 0, i64 0, i64 %15, i64 %408
  %412 = load i16, ptr addrspace(1) %411, align 2
  %413 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg212, i64 0, i64 0, i64 %15, i64 %408
  %414 = load i16, ptr addrspace(1) %413, align 2
  %415 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg314, i64 0, i64 0, i64 %15, i64 %408
  %416 = load i16, ptr addrspace(1) %415, align 2
  %417 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg416, i64 0, i64 0, i64 %15, i64 %408
  %418 = load i16, ptr addrspace(1) %417, align 2
  %419 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg518, i64 0, i64 0, i64 %15, i64 %408
  %420 = load i16, ptr addrspace(1) %419, align 2
  %421 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg620, i64 0, i64 0, i64 %15, i64 %408
  %422 = load i16, ptr addrspace(1) %421, align 2
  %423 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg722, i64 0, i64 0, i64 %15, i64 %408
  %424 = load i16, ptr addrspace(1) %423, align 2
  %425 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg824, i64 0, i64 0, i64 %15, i64 %408
  %426 = load i16, ptr addrspace(1) %425, align 2
  %427 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg926, i64 0, i64 0, i64 %15, i64 %408
  %428 = load i16, ptr addrspace(1) %427, align 2
  %429 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1028, i64 0, i64 0, i64 %15, i64 %408
  %430 = load i16, ptr addrspace(1) %429, align 2
  %431 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1130, i64 0, i64 0, i64 %15, i64 %408
  %432 = load i16, ptr addrspace(1) %431, align 2
  %433 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1232, i64 0, i64 0, i64 %15, i64 %408
  %434 = load i16, ptr addrspace(1) %433, align 2
  %435 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1334, i64 0, i64 0, i64 %15, i64 %408
  %436 = load i16, ptr addrspace(1) %435, align 2
  %437 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1436, i64 0, i64 0, i64 %15, i64 %408
  %438 = load i16, ptr addrspace(1) %437, align 2
  %439 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1538, i64 0, i64 0, i64 %15, i64 %408
  %440 = load i16, ptr addrspace(1) %439, align 2
  %441 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1640, i64 0, i64 0, i64 %15, i64 %408
  %442 = load i16, ptr addrspace(1) %441, align 2
  %443 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1742, i64 0, i64 0, i64 %15, i64 %408
  %444 = load i16, ptr addrspace(1) %443, align 2
  %445 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1844, i64 0, i64 0, i64 %15, i64 %408
  %446 = load i16, ptr addrspace(1) %445, align 2
  %447 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg1946, i64 0, i64 0, i64 %15, i64 %408
  %448 = load i16, ptr addrspace(1) %447, align 2
  %449 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2048, i64 0, i64 0, i64 %15, i64 %408
  %450 = load i16, ptr addrspace(1) %449, align 2
  %451 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2150, i64 0, i64 0, i64 %15, i64 %408
  %452 = load i16, ptr addrspace(1) %451, align 2
  %453 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2252, i64 0, i64 0, i64 %15, i64 %408
  %454 = load i16, ptr addrspace(1) %453, align 2
  %455 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2354, i64 0, i64 0, i64 %15, i64 %408
  %456 = load i16, ptr addrspace(1) %455, align 2
  %457 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2456, i64 0, i64 0, i64 %15, i64 %408
  %458 = load i16, ptr addrspace(1) %457, align 2
  %459 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2558, i64 0, i64 0, i64 %15, i64 %408
  %460 = load i16, ptr addrspace(1) %459, align 2
  %461 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2660, i64 0, i64 0, i64 %15, i64 %408
  %462 = load i16, ptr addrspace(1) %461, align 2
  %463 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2762, i64 0, i64 0, i64 %15, i64 %408
  %464 = load i16, ptr addrspace(1) %463, align 2
  %465 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2864, i64 0, i64 0, i64 %15, i64 %408
  %466 = load i16, ptr addrspace(1) %465, align 2
  %467 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg2966, i64 0, i64 0, i64 %15, i64 %408
  %468 = load i16, ptr addrspace(1) %467, align 2
  %469 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3068, i64 0, i64 0, i64 %15, i64 %408
  %470 = load i16, ptr addrspace(1) %469, align 2
  %471 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3170, i64 0, i64 0, i64 %15, i64 %408
  %472 = load i16, ptr addrspace(1) %471, align 2
  %473 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3272, i64 0, i64 0, i64 %15, i64 %408
  %474 = load i16, ptr addrspace(1) %473, align 2
  %475 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3374, i64 0, i64 0, i64 %15, i64 %408
  %476 = load i16, ptr addrspace(1) %475, align 2
  %477 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3476, i64 0, i64 0, i64 %15, i64 %408
  %478 = load i16, ptr addrspace(1) %477, align 2
  %479 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3578, i64 0, i64 0, i64 %15, i64 %408
  %480 = load i16, ptr addrspace(1) %479, align 2
  %481 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3680, i64 0, i64 0, i64 %15, i64 %408
  %482 = load i16, ptr addrspace(1) %481, align 2
  %483 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3782, i64 0, i64 0, i64 %15, i64 %408
  %484 = load i16, ptr addrspace(1) %483, align 2
  %485 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3884, i64 0, i64 0, i64 %15, i64 %408
  %486 = load i16, ptr addrspace(1) %485, align 2
  %487 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg3986, i64 0, i64 0, i64 %15, i64 %408
  %488 = load i16, ptr addrspace(1) %487, align 2
  %489 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4088, i64 0, i64 0, i64 %15, i64 %408
  %490 = load i16, ptr addrspace(1) %489, align 2
  %491 = getelementptr inbounds [1 x [184320 x [64 x i16]]], ptr addrspace(1) %arg4190, i64 0, i64 0, i64 %15, i64 %408
  %492 = load i16, ptr addrspace(1) %491, align 2
  %493 = zext i32 %linear_index3 to i64
  %494 = getelementptr inbounds i16, ptr addrspace(1) %arg4292, i64 %493
  store i16 %410, ptr addrspace(1) %494, align 2
  %495 = getelementptr inbounds i16, ptr addrspace(1) %arg4394, i64 %493
  store i16 %412, ptr addrspace(1) %495, align 2
  %496 = getelementptr inbounds i16, ptr addrspace(1) %arg4496, i64 %493
  store i16 %414, ptr addrspace(1) %496, align 2
  %497 = getelementptr inbounds i16, ptr addrspace(1) %arg4598, i64 %493
  store i16 %416, ptr addrspace(1) %497, align 2
  %498 = getelementptr inbounds i16, ptr addrspace(1) %arg46100, i64 %493
  store i16 %418, ptr addrspace(1) %498, align 2
  %499 = getelementptr inbounds i16, ptr addrspace(1) %arg47102, i64 %493
  store i16 %420, ptr addrspace(1) %499, align 2
  %500 = getelementptr inbounds i16, ptr addrspace(1) %arg48104, i64 %493
  store i16 %422, ptr addrspace(1) %500, align 2
  %501 = getelementptr inbounds i16, ptr addrspace(1) %arg49106, i64 %493
  store i16 %424, ptr addrspace(1) %501, align 2
  %502 = getelementptr inbounds i16, ptr addrspace(1) %arg50108, i64 %493
  store i16 %426, ptr addrspace(1) %502, align 2
  %503 = getelementptr inbounds i16, ptr addrspace(1) %arg51110, i64 %493
  store i16 %428, ptr addrspace(1) %503, align 2
  %504 = getelementptr inbounds i16, ptr addrspace(1) %arg52112, i64 %493
  store i16 %430, ptr addrspace(1) %504, align 2
  %505 = getelementptr inbounds i16, ptr addrspace(1) %arg53114, i64 %493
  store i16 %432, ptr addrspace(1) %505, align 2
  %506 = getelementptr inbounds i16, ptr addrspace(1) %arg54116, i64 %493
  store i16 %434, ptr addrspace(1) %506, align 2
  %507 = getelementptr inbounds i16, ptr addrspace(1) %arg55118, i64 %493
  store i16 %436, ptr addrspace(1) %507, align 2
  %508 = getelementptr inbounds i16, ptr addrspace(1) %arg56120, i64 %493
  store i16 %438, ptr addrspace(1) %508, align 2
  %509 = getelementptr inbounds i16, ptr addrspace(1) %arg57122, i64 %493
  store i16 %440, ptr addrspace(1) %509, align 2
  %510 = getelementptr inbounds i16, ptr addrspace(1) %arg58124, i64 %493
  store i16 %442, ptr addrspace(1) %510, align 2
  %511 = getelementptr inbounds i16, ptr addrspace(1) %arg59126, i64 %493
  store i16 %444, ptr addrspace(1) %511, align 2
  %512 = getelementptr inbounds i16, ptr addrspace(1) %arg60128, i64 %493
  store i16 %440, ptr addrspace(1) %512, align 2
  %513 = getelementptr inbounds i16, ptr addrspace(1) %arg61130, i64 %493
  store i16 %442, ptr addrspace(1) %513, align 2
  %514 = getelementptr inbounds i16, ptr addrspace(1) %arg62132, i64 %493
  store i16 %444, ptr addrspace(1) %514, align 2
  %515 = getelementptr inbounds i16, ptr addrspace(1) %arg63134, i64 %493
  store i16 %446, ptr addrspace(1) %515, align 2
  %516 = getelementptr inbounds i16, ptr addrspace(1) %arg64136, i64 %493
  store i16 %448, ptr addrspace(1) %516, align 2
  %517 = getelementptr inbounds i16, ptr addrspace(1) %arg65138, i64 %493
  store i16 %450, ptr addrspace(1) %517, align 2
  %518 = getelementptr inbounds i16, ptr addrspace(1) %arg66140, i64 %493
  store i16 %452, ptr addrspace(1) %518, align 2
  %519 = getelementptr inbounds i16, ptr addrspace(1) %arg67142, i64 %493
  store i16 %454, ptr addrspace(1) %519, align 2
  %520 = getelementptr inbounds i16, ptr addrspace(1) %arg68144, i64 %493
  store i16 %456, ptr addrspace(1) %520, align 2
  %521 = getelementptr inbounds i16, ptr addrspace(1) %arg69146, i64 %493
  store i16 %458, ptr addrspace(1) %521, align 2
  %522 = getelementptr inbounds i16, ptr addrspace(1) %arg70148, i64 %493
  store i16 %460, ptr addrspace(1) %522, align 2
  %523 = getelementptr inbounds i16, ptr addrspace(1) %arg71150, i64 %493
  store i16 %462, ptr addrspace(1) %523, align 2
  %524 = getelementptr inbounds i16, ptr addrspace(1) %arg72152, i64 %493
  store i16 %464, ptr addrspace(1) %524, align 2
  %525 = getelementptr inbounds i16, ptr addrspace(1) %arg73154, i64 %493
  store i16 %466, ptr addrspace(1) %525, align 2
  %526 = getelementptr inbounds i16, ptr addrspace(1) %arg74156, i64 %493
  store i16 %468, ptr addrspace(1) %526, align 2
  %527 = getelementptr inbounds i16, ptr addrspace(1) %arg75158, i64 %493
  store i16 %470, ptr addrspace(1) %527, align 2
  %528 = getelementptr inbounds i16, ptr addrspace(1) %arg76160, i64 %493
  store i16 %472, ptr addrspace(1) %528, align 2
  %529 = getelementptr inbounds i16, ptr addrspace(1) %arg77162, i64 %493
  store i16 %474, ptr addrspace(1) %529, align 2
  %530 = getelementptr inbounds i16, ptr addrspace(1) %arg78164, i64 %493
  store i16 %476, ptr addrspace(1) %530, align 2
  %531 = getelementptr inbounds i16, ptr addrspace(1) %arg79166, i64 %493
  store i16 %478, ptr addrspace(1) %531, align 2
  %532 = getelementptr inbounds i16, ptr addrspace(1) %arg80168, i64 %493
  store i16 %480, ptr addrspace(1) %532, align 2
  %533 = getelementptr inbounds i16, ptr addrspace(1) %arg81170, i64 %493
  store i16 %482, ptr addrspace(1) %533, align 2
  %534 = getelementptr inbounds i16, ptr addrspace(1) %arg82172, i64 %493
  store i16 %484, ptr addrspace(1) %534, align 2
  %535 = getelementptr inbounds i16, ptr addrspace(1) %arg83174, i64 %493
  store i16 %486, ptr addrspace(1) %535, align 2
  %536 = getelementptr inbounds i16, ptr addrspace(1) %arg84176, i64 %493
  store i16 %488, ptr addrspace(1) %536, align 2
  %537 = getelementptr inbounds i16, ptr addrspace(1) %arg85178, i64 %493
  store i16 %490, ptr addrspace(1) %537, align 2
  %538 = getelementptr inbounds i16, ptr addrspace(1) %arg86180, i64 %493
  store i16 %492, ptr addrspace(1) %538, align 2
  br label %copy_fusion_1.in_bounds-after

; CHECK-LABEL: @copy_fusion_1
; CHECK-DAG: store <4 x i16>
}
