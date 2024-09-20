%struct.rocfft_complex = type { half, half }

$_Z32real_post_process_kernel_inplaceI14rocfft_complexIDF16_ELb1EEvmmmPT_mPKS2_ = comdat any

; Function Attrs: convergent inlinehint mustprogress nounwind
define weak_odr hidden void @_Z32real_post_process_kernel_inplaceI14rocfft_complexIDF16_ELb1EEvmmmPT_mPKS2_(i64 noundef %0, i64 noundef %1, i64 noundef %2, ptr noundef %3, i64 noundef %4, ptr noundef %5) #2 comdat {
  %7 = alloca i64, align 8, addrspace(5)
  %8 = alloca i64, align 8, addrspace(5)
  %9 = alloca i64, align 8, addrspace(5)
  %10 = alloca ptr, align 8, addrspace(5)
  %11 = alloca i64, align 8, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %14 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %15 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %16 = alloca double, align 8, addrspace(5)
  %17 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %18 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %19 = alloca double, align 8, addrspace(5)
  %20 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %21 = alloca %struct.rocfft_complex, align 2, addrspace(5)
  %22 = addrspacecast ptr addrspace(5) %7 to ptr
  %23 = addrspacecast ptr addrspace(5) %8 to ptr
  %24 = addrspacecast ptr addrspace(5) %9 to ptr
  %25 = addrspacecast ptr addrspace(5) %10 to ptr
  %26 = addrspacecast ptr addrspace(5) %11 to ptr
  %27 = addrspacecast ptr addrspace(5) %12 to ptr
  %28 = addrspacecast ptr addrspace(5) %13 to ptr
  %29 = addrspacecast ptr addrspace(5) %14 to ptr
  %30 = addrspacecast ptr addrspace(5) %15 to ptr
  %31 = addrspacecast ptr addrspace(5) %16 to ptr
  %32 = addrspacecast ptr addrspace(5) %17 to ptr
  %33 = addrspacecast ptr addrspace(5) %18 to ptr
  %34 = addrspacecast ptr addrspace(5) %19 to ptr
  %35 = addrspacecast ptr addrspace(5) %20 to ptr
  %36 = addrspacecast ptr addrspace(5) %21 to ptr
  store i64 %0, ptr %22, align 8, !tbaa !6
  store i64 %1, ptr %23, align 8, !tbaa !6
  store i64 %2, ptr %24, align 8, !tbaa !6
  store ptr %3, ptr %25, align 8, !tbaa !10
  store i64 %4, ptr %26, align 8, !tbaa !6
  store ptr %5, ptr %27, align 8, !tbaa !10
  %37 = load i64, ptr %22, align 8, !tbaa !6
  %38 = load i64, ptr %24, align 8, !tbaa !6
  br label %40

40:                                               ; preds = %6
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %13) #4
  %41 = load ptr, ptr %25, align 8, !tbaa !10
  %42 = load i64, ptr %26, align 8, !tbaa !6
  %43 = load i64, ptr %22, align 8, !tbaa !6
  %44 = add i64 %42, %43
  %45 = getelementptr inbounds %struct.rocfft_complex, ptr %41, i64 %44
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %28, ptr align 2 %45, i64 4, i1 false), !tbaa.struct !12
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %14) #4
  %46 = load ptr, ptr %25, align 8, !tbaa !10
  %47 = load i64, ptr %26, align 8, !tbaa !6
  %48 = load i64, ptr %23, align 8, !tbaa !6
  %49 = add i64 %47, %48
  %50 = getelementptr inbounds %struct.rocfft_complex, ptr %46, i64 %49
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %29, ptr align 2 %50, i64 4, i1 false), !tbaa.struct !12
  %51 = load i64, ptr %22, align 8, !tbaa !6
  %52 = icmp eq i64 %51, 0
  br i1 %52, label %53, label %102

53:                                               ; preds = %40
  %54 = getelementptr inbounds %struct.rocfft_complex, ptr %28, i32 0, i32 0
  %55 = load half, ptr %54, align 2, !tbaa !15
  %56 = getelementptr inbounds %struct.rocfft_complex, ptr %28, i32 0, i32 1
  %57 = load half, ptr %56, align 2, !tbaa !17
  %58 = fadd contract half %55, %57
  %59 = load ptr, ptr %25, align 8, !tbaa !10
  %60 = load i64, ptr %26, align 8, !tbaa !6
  %61 = load i64, ptr %22, align 8, !tbaa !6
  %62 = add i64 %60, %61
  %63 = getelementptr inbounds %struct.rocfft_complex, ptr %59, i64 %62
  %64 = getelementptr inbounds %struct.rocfft_complex, ptr %63, i32 0, i32 0
  store half %58, ptr %64, align 2, !tbaa !15
  %65 = load ptr, ptr %25, align 8, !tbaa !10
  %66 = load i64, ptr %26, align 8, !tbaa !6
  %67 = load i64, ptr %22, align 8, !tbaa !6
  %68 = add i64 %66, %67
  %69 = getelementptr inbounds %struct.rocfft_complex, ptr %65, i64 %68
  %70 = getelementptr inbounds %struct.rocfft_complex, ptr %69, i32 0, i32 1
  store half 0xH0000, ptr %70, align 2, !tbaa !17
  %71 = getelementptr inbounds %struct.rocfft_complex, ptr %28, i32 0, i32 0
  %72 = load half, ptr %71, align 2, !tbaa !15
  %73 = getelementptr inbounds %struct.rocfft_complex, ptr %28, i32 0, i32 1
  %74 = load half, ptr %73, align 2, !tbaa !17
  %75 = fsub contract half %72, %74
  %76 = load ptr, ptr %25, align 8, !tbaa !10
  %77 = load i64, ptr %26, align 8, !tbaa !6
  %78 = load i64, ptr %23, align 8, !tbaa !6
  %79 = add i64 %77, %78
  %80 = getelementptr inbounds %struct.rocfft_complex, ptr %76, i64 %79
  %81 = getelementptr inbounds %struct.rocfft_complex, ptr %80, i32 0, i32 0
  store half %75, ptr %81, align 2, !tbaa !15
  %82 = load ptr, ptr %25, align 8, !tbaa !10
  %83 = load i64, ptr %26, align 8, !tbaa !6
  %84 = load i64, ptr %23, align 8, !tbaa !6
  %85 = add i64 %83, %84
  %86 = getelementptr inbounds %struct.rocfft_complex, ptr %82, i64 %85
  %87 = getelementptr inbounds %struct.rocfft_complex, ptr %86, i32 0, i32 1
  store half 0xH0000, ptr %87, align 2, !tbaa !17
  %88 = load ptr, ptr %25, align 8, !tbaa !10
  %89 = load i64, ptr %26, align 8, !tbaa !6
  %90 = load i64, ptr %24, align 8, !tbaa !6
  %91 = add i64 %89, %90
  %92 = getelementptr inbounds %struct.rocfft_complex, ptr %88, i64 %91
  %93 = getelementptr inbounds %struct.rocfft_complex, ptr %92, i32 0, i32 1
  %94 = load half, ptr %93, align 2, !tbaa !17
  %95 = fneg contract half %94
  %96 = load ptr, ptr %25, align 8, !tbaa !10
  %97 = load i64, ptr %26, align 8, !tbaa !6
  %98 = load i64, ptr %24, align 8, !tbaa !6
  %99 = add i64 %97, %98
  %100 = getelementptr inbounds %struct.rocfft_complex, ptr %96, i64 %99
  %101 = getelementptr inbounds %struct.rocfft_complex, ptr %100, i32 0, i32 1
  store half %95, ptr %101, align 2, !tbaa !17
  ret void

102:                                              ; preds = %40
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %15) #4
  call void @llvm.lifetime.start.p5(i64 8, ptr addrspace(5) %16) #4
  store double 5.000000e-01, ptr %31, align 8, !tbaa !18
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %17) #4
  store i32 0, ptr %32, align 2
  store i32 0, ptr %30, align 2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %17) #4
  call void @llvm.lifetime.end.p5(i64 8, ptr addrspace(5) %16) #4
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %18) #4
  call void @llvm.lifetime.start.p5(i64 8, ptr addrspace(5) %19) #4
  store double 5.000000e-01, ptr %34, align 8, !tbaa !18
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %20) #4
  store i32 0, ptr %35, align 2
  store i32 0, ptr %33, align 2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %20) #4
  call void @llvm.lifetime.end.p5(i64 8, ptr addrspace(5) %19) #4
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %21) #4
  %107 = load ptr, ptr %27, align 8, !tbaa !10
  %108 = load i64, ptr %22, align 8, !tbaa !6
  %109 = getelementptr inbounds %struct.rocfft_complex, ptr %107, i64 %108
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %36, ptr align 2 %109, i64 4, i1 false), !tbaa.struct !12
  %110 = getelementptr inbounds %struct.rocfft_complex, ptr %30, i32 0, i32 0
  %111 = load half, ptr %110, align 2, !tbaa !15
  %112 = getelementptr inbounds %struct.rocfft_complex, ptr %33, i32 0, i32 0
  %113 = load half, ptr %112, align 2, !tbaa !15
  %114 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 1
  %115 = load half, ptr %114, align 2, !tbaa !17
  %116 = fmul contract half %113, %115
  %117 = fadd contract half %111, %116
  %118 = getelementptr inbounds %struct.rocfft_complex, ptr %30, i32 0, i32 1
  %119 = load half, ptr %118, align 2, !tbaa !17
  %120 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 0
  %121 = load half, ptr %120, align 2, !tbaa !15
  %122 = fmul contract half %119, %121
  %123 = fadd contract half %117, %122
  %124 = load ptr, ptr %25, align 8, !tbaa !10
  %125 = load i64, ptr %26, align 8, !tbaa !6
  %126 = load i64, ptr %22, align 8, !tbaa !6
  %127 = add i64 %125, %126
  %128 = getelementptr inbounds %struct.rocfft_complex, ptr %124, i64 %127
  %129 = getelementptr inbounds %struct.rocfft_complex, ptr %128, i32 0, i32 0
  store half %123, ptr %129, align 2, !tbaa !15
  %130 = getelementptr inbounds %struct.rocfft_complex, ptr %33, i32 0, i32 1
  %131 = load half, ptr %130, align 2, !tbaa !17
  %132 = getelementptr inbounds %struct.rocfft_complex, ptr %30, i32 0, i32 1
  %133 = load half, ptr %132, align 2, !tbaa !17
  %134 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 1
  %135 = load half, ptr %134, align 2, !tbaa !17
  %136 = fmul contract half %133, %135
  %137 = fadd contract half %131, %136
  %138 = getelementptr inbounds %struct.rocfft_complex, ptr %33, i32 0, i32 0
  %139 = load half, ptr %138, align 2, !tbaa !15
  %140 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 0
  %141 = load half, ptr %140, align 2, !tbaa !15
  %142 = fmul contract half %139, %141
  %143 = fsub contract half %137, %142
  %144 = load ptr, ptr %25, align 8, !tbaa !10
  %145 = load i64, ptr %26, align 8, !tbaa !6
  %146 = load i64, ptr %22, align 8, !tbaa !6
  %147 = add i64 %145, %146
  %148 = getelementptr inbounds %struct.rocfft_complex, ptr %144, i64 %147
  %149 = getelementptr inbounds %struct.rocfft_complex, ptr %148, i32 0, i32 1
  store half %143, ptr %149, align 2, !tbaa !17
  %150 = getelementptr inbounds %struct.rocfft_complex, ptr %30, i32 0, i32 0
  %151 = load half, ptr %150, align 2, !tbaa !15
  %152 = getelementptr inbounds %struct.rocfft_complex, ptr %33, i32 0, i32 0
  %153 = load half, ptr %152, align 2, !tbaa !15
  %154 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 1
  %155 = load half, ptr %154, align 2, !tbaa !17
  %156 = fmul contract half %153, %155
  %157 = fsub contract half %151, %156
  %158 = getelementptr inbounds %struct.rocfft_complex, ptr %30, i32 0, i32 1
  %159 = load half, ptr %158, align 2, !tbaa !17
  %160 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 0
  %161 = load half, ptr %160, align 2, !tbaa !15
  %162 = fmul contract half %159, %161
  %163 = fsub contract half %157, %162
  %164 = load ptr, ptr %25, align 8, !tbaa !10
  %165 = load i64, ptr %26, align 8, !tbaa !6
  %166 = load i64, ptr %23, align 8, !tbaa !6
  %167 = add i64 %165, %166
  %168 = getelementptr inbounds %struct.rocfft_complex, ptr %164, i64 %167
  %169 = getelementptr inbounds %struct.rocfft_complex, ptr %168, i32 0, i32 0
  store half %163, ptr %169, align 2, !tbaa !15
  %170 = getelementptr inbounds %struct.rocfft_complex, ptr %33, i32 0, i32 1
  %171 = load half, ptr %170, align 2, !tbaa !17
  %172 = fneg contract half %171
  %173 = getelementptr inbounds %struct.rocfft_complex, ptr %30, i32 0, i32 1
  %174 = load half, ptr %173, align 2, !tbaa !17
  %175 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 1
  %176 = load half, ptr %175, align 2, !tbaa !17
  %177 = fmul contract half %174, %176
  %178 = fadd contract half %172, %177
  %179 = getelementptr inbounds %struct.rocfft_complex, ptr %33, i32 0, i32 0
  %180 = load half, ptr %179, align 2, !tbaa !15
  %181 = getelementptr inbounds %struct.rocfft_complex, ptr %36, i32 0, i32 0
  %182 = load half, ptr %181, align 2, !tbaa !15
  %183 = fmul contract half %180, %182
  %184 = fsub contract half %178, %183
  %185 = load ptr, ptr %25, align 8, !tbaa !10
  %186 = load i64, ptr %26, align 8, !tbaa !6
  %187 = load i64, ptr %23, align 8, !tbaa !6
  %188 = add i64 %186, %187
  %189 = getelementptr inbounds %struct.rocfft_complex, ptr %185, i64 %188
  %190 = getelementptr inbounds %struct.rocfft_complex, ptr %189, i32 0, i32 1
  store half %184, ptr %190, align 2, !tbaa !17
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %21) #4
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %18) #4
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %15) #4
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { convergent inlinehint mustprogress nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+cumode,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+sramecc,+wavefrontsize64,-xnack" }
attributes #3 = { convergent mustprogress nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+cumode,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+sramecc,+wavefrontsize64,-xnack" }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4}
!opencl.ocl.version = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{!"clang version 19.0.0git (ssh://padivedi@gerrit-git.amd.com:29418/lightning/ec/llvm-project a2421f3d00e8e99003ddde4ce19939737b57d043)"}
!5 = !{i32 2, i32 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"any pointer", !8, i64 0}
!12 = !{i64 0, i64 2, !13, i64 2, i64 2, !13}
!13 = !{!14, !14, i64 0}
!14 = !{!"_Float16", !8, i64 0}
!15 = !{!16, !14, i64 0}
!16 = !{!"_ZTS14rocfft_complexIDF16_E", !14, i64 0, !14, i64 2}
!17 = !{!16, !14, i64 2}
!18 = !{!19, !19, i64 0}
!19 = !{!"double", !8, i64 0}
