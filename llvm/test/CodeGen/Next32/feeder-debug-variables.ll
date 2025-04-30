; RUN: llc -mtriple=next32 < %s | FileCheck %s --check-prefix=CHECK

; Original C source:
; #include <math.h>
; 
; void Step10_orig( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, float *xx1, float *yy1, float *zz1, float *mass1, float *dxi, float *dyi, float *dzi )
; {
; 
;     const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
;     
;     float dxc, dyc, dzc, m, r2, f, xi, yi, zi;
;     float s0, s1, s2;
;     int j;
; 
;     xi = 0.; yi = 0.; zi = 0.;
; 
;     for ( j = 0; j < count1; j++ ) 
;     {
;         dxc = xx1[j] - xxi;
;         dyc = yy1[j] - yyi;
;         dzc = zz1[j] - zzi;
;   
;         r2 = dxc * dxc + dyc * dyc + dzc * dzc;
;        
;         m = ( r2 < fsrrmax2 ) ? mass1[j] : 0.0f;
; 
;         s0 = r2 + mp_rsm2;
;         s1 = s0 * s0 * s0;
;         s2 = 1.0f / sqrtf( s1 ) - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));
; 
;         f = ( r2 > 0.0f ) ? m * s2 : 0.0f;
; 
;         xi = xi + f * dxc;
;         yi = yi + f * dyc;
;         zi = zi + f * dzc;
;     }
; 
;     *dxi = xi;
;     *dyi = yi;
;     *dzi = zi;
; }

; Function Attrs: noinline nounwind optnone
define dso_local void @Step10_orig(i32 %0, float %1, float %2, float %3, float %4, float %5, float* %6, float* %7, float* %8, float* %9, float* %10, float* %11, float* %12) #0 !dbg !7 {
; CHECK: 			#DEBUG_VALUE: Step10_orig:count1 <- $r1
; CHECK-NEXT: feeder.32	r1
; CHECK:			#DEBUG_VALUE: Step10_orig:xxi <- $r2
; CHECK-NEXT:	feeder.32	r2
; CHECK:			#DEBUG_VALUE: Step10_orig:yyi <- $r3
; CHECK-NEXT:	feeder.32	r3
; CHECK:			#DEBUG_VALUE: Step10_orig:zzi <- $r4
; CHECK-NEXT:	feeder.32	r4
; CHECK:			#DEBUG_VALUE: Step10_orig:fsrrmax2 <- $r5
; CHECK-NEXT:	feeder.32	r5
; CHECK:			#DEBUG_VALUE: Step10_orig:mp_rsm2 <- $r6
; CHECK-NEXT:	feeder.32	r6
; CHECK:			#DEBUG_VALUE: Step10_orig:xx1 <- [DW_OP_LLVM_fragment 0 32] $r7
; CHECK-NEXT:	feeder.64	r7
; CHECK:			#DEBUG_VALUE: Step10_orig:xx1 <- [DW_OP_LLVM_fragment 32 32] $r8
; CHECK-NEXT:	feeder.64	r8
; CHECK:			#DEBUG_VALUE: Step10_orig:yy1 <- [DW_OP_LLVM_fragment 0 32] $r9
; CHECK-NEXT:	feeder.64	r9
; CHECK:			#DEBUG_VALUE: Step10_orig:yy1 <- [DW_OP_LLVM_fragment 32 32] $r10
; CHECK-NEXT:	feeder.64	r10
; CHECK:			#DEBUG_VALUE: Step10_orig:zz1 <- [DW_OP_LLVM_fragment 0 32] $r11
; CHECK-NEXT:	feeder.64	r11
; CHECK:			#DEBUG_VALUE: Step10_orig:zz1 <- [DW_OP_LLVM_fragment 32 32] $r12
; CHECK-NEXT:	feeder.64	r12
; CHECK:			#DEBUG_VALUE: Step10_orig:mass1 <- [DW_OP_LLVM_fragment 0 32] $r13
; CHECK-NEXT:	feeder.64	r13
; CHECK:	#DEBUG_VALUE: Step10_orig:mass1 <- [DW_OP_LLVM_fragment 32 32] $r14
; CHECK-NEXT:	feeder.64	r14
; CHECK:	#DEBUG_VALUE: Step10_orig:dxi <- [DW_OP_LLVM_fragment 0 32] $r15
; CHECK-NEXT:	feeder.64	r15
; CHECK:	#DEBUG_VALUE: Step10_orig:dxi <- [DW_OP_LLVM_fragment 32 32] $r16
; CHECK-NEXT:	feeder.64	r16
; CHECK:	#DEBUG_VALUE: Step10_orig:dyi <- [DW_OP_LLVM_fragment 0 32] $r17
; CHECK-NEXT:	feeder.64	r17
; CHECK:	#DEBUG_VALUE: Step10_orig:dyi <- [DW_OP_LLVM_fragment 32 32] $r18
; CHECK-NEXT:	feeder.64	r18
; CHECK:	#DEBUG_VALUE: Step10_orig:dzi <- [DW_OP_LLVM_fragment 0 32] $r19
; CHECK-NEXT:	feeder.64	r19
; CHECK:	#DEBUG_VALUE: Step10_orig:dzi <- [DW_OP_LLVM_fragment 32 32] $r20
; CHECK-NEXT:	feeder.64	r20
  %14 = alloca i32, align 4
  %15 = alloca float, align 4
  %16 = alloca float, align 4
  %17 = alloca float, align 4
  %18 = alloca float, align 4
  %19 = alloca float, align 4
  %20 = alloca float*, align 8
  %21 = alloca float*, align 8
  %22 = alloca float*, align 8
  %23 = alloca float*, align 8
  %24 = alloca float*, align 8
  %25 = alloca float*, align 8
  %26 = alloca float*, align 8
  %27 = alloca float, align 4
  %28 = alloca float, align 4
  %29 = alloca float, align 4
  %30 = alloca float, align 4
  %31 = alloca float, align 4
  %32 = alloca float, align 4
  %33 = alloca float, align 4
  %34 = alloca float, align 4
  %35 = alloca float, align 4
  %36 = alloca float, align 4
  %37 = alloca float, align 4
  %38 = alloca float, align 4
  %39 = alloca float, align 4
  %40 = alloca float, align 4
  %41 = alloca float, align 4
  %42 = alloca float, align 4
  %43 = alloca float, align 4
  %44 = alloca float, align 4
  %45 = alloca i32, align 4
  store i32 %0, i32* %14, align 4
  call void @llvm.dbg.declare(metadata i32* %14, metadata !13, metadata !DIExpression()), !dbg !14
  store float %1, float* %15, align 4
  call void @llvm.dbg.declare(metadata float* %15, metadata !15, metadata !DIExpression()), !dbg !16
  store float %2, float* %16, align 4
  call void @llvm.dbg.declare(metadata float* %16, metadata !17, metadata !DIExpression()), !dbg !18
  store float %3, float* %17, align 4
  call void @llvm.dbg.declare(metadata float* %17, metadata !19, metadata !DIExpression()), !dbg !20
  store float %4, float* %18, align 4
  call void @llvm.dbg.declare(metadata float* %18, metadata !21, metadata !DIExpression()), !dbg !22
  store float %5, float* %19, align 4
  call void @llvm.dbg.declare(metadata float* %19, metadata !23, metadata !DIExpression()), !dbg !24
  store float* %6, float** %20, align 8
  call void @llvm.dbg.declare(metadata float** %20, metadata !25, metadata !DIExpression()), !dbg !26
  store float* %7, float** %21, align 8
  call void @llvm.dbg.declare(metadata float** %21, metadata !27, metadata !DIExpression()), !dbg !28
  store float* %8, float** %22, align 8
  call void @llvm.dbg.declare(metadata float** %22, metadata !29, metadata !DIExpression()), !dbg !30
  store float* %9, float** %23, align 8
  call void @llvm.dbg.declare(metadata float** %23, metadata !31, metadata !DIExpression()), !dbg !32
  store float* %10, float** %24, align 8
  call void @llvm.dbg.declare(metadata float** %24, metadata !33, metadata !DIExpression()), !dbg !34
  store float* %11, float** %25, align 8
  call void @llvm.dbg.declare(metadata float** %25, metadata !35, metadata !DIExpression()), !dbg !36
  store float* %12, float** %26, align 8
  call void @llvm.dbg.declare(metadata float** %26, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata float* %27, metadata !39, metadata !DIExpression()), !dbg !41
  store float 0x3FD13CA760000000, float* %27, align 4, !dbg !41
  call void @llvm.dbg.declare(metadata float* %28, metadata !42, metadata !DIExpression()), !dbg !43
  store float 0xBFB3399C00000000, float* %28, align 4, !dbg !43
  call void @llvm.dbg.declare(metadata float* %29, metadata !44, metadata !DIExpression()), !dbg !45
  store float 0x3F87833EE0000000, float* %29, align 4, !dbg !45
  call void @llvm.dbg.declare(metadata float* %30, metadata !46, metadata !DIExpression()), !dbg !47
  store float 0xBF51E8EB60000000, float* %30, align 4, !dbg !47
  call void @llvm.dbg.declare(metadata float* %31, metadata !48, metadata !DIExpression()), !dbg !49
  store float 0x3F0FBEC340000000, float* %31, align 4, !dbg !49
  call void @llvm.dbg.declare(metadata float* %32, metadata !50, metadata !DIExpression()), !dbg !51
  store float 0xBEB8B13440000000, float* %32, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata float* %33, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata float* %34, metadata !54, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.declare(metadata float* %35, metadata !56, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata float* %36, metadata !58, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.declare(metadata float* %37, metadata !60, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.declare(metadata float* %38, metadata !62, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata float* %39, metadata !64, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.declare(metadata float* %40, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata float* %41, metadata !68, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.declare(metadata float* %42, metadata !70, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.declare(metadata float* %43, metadata !72, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata float* %44, metadata !74, metadata !DIExpression()), !dbg !75
  call void @llvm.dbg.declare(metadata i32* %45, metadata !76, metadata !DIExpression()), !dbg !77
  store float 0.000000e+00, float* %39, align 4, !dbg !78
  store float 0.000000e+00, float* %40, align 4, !dbg !79
  store float 0.000000e+00, float* %41, align 4, !dbg !80
  store i32 0, i32* %45, align 4, !dbg !81
  br label %46, !dbg !83

46:                                               ; preds = %146, %13
; CHECK:	#DEBUG_VALUE: Step10_orig:xi <- $r21
; CHECK-NEXT:	.loc	1 0 0                           # Step10_orig.c:0:0
; CHECK-NEXT:	feeder.32	r21
; CHECK:	#DEBUG_VALUE: Step10_orig:zi <- $r22
; CHECK-NEXT:	feeder.32	r22
; CHECK:	#DEBUG_VALUE: Step10_orig:yi <- $r23
; CHECK-NEXT:	feeder.32	r23
; CHECK:	#DEBUG_VALUE: Step10_orig:j <- $r24
; CHECK-NEXT:	feeder.32	r24
  %47 = load i32, i32* %45, align 4, !dbg !84
  %48 = load i32, i32* %14, align 4, !dbg !86
  %49 = icmp slt i32 %47, %48, !dbg !87
  br i1 %49, label %50, label %149, !dbg !88

50:                                               ; preds = %46
  %51 = load float*, float** %20, align 8, !dbg !89
  %52 = load i32, i32* %45, align 4, !dbg !91
  %53 = sext i32 %52 to i64, !dbg !89
  %54 = getelementptr inbounds float, float* %51, i64 %53, !dbg !89
  %55 = load float, float* %54, align 4, !dbg !89
  %56 = load float, float* %15, align 4, !dbg !92
  %57 = fsub float %55, %56, !dbg !93
  store float %57, float* %33, align 4, !dbg !94
  %58 = load float*, float** %21, align 8, !dbg !95
  %59 = load i32, i32* %45, align 4, !dbg !96
  %60 = sext i32 %59 to i64, !dbg !95
  %61 = getelementptr inbounds float, float* %58, i64 %60, !dbg !95
  %62 = load float, float* %61, align 4, !dbg !95
  %63 = load float, float* %16, align 4, !dbg !97
  %64 = fsub float %62, %63, !dbg !98
  store float %64, float* %34, align 4, !dbg !99
  %65 = load float*, float** %22, align 8, !dbg !100
  %66 = load i32, i32* %45, align 4, !dbg !101
  %67 = sext i32 %66 to i64, !dbg !100
  %68 = getelementptr inbounds float, float* %65, i64 %67, !dbg !100
  %69 = load float, float* %68, align 4, !dbg !100
  %70 = load float, float* %17, align 4, !dbg !102
  %71 = fsub float %69, %70, !dbg !103
  store float %71, float* %35, align 4, !dbg !104
  %72 = load float, float* %33, align 4, !dbg !105
  %73 = load float, float* %33, align 4, !dbg !106
  %74 = fmul float %72, %73, !dbg !107
  %75 = load float, float* %34, align 4, !dbg !108
  %76 = load float, float* %34, align 4, !dbg !109
  %77 = fmul float %75, %76, !dbg !110
  %78 = fadd float %74, %77, !dbg !111
  %79 = load float, float* %35, align 4, !dbg !112
  %80 = load float, float* %35, align 4, !dbg !113
  %81 = fmul float %79, %80, !dbg !114
  %82 = fadd float %78, %81, !dbg !115
  store float %82, float* %37, align 4, !dbg !116
  %83 = load float, float* %37, align 4, !dbg !117
  %84 = load float, float* %18, align 4, !dbg !118
  %85 = fcmp olt float %83, %84, !dbg !119
  br i1 %85, label %86, label %92, !dbg !120

86:                                               ; preds = %50
  %87 = load float*, float** %23, align 8, !dbg !121
  %88 = load i32, i32* %45, align 4, !dbg !122
  %89 = sext i32 %88 to i64, !dbg !121
  %90 = getelementptr inbounds float, float* %87, i64 %89, !dbg !121
  %91 = load float, float* %90, align 4, !dbg !121
  br label %93, !dbg !120

92:                                               ; preds = %50
  br label %93, !dbg !120

93:                                               ; preds = %92, %86
  %94 = phi float [ %91, %86 ], [ 0.000000e+00, %92 ], !dbg !120
  store float %94, float* %36, align 4, !dbg !123
  %95 = load float, float* %37, align 4, !dbg !124
  %96 = load float, float* %19, align 4, !dbg !125
  %97 = fadd float %95, %96, !dbg !126
  store float %97, float* %42, align 4, !dbg !127
  %98 = load float, float* %42, align 4, !dbg !128
  %99 = load float, float* %42, align 4, !dbg !129
  %100 = fmul float %98, %99, !dbg !130
  %101 = load float, float* %42, align 4, !dbg !131
  %102 = fmul float %100, %101, !dbg !132
  store float %102, float* %43, align 4, !dbg !133
  %103 = load float, float* %43, align 4, !dbg !134
  %104 = call float @sqrtf(float %103) #3, !dbg !135
  %105 = fdiv float 1.000000e+00, %104, !dbg !136
  %106 = load float, float* %37, align 4, !dbg !137
  %107 = load float, float* %37, align 4, !dbg !138
  %108 = load float, float* %37, align 4, !dbg !139
  %109 = load float, float* %37, align 4, !dbg !140
  %110 = load float, float* %37, align 4, !dbg !141
  %111 = fmul float %110, 0xBEB8B13440000000, !dbg !142
  %112 = fadd float 0x3F0FBEC340000000, %111, !dbg !143
  %113 = fmul float %109, %112, !dbg !144
  %114 = fadd float 0xBF51E8EB60000000, %113, !dbg !145
  %115 = fmul float %108, %114, !dbg !146
  %116 = fadd float 0x3F87833EE0000000, %115, !dbg !147
  %117 = fmul float %107, %116, !dbg !148
  %118 = fadd float 0xBFB3399C00000000, %117, !dbg !149
  %119 = fmul float %106, %118, !dbg !150
  %120 = fadd float 0x3FD13CA760000000, %119, !dbg !151
  %121 = fsub float %105, %120, !dbg !152
  store float %121, float* %44, align 4, !dbg !153
  %122 = load float, float* %37, align 4, !dbg !154
  %123 = fcmp ogt float %122, 0.000000e+00, !dbg !155
  br i1 %123, label %124, label %128, !dbg !156

124:                                              ; preds = %93
  %125 = load float, float* %36, align 4, !dbg !157
  %126 = load float, float* %44, align 4, !dbg !158
  %127 = fmul float %125, %126, !dbg !159
  br label %129, !dbg !156

128:                                              ; preds = %93
  br label %129, !dbg !156

129:                                              ; preds = %128, %124
  %130 = phi float [ %127, %124 ], [ 0.000000e+00, %128 ], !dbg !156
  store float %130, float* %38, align 4, !dbg !160
  %131 = load float, float* %39, align 4, !dbg !161
  %132 = load float, float* %38, align 4, !dbg !162
  %133 = load float, float* %33, align 4, !dbg !163
  %134 = fmul float %132, %133, !dbg !164
  %135 = fadd float %131, %134, !dbg !165
  store float %135, float* %39, align 4, !dbg !166
  %136 = load float, float* %40, align 4, !dbg !167
  %137 = load float, float* %38, align 4, !dbg !168
  %138 = load float, float* %34, align 4, !dbg !169
  %139 = fmul float %137, %138, !dbg !170
  %140 = fadd float %136, %139, !dbg !171
  store float %140, float* %40, align 4, !dbg !172
  %141 = load float, float* %41, align 4, !dbg !173
  %142 = load float, float* %38, align 4, !dbg !174
  %143 = load float, float* %35, align 4, !dbg !175
  %144 = fmul float %142, %143, !dbg !176
  %145 = fadd float %141, %144, !dbg !177
  store float %145, float* %41, align 4, !dbg !178
  br label %146, !dbg !179

146:                                              ; preds = %129
  %147 = load i32, i32* %45, align 4, !dbg !180
  %148 = add nsw i32 %147, 1, !dbg !180
  store i32 %148, i32* %45, align 4, !dbg !180
  br label %46, !dbg !181, !llvm.loop !182

149:                                              ; preds = %46
  %150 = load float, float* %39, align 4, !dbg !185
  %151 = load float*, float** %24, align 8, !dbg !186
  store float %150, float* %151, align 4, !dbg !187
  %152 = load float, float* %40, align 4, !dbg !188
  %153 = load float*, float** %25, align 8, !dbg !189
  store float %152, float* %153, align 4, !dbg !190
  %154 = load float, float* %41, align 4, !dbg !191
  %155 = load float*, float** %26, align 8, !dbg !192
  store float %154, float* %155, align 4, !dbg !193
  ret void, !dbg !194
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare dso_local float @sqrtf(float) #2

attributes #0 = { noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "NextSilicon clang version 12.0.1 (git@github.com:nextsilicon/next-llvm-project.git 3de6f3afbe52e1056617689bd3c13bbf314b4568)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "Step10_orig.c", directory: "/space/users/zivd/sw/nextutils/applications/nextrunner/applications/haccmk/haccmk/HACCmk")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"NextSilicon clang version 12.0.1 (git@github.com:nextsilicon/next-llvm-project.git 3de6f3afbe52e1056617689bd3c13bbf314b4568)"}
!7 = distinct !DISubprogram(name: "Step10_orig", scope: !1, file: !1, line: 9, type: !8, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11, !11, !11, !11, !11, !12, !12, !12, !12, !12, !12, !12}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!13 = !DILocalVariable(name: "count1", arg: 1, scope: !7, file: !1, line: 9, type: !10)
!14 = !DILocation(line: 9, column: 23, scope: !7)
!15 = !DILocalVariable(name: "xxi", arg: 2, scope: !7, file: !1, line: 9, type: !11)
!16 = !DILocation(line: 9, column: 37, scope: !7)
!17 = !DILocalVariable(name: "yyi", arg: 3, scope: !7, file: !1, line: 9, type: !11)
!18 = !DILocation(line: 9, column: 48, scope: !7)
!19 = !DILocalVariable(name: "zzi", arg: 4, scope: !7, file: !1, line: 9, type: !11)
!20 = !DILocation(line: 9, column: 59, scope: !7)
!21 = !DILocalVariable(name: "fsrrmax2", arg: 5, scope: !7, file: !1, line: 9, type: !11)
!22 = !DILocation(line: 9, column: 70, scope: !7)
!23 = !DILocalVariable(name: "mp_rsm2", arg: 6, scope: !7, file: !1, line: 9, type: !11)
!24 = !DILocation(line: 9, column: 86, scope: !7)
!25 = !DILocalVariable(name: "xx1", arg: 7, scope: !7, file: !1, line: 9, type: !12)
!26 = !DILocation(line: 9, column: 102, scope: !7)
!27 = !DILocalVariable(name: "yy1", arg: 8, scope: !7, file: !1, line: 9, type: !12)
!28 = !DILocation(line: 9, column: 114, scope: !7)
!29 = !DILocalVariable(name: "zz1", arg: 9, scope: !7, file: !1, line: 9, type: !12)
!30 = !DILocation(line: 9, column: 126, scope: !7)
!31 = !DILocalVariable(name: "mass1", arg: 10, scope: !7, file: !1, line: 9, type: !12)
!32 = !DILocation(line: 9, column: 138, scope: !7)
!33 = !DILocalVariable(name: "dxi", arg: 11, scope: !7, file: !1, line: 9, type: !12)
!34 = !DILocation(line: 9, column: 152, scope: !7)
!35 = !DILocalVariable(name: "dyi", arg: 12, scope: !7, file: !1, line: 9, type: !12)
!36 = !DILocation(line: 9, column: 164, scope: !7)
!37 = !DILocalVariable(name: "dzi", arg: 13, scope: !7, file: !1, line: 9, type: !12)
!38 = !DILocation(line: 9, column: 176, scope: !7)
!39 = !DILocalVariable(name: "ma0", scope: !7, file: !1, line: 12, type: !40)
!40 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!41 = !DILocation(line: 12, column: 17, scope: !7)
!42 = !DILocalVariable(name: "ma1", scope: !7, file: !1, line: 12, type: !40)
!43 = !DILocation(line: 12, column: 33, scope: !7)
!44 = !DILocalVariable(name: "ma2", scope: !7, file: !1, line: 12, type: !40)
!45 = !DILocation(line: 12, column: 51, scope: !7)
!46 = !DILocalVariable(name: "ma3", scope: !7, file: !1, line: 12, type: !40)
!47 = !DILocation(line: 12, column: 68, scope: !7)
!48 = !DILocalVariable(name: "ma4", scope: !7, file: !1, line: 12, type: !40)
!49 = !DILocation(line: 12, column: 87, scope: !7)
!50 = !DILocalVariable(name: "ma5", scope: !7, file: !1, line: 12, type: !40)
!51 = !DILocation(line: 12, column: 107, scope: !7)
!52 = !DILocalVariable(name: "dxc", scope: !7, file: !1, line: 14, type: !11)
!53 = !DILocation(line: 14, column: 11, scope: !7)
!54 = !DILocalVariable(name: "dyc", scope: !7, file: !1, line: 14, type: !11)
!55 = !DILocation(line: 14, column: 16, scope: !7)
!56 = !DILocalVariable(name: "dzc", scope: !7, file: !1, line: 14, type: !11)
!57 = !DILocation(line: 14, column: 21, scope: !7)
!58 = !DILocalVariable(name: "m", scope: !7, file: !1, line: 14, type: !11)
!59 = !DILocation(line: 14, column: 26, scope: !7)
!60 = !DILocalVariable(name: "r2", scope: !7, file: !1, line: 14, type: !11)
!61 = !DILocation(line: 14, column: 29, scope: !7)
!62 = !DILocalVariable(name: "f", scope: !7, file: !1, line: 14, type: !11)
!63 = !DILocation(line: 14, column: 33, scope: !7)
!64 = !DILocalVariable(name: "xi", scope: !7, file: !1, line: 14, type: !11)
!65 = !DILocation(line: 14, column: 36, scope: !7)
!66 = !DILocalVariable(name: "yi", scope: !7, file: !1, line: 14, type: !11)
!67 = !DILocation(line: 14, column: 40, scope: !7)
!68 = !DILocalVariable(name: "zi", scope: !7, file: !1, line: 14, type: !11)
!69 = !DILocation(line: 14, column: 44, scope: !7)
!70 = !DILocalVariable(name: "s0", scope: !7, file: !1, line: 15, type: !11)
!71 = !DILocation(line: 15, column: 11, scope: !7)
!72 = !DILocalVariable(name: "s1", scope: !7, file: !1, line: 15, type: !11)
!73 = !DILocation(line: 15, column: 15, scope: !7)
!74 = !DILocalVariable(name: "s2", scope: !7, file: !1, line: 15, type: !11)
!75 = !DILocation(line: 15, column: 19, scope: !7)
!76 = !DILocalVariable(name: "j", scope: !7, file: !1, line: 16, type: !10)
!77 = !DILocation(line: 16, column: 9, scope: !7)
!78 = !DILocation(line: 18, column: 8, scope: !7)
!79 = !DILocation(line: 18, column: 17, scope: !7)
!80 = !DILocation(line: 18, column: 26, scope: !7)
!81 = !DILocation(line: 20, column: 13, scope: !82)
!82 = distinct !DILexicalBlock(scope: !7, file: !1, line: 20, column: 5)
!83 = !DILocation(line: 20, column: 11, scope: !82)
!84 = !DILocation(line: 20, column: 18, scope: !85)
!85 = distinct !DILexicalBlock(scope: !82, file: !1, line: 20, column: 5)
!86 = !DILocation(line: 20, column: 22, scope: !85)
!87 = !DILocation(line: 20, column: 20, scope: !85)
!88 = !DILocation(line: 20, column: 5, scope: !82)
!89 = !DILocation(line: 22, column: 15, scope: !90)
!90 = distinct !DILexicalBlock(scope: !85, file: !1, line: 21, column: 5)
!91 = !DILocation(line: 22, column: 19, scope: !90)
!92 = !DILocation(line: 22, column: 24, scope: !90)
!93 = !DILocation(line: 22, column: 22, scope: !90)
!94 = !DILocation(line: 22, column: 13, scope: !90)
!95 = !DILocation(line: 23, column: 15, scope: !90)
!96 = !DILocation(line: 23, column: 19, scope: !90)
!97 = !DILocation(line: 23, column: 24, scope: !90)
!98 = !DILocation(line: 23, column: 22, scope: !90)
!99 = !DILocation(line: 23, column: 13, scope: !90)
!100 = !DILocation(line: 24, column: 15, scope: !90)
!101 = !DILocation(line: 24, column: 19, scope: !90)
!102 = !DILocation(line: 24, column: 24, scope: !90)
!103 = !DILocation(line: 24, column: 22, scope: !90)
!104 = !DILocation(line: 24, column: 13, scope: !90)
!105 = !DILocation(line: 26, column: 14, scope: !90)
!106 = !DILocation(line: 26, column: 20, scope: !90)
!107 = !DILocation(line: 26, column: 18, scope: !90)
!108 = !DILocation(line: 26, column: 26, scope: !90)
!109 = !DILocation(line: 26, column: 32, scope: !90)
!110 = !DILocation(line: 26, column: 30, scope: !90)
!111 = !DILocation(line: 26, column: 24, scope: !90)
!112 = !DILocation(line: 26, column: 38, scope: !90)
!113 = !DILocation(line: 26, column: 44, scope: !90)
!114 = !DILocation(line: 26, column: 42, scope: !90)
!115 = !DILocation(line: 26, column: 36, scope: !90)
!116 = !DILocation(line: 26, column: 12, scope: !90)
!117 = !DILocation(line: 28, column: 15, scope: !90)
!118 = !DILocation(line: 28, column: 20, scope: !90)
!119 = !DILocation(line: 28, column: 18, scope: !90)
!120 = !DILocation(line: 28, column: 13, scope: !90)
!121 = !DILocation(line: 28, column: 33, scope: !90)
!122 = !DILocation(line: 28, column: 39, scope: !90)
!123 = !DILocation(line: 28, column: 11, scope: !90)
!124 = !DILocation(line: 30, column: 14, scope: !90)
!125 = !DILocation(line: 30, column: 19, scope: !90)
!126 = !DILocation(line: 30, column: 17, scope: !90)
!127 = !DILocation(line: 30, column: 12, scope: !90)
!128 = !DILocation(line: 31, column: 14, scope: !90)
!129 = !DILocation(line: 31, column: 19, scope: !90)
!130 = !DILocation(line: 31, column: 17, scope: !90)
!131 = !DILocation(line: 31, column: 24, scope: !90)
!132 = !DILocation(line: 31, column: 22, scope: !90)
!133 = !DILocation(line: 31, column: 12, scope: !90)
!134 = !DILocation(line: 32, column: 28, scope: !90)
!135 = !DILocation(line: 32, column: 21, scope: !90)
!136 = !DILocation(line: 32, column: 19, scope: !90)
!137 = !DILocation(line: 32, column: 43, scope: !90)
!138 = !DILocation(line: 32, column: 53, scope: !90)
!139 = !DILocation(line: 32, column: 63, scope: !90)
!140 = !DILocation(line: 32, column: 73, scope: !90)
!141 = !DILocation(line: 32, column: 83, scope: !90)
!142 = !DILocation(line: 32, column: 85, scope: !90)
!143 = !DILocation(line: 32, column: 81, scope: !90)
!144 = !DILocation(line: 32, column: 75, scope: !90)
!145 = !DILocation(line: 32, column: 71, scope: !90)
!146 = !DILocation(line: 32, column: 65, scope: !90)
!147 = !DILocation(line: 32, column: 61, scope: !90)
!148 = !DILocation(line: 32, column: 55, scope: !90)
!149 = !DILocation(line: 32, column: 51, scope: !90)
!150 = !DILocation(line: 32, column: 45, scope: !90)
!151 = !DILocation(line: 32, column: 41, scope: !90)
!152 = !DILocation(line: 32, column: 33, scope: !90)
!153 = !DILocation(line: 32, column: 12, scope: !90)
!154 = !DILocation(line: 34, column: 15, scope: !90)
!155 = !DILocation(line: 34, column: 18, scope: !90)
!156 = !DILocation(line: 34, column: 13, scope: !90)
!157 = !DILocation(line: 34, column: 29, scope: !90)
!158 = !DILocation(line: 34, column: 33, scope: !90)
!159 = !DILocation(line: 34, column: 31, scope: !90)
!160 = !DILocation(line: 34, column: 11, scope: !90)
!161 = !DILocation(line: 36, column: 14, scope: !90)
!162 = !DILocation(line: 36, column: 19, scope: !90)
!163 = !DILocation(line: 36, column: 23, scope: !90)
!164 = !DILocation(line: 36, column: 21, scope: !90)
!165 = !DILocation(line: 36, column: 17, scope: !90)
!166 = !DILocation(line: 36, column: 12, scope: !90)
!167 = !DILocation(line: 37, column: 14, scope: !90)
!168 = !DILocation(line: 37, column: 19, scope: !90)
!169 = !DILocation(line: 37, column: 23, scope: !90)
!170 = !DILocation(line: 37, column: 21, scope: !90)
!171 = !DILocation(line: 37, column: 17, scope: !90)
!172 = !DILocation(line: 37, column: 12, scope: !90)
!173 = !DILocation(line: 38, column: 14, scope: !90)
!174 = !DILocation(line: 38, column: 19, scope: !90)
!175 = !DILocation(line: 38, column: 23, scope: !90)
!176 = !DILocation(line: 38, column: 21, scope: !90)
!177 = !DILocation(line: 38, column: 17, scope: !90)
!178 = !DILocation(line: 38, column: 12, scope: !90)
!179 = !DILocation(line: 39, column: 5, scope: !90)
!180 = !DILocation(line: 20, column: 31, scope: !85)
!181 = !DILocation(line: 20, column: 5, scope: !85)
!182 = distinct !{!182, !88, !183, !184}
!183 = !DILocation(line: 39, column: 5, scope: !82)
!184 = !{!"llvm.loop.mustprogress"}
!185 = !DILocation(line: 41, column: 12, scope: !7)
!186 = !DILocation(line: 41, column: 6, scope: !7)
!187 = !DILocation(line: 41, column: 10, scope: !7)
!188 = !DILocation(line: 42, column: 12, scope: !7)
!189 = !DILocation(line: 42, column: 6, scope: !7)
!190 = !DILocation(line: 42, column: 10, scope: !7)
!191 = !DILocation(line: 43, column: 12, scope: !7)
!192 = !DILocation(line: 43, column: 6, scope: !7)
!193 = !DILocation(line: 43, column: 10, scope: !7)
!194 = !DILocation(line: 44, column: 1, scope: !7)