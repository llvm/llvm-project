; REQUIRES: hexagon-registered-target, silver
; This tests correct handling of register spills and fills of
; qf operands during register allocation.

; RUN: llc -mcpu=hexagonv79 -mattr=+hvx-length128b,+hvxv79,+hvx-ieee-fp,+hvx-qfloat,-long-calls -debug-only=handle-qfp %s 2>&1 -o - | FileCheck %s --check-prefixes V79-81,V79
; RUN: llc -mcpu=hexagonv81 -mattr=+hvx-length128b,+hvxv81,+hvx-ieee-fp,+hvx-qfloat,-long-calls -debug-only=handle-qfp %s 2>&1 -o - | FileCheck %s --check-prefixes V79-81,V81

; V79-81: Finding uses of:   renamable $w{{[0-9]+}} = V6_vmpy_qf32_hf
; V79-81: Inserting after conv:   [[VREG0:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable [[VREG0]]
; V79-81-NEXT: Inserting after conv:   [[VREG1:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable [[VREG1]]
; V79-81: Finding uses of:   renamable $w{{[0-9]+}} = V6_vmpy_qf32_hf
; V79-81: Inserting after conv:   [[VREG2:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable [[VREG2]]
; V79-81-NEXT: Inserting after conv:   [[VREG3:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable [[VREG3]]
; V79-81: Finding uses of:   renamable $w{{[0-9]+}} = V6_vmpy_qf32_hf
; V79-81-DAG: Inserting after conv:   [[VREG4:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable [[VREG4]]
; V79-81-DAG: Inserting after conv:   [[VREG5:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable [[VREG5]]
; V79-81-DAG: Inserting new instruction:   $v{{[0-9]+}} = V6_vadd_sf killed renamable [[VREG2]], killed renamable [[VREG0]]
; V79-81-DAG: Inserting new instruction:   $v{{[0-9]+}} = V6_vsub_sf killed renamable $v{{[0-9]+}}, killed renamable $v{{[0-9]+}}
;
; V79-81: Analyzing convert instruction:   renamable [[VREG6:\$v[0-9]+]] = V6_vconv_hf_qf32 killed renamable $w{{[0-9]+}}
; V79: Inserting new instruction:   [[VREG30:\$v[0-9]+]] = V6_vd0
; V79-NEXT: Inserting new instruction:   [[VREG7:\$v[0-9]+]] = V6_vadd_sf killed renamable [[VREG7]], killed [[VREG30]]
; V79: Inserting new instruction:   [[VREG30]] = V6_vd0
; V79-NEXT: Inserting new instruction:   [[VREG8:\$v[0-9]+]] = V6_vadd_sf killed renamable [[VREG8]], killed [[VREG30]]
; V81: Inserting new instruction:  [[VREG7:\$v[0-9]+]] = V6_vconv_qf32_sf killed renamable [[VREG7]]
; V81: Inserting new instruction:  [[VREG8:\$v[0-9]+]] = V6_vconv_qf32_sf killed renamable [[VREG8]]

; V79-81: Analyzing convert instruction:   renamable [[VREG9:\$v[0-9]+]] = V6_vconv_sf_qf32 killed renamable $v{{[0-9]+}}
; V79: Inserting new instruction:   [[VREG30]] = V6_vd0
; V79-NEXT: Inserting new instruction:   [[VREG10:\$v[0-9]+]] = V6_vadd_sf killed renamable [[VREG10]], killed [[VREG30]]
; V81: Inserting new instruction:  [[VREG8:\$v[0-9]+]] = V6_vconv_qf32_sf killed renamable [[VREG8]]

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@.str.1 = private unnamed_addr constant [9 x i8] c"0x%08lx \00", align 1
@.str.3 = private unnamed_addr constant [173 x i8] c"/prj/qct/llvm/devops/aether/hexbuild/test_trees/MASTER/test/regress/features/hexagon/arch_v68/hvx_ieee_fp/hvx_ieee_fp_test.c:126 0 && \22ERROR: Failed to acquire HVX unit.\\n\22\00", align 1
@__func__.main = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.5 = private unnamed_addr constant [33 x i8] c"half -3 converted to vhf = %.2f\0A\00", align 1
@.str.6 = private unnamed_addr constant [35 x i8] c"uhalf 32k converted to vhf = %.2f\0A\00", align 1
@.str.7 = private unnamed_addr constant [32 x i8] c"sf 0.5 converted to vhf = %.2f\0A\00", align 1
@.str.8 = private unnamed_addr constant [32 x i8] c"vhf 4.0 conveted to ubyte = %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [32 x i8] c"vhf 2.0 conveted to uhalf = %d\0A\00", align 1
@.str.10 = private unnamed_addr constant [30 x i8] c"byte 4 conveted to hf = %.2f\0A\00", align 1
@.str.11 = private unnamed_addr constant [31 x i8] c"ubyte 4 conveted to hf = %.2f\0A\00", align 1
@.str.12 = private unnamed_addr constant [27 x i8] c"hf -3 conveted to sf = %f\0A\00", align 1
@.str.13 = private unnamed_addr constant [31 x i8] c"vhf 4.0 conveted to byte = %d\0A\00", align 1
@.str.14 = private unnamed_addr constant [31 x i8] c"vhf 4.0 conveted to half = %d\0A\00", align 1
@.str.16 = private unnamed_addr constant [33 x i8] c"max of hf 2.0 and hf 4.0 = %.2f\0A\00", align 1
@.str.17 = private unnamed_addr constant [33 x i8] c"min of hf 2.0 and hf 4.0 = %.2f\0A\00", align 1
@.str.18 = private unnamed_addr constant [32 x i8] c"max of sf 0.5 and sf 0.25 = %f\0A\00", align 1
@.str.19 = private unnamed_addr constant [32 x i8] c"min of sf 0.5 and sf 0.25 = %f\0A\00", align 1
@.str.21 = private unnamed_addr constant [25 x i8] c"negate of hf 4.0 = %.2f\0A\00", align 1
@.str.22 = private unnamed_addr constant [23 x i8] c"abs of hf -6.0 = %.2f\0A\00", align 1
@.str.23 = private unnamed_addr constant [23 x i8] c"negate of sf 0.5 = %f\0A\00", align 1
@.str.24 = private unnamed_addr constant [22 x i8] c"abs of sf -0.25 = %f\0A\00", align 1
@.str.26 = private unnamed_addr constant [32 x i8] c"hf add of 4.0 and -6.0  = %.2f\0A\00", align 1
@.str.27 = private unnamed_addr constant [32 x i8] c"hf sub of 4.0 and -6.0  = %.2f\0A\00", align 1
@.str.28 = private unnamed_addr constant [31 x i8] c"sf add of 0.5 and -0.25  = %f\0A\00", align 1
@.str.29 = private unnamed_addr constant [31 x i8] c"sf sub of 0.5 and -0.25  = %f\0A\00", align 1
@.str.30 = private unnamed_addr constant [36 x i8] c"sf add of hf 4.0 and hf -6.0  = %f\0A\00", align 1
@.str.31 = private unnamed_addr constant [36 x i8] c"sf sub of hf 4.0 and hf -6.0  = %f\0A\00", align 1
@.str.33 = private unnamed_addr constant [32 x i8] c"hf mpy of 4.0 and -6.0  = %.2f\0A\00", align 1
@.str.34 = private unnamed_addr constant [35 x i8] c"hf accmpy of 4.0 and -6.0  = %.2f\0A\00", align 1
@.str.35 = private unnamed_addr constant [36 x i8] c"sf mpy of hf 4.0 and hf -6.0  = %f\0A\00", align 1
@.str.36 = private unnamed_addr constant [39 x i8] c"sf accmpy of hf 4.0 and hf -6.0  = %f\0A\00", align 1
@.str.37 = private unnamed_addr constant [31 x i8] c"sf mpy of 0.5 and -0.25  = %f\0A\00", align 1
@.str.39 = private unnamed_addr constant [25 x i8] c"w copy from sf 0.5 = %f\0A\00", align 1
@str = private unnamed_addr constant [35 x i8] c"ERROR: Failed to acquire HVX unit.\00", align 1
@str.40 = private unnamed_addr constant [25 x i8] c"\0AConversion intructions\0A\00", align 1
@str.41 = private unnamed_addr constant [23 x i8] c"\0AMin/Max instructions\0A\00", align 1
@str.42 = private unnamed_addr constant [23 x i8] c"\0Aabs/neg instructions\0A\00", align 1
@str.43 = private unnamed_addr constant [23 x i8] c"\0Aadd/sub instructions\0A\00", align 1
@str.44 = private unnamed_addr constant [24 x i8] c"\0Amultiply instructions\0A\00", align 1
@str.45 = private unnamed_addr constant [19 x i8] c"\0Acopy instruction\0A\00", align 1

declare dso_local void @print_vector_words(<32 x i32> noundef %x) local_unnamed_addr #0

; Function Attrs: nofree nounwind optsize
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0

; Function Attrs: nounwind optsize
define dso_local i32 @main(i32 noundef %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #1 {
entry:
  %call = tail call i32 @acquire_vector_unit(i8 noundef zeroext 0) #6
  %tobool.not = icmp eq i32 %call, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @_Assert(ptr noundef nonnull @.str.3, ptr noundef nonnull @__func__.main) #7
  unreachable

if.end:                                           ; preds = %entry
  tail call void @set_double_vector_mode() #6
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 16384)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 17408)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 -14848)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1056964608)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1048576000)
  %5 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -1098907648)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 -3)
  %7 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 32768)
  %puts147 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.40)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(<32 x i32> %6)
  %bc.i = bitcast <32 x i32> %8 to <64 x half>
  %9 = extractelement <64 x half> %bc.i, i64 0
  %conv = fpext half %9 to double
  %call12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, double noundef %conv) #6
  %10 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.uh.128B(<32 x i32> %7)
  %bc.i153 = bitcast <32 x i32> %10 to <64 x half>
  %11 = extractelement <64 x half> %bc.i153, i64 0
  %conv14 = fpext half %11 to double
  %call15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, double noundef %conv14) #6
  %12 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> %3, <32 x i32> %3)
  %bc.i155 = bitcast <32 x i32> %12 to <64 x half>
  %13 = extractelement <64 x half> %bc.i155, i64 0
  %conv17 = fpext half %13 to double
  %call18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, double noundef %conv17) #6
  %14 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.ub.hf.128B(<32 x i32> %1, <32 x i32> %1)
  %15 = bitcast <32 x i32> %14 to <128 x i8>
  %conv.i = extractelement <128 x i8> %15, i64 0
  %conv20 = zext i8 %conv.i to i32
  %call21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef %conv20) #6
  %16 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.uh.hf.128B(<32 x i32> %0)
  %17 = bitcast <32 x i32> %16 to <64 x i16>
  %conv.i157 = extractelement <64 x i16> %17, i64 0
  %conv23 = sext i16 %conv.i157 to i32
  %call24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef %conv23) #6
  %18 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.hf.b.128B(<32 x i32> %14)
  %bc.i158 = bitcast <64 x i32> %18 to <128 x half>
  %19 = extractelement <128 x half> %bc.i158, i64 0
  %conv26 = fpext half %19 to double
  %call27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, double noundef %conv26) #6
  %20 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.hf.ub.128B(<32 x i32> %14)
  %bc.i159 = bitcast <64 x i32> %20 to <128 x half>
  %21 = extractelement <128 x half> %bc.i159, i64 0
  %conv29 = fpext half %21 to double
  %call30 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, double noundef %conv29) #6
  %22 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32> %8)
  %bc.i161 = bitcast <64 x i32> %22 to <64 x float>
  %23 = extractelement <64 x float> %bc.i161, i64 0
  %conv32 = fpext float %23 to double
  %call33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, double noundef %conv32) #6
  %24 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.b.hf.128B(<32 x i32> %1, <32 x i32> %1)
  %25 = bitcast <32 x i32> %24 to <128 x i8>
  %conv.i162 = extractelement <128 x i8> %25, i64 0
  %conv35 = zext i8 %conv.i162 to i32
  %call36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef %conv35) #6
  %26 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.h.hf.128B(<32 x i32> %1)
  %27 = bitcast <32 x i32> %26 to <64 x i16>
  %conv.i163 = extractelement <64 x i16> %27, i64 0
  %conv38 = sext i16 %conv.i163 to i32
  %call39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef %conv38) #6
  %28 = tail call <32 x i32> @llvm.hexagon.V6.vfmax.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %puts148 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.41)
  %bc.i164 = bitcast <32 x i32> %28 to <64 x half>
  %29 = extractelement <64 x half> %bc.i164, i64 0
  %conv42 = fpext half %29 to double
  %call43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, double noundef %conv42) #6
  %30 = tail call <32 x i32> @llvm.hexagon.V6.vfmin.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %bc.i166 = bitcast <32 x i32> %30 to <64 x half>
  %31 = extractelement <64 x half> %bc.i166, i64 0
  %conv45 = fpext half %31 to double
  %call46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.17, double noundef %conv45) #6
  %32 = tail call <32 x i32> @llvm.hexagon.V6.vfmax.sf.128B(<32 x i32> %3, <32 x i32> %4)
  %bc.i168 = bitcast <32 x i32> %32 to <32 x float>
  %33 = extractelement <32 x float> %bc.i168, i64 0
  %conv48 = fpext float %33 to double
  %call49 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.18, double noundef %conv48) #6
  %34 = tail call <32 x i32> @llvm.hexagon.V6.vfmin.sf.128B(<32 x i32> %3, <32 x i32> %4)
  %bc.i169 = bitcast <32 x i32> %34 to <32 x float>
  %35 = extractelement <32 x float> %bc.i169, i64 0
  %conv51 = fpext float %35 to double
  %call52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.19, double noundef %conv51) #6
  %puts149 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.42)
  %36 = tail call <32 x i32> @llvm.hexagon.V6.vfneg.hf.128B(<32 x i32> %1)
  %bc.i170 = bitcast <32 x i32> %36 to <64 x half>
  %37 = extractelement <64 x half> %bc.i170, i64 0
  %conv55 = fpext half %37 to double
  %call56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.21, double noundef %conv55) #6
  %38 = tail call <32 x i32> @llvm.hexagon.V6.vabs.hf.128B(<32 x i32> %2)
  %bc.i172 = bitcast <32 x i32> %38 to <64 x half>
  %39 = extractelement <64 x half> %bc.i172, i64 0
  %conv58 = fpext half %39 to double
  %call59 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.22, double noundef %conv58) #6
  %40 = tail call <32 x i32> @llvm.hexagon.V6.vfneg.sf.128B(<32 x i32> %3)
  %bc.i174 = bitcast <32 x i32> %40 to <32 x float>
  %41 = extractelement <32 x float> %bc.i174, i64 0
  %conv61 = fpext float %41 to double
  %call62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, double noundef %conv61) #6
  %42 = tail call <32 x i32> @llvm.hexagon.V6.vabs.sf.128B(<32 x i32> %5)
  %bc.i175 = bitcast <32 x i32> %42 to <32 x float>
  %43 = extractelement <32 x float> %bc.i175, i64 0
  %conv64 = fpext float %43 to double
  %call65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.24, double noundef %conv64) #6
  %puts150 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.43)
  %44 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.hf.128B(<32 x i32> %1, <32 x i32> %2)
  %bc.i176 = bitcast <32 x i32> %44 to <64 x half>
  %45 = extractelement <64 x half> %bc.i176, i64 0
  %conv68 = fpext half %45 to double
  %call69 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.26, double noundef %conv68) #6
  %46 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.hf.128B(<32 x i32> %1, <32 x i32> %2)
  %bc.i178 = bitcast <32 x i32> %46 to <64 x half>
  %47 = extractelement <64 x half> %bc.i178, i64 0
  %conv71 = fpext half %47 to double
  %call72 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, double noundef %conv71) #6
  %48 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> %3, <32 x i32> %5)
  %bc.i180 = bitcast <32 x i32> %48 to <32 x float>
  %49 = extractelement <32 x float> %bc.i180, i64 0
  %conv74 = fpext float %49 to double
  %call75 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.28, double noundef %conv74) #6
  %50 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32> %3, <32 x i32> %5)
  %bc.i181 = bitcast <32 x i32> %50 to <32 x float>
  %51 = extractelement <32 x float> %bc.i181, i64 0
  %conv77 = fpext float %51 to double
  %call78 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.29, double noundef %conv77) #6
  %52 = tail call <64 x i32> @llvm.hexagon.V6.vadd.sf.hf.128B(<32 x i32> %1, <32 x i32> %2)
  %bc.i182 = bitcast <64 x i32> %52 to <64 x float>
  %53 = extractelement <64 x float> %bc.i182, i64 0
  %conv80 = fpext float %53 to double
  %call81 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.30, double noundef %conv80) #6
  %54 = tail call <64 x i32> @llvm.hexagon.V6.vsub.sf.hf.128B(<32 x i32> %1, <32 x i32> %2)
  %bc.i183 = bitcast <64 x i32> %54 to <64 x float>
  %55 = extractelement <64 x float> %bc.i183, i64 0
  %conv83 = fpext float %55 to double
  %call84 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.31, double noundef %conv83) #6
  %puts151 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.44)
  %56 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.128B(<32 x i32> %1, <32 x i32> %2)
  %bc.i184 = bitcast <32 x i32> %56 to <64 x half>
  %57 = extractelement <64 x half> %bc.i184, i64 0
  %conv87 = fpext half %57 to double
  %call88 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.33, double noundef %conv87) #6
  %58 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.acc.128B(<32 x i32> %56, <32 x i32> %1, <32 x i32> %2)
  %bc.i186 = bitcast <32 x i32> %58 to <64 x half>
  %59 = extractelement <64 x half> %bc.i186, i64 0
  %conv90 = fpext half %59 to double
  %call91 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.34, double noundef %conv90) #6
  %60 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(<32 x i32> %1, <32 x i32> %2)
  %bc.i188 = bitcast <64 x i32> %60 to <64 x float>
  %61 = extractelement <64 x float> %bc.i188, i64 0
  %conv93 = fpext float %61 to double
  %call94 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.35, double noundef %conv93) #6
  %62 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32> %60, <32 x i32> %1, <32 x i32> %2)
  %bc.i189 = bitcast <64 x i32> %62 to <64 x float>
  %63 = extractelement <64 x float> %bc.i189, i64 0
  %conv96 = fpext float %63 to double
  %call97 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.36, double noundef %conv96) #6
  %64 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %3, <32 x i32> %5)
  %bc.i190 = bitcast <32 x i32> %64 to <32 x float>
  %65 = extractelement <32 x float> %bc.i190, i64 0
  %conv99 = fpext float %65 to double
  %call100 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.37, double noundef %conv99) #6
  %puts152 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.45)
  %66 = tail call <32 x i32> @llvm.hexagon.V6.vassign.fp.128B(<32 x i32> %3)
  %bc.i191 = bitcast <32 x i32> %66 to <32 x float>
  %67 = extractelement <32 x float> %bc.i191, i64 0
  %conv103 = fpext float %67 to double
  %call104 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.39, double noundef %conv103) #6
  ret i32 0
}

; Function Attrs: optsize
declare dso_local i32 @acquire_vector_unit(i8 noundef zeroext) local_unnamed_addr #2

; Function Attrs: noreturn nounwind optsize
declare dso_local void @_Assert(ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: optsize
declare dso_local void @set_double_vector_mode(...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.uh.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.ub.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.uh.hf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vcvt.hf.b.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vcvt.hf.ub.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.b.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.h.hf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vfmax.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vfmin.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vfmax.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vfmin.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vfneg.hf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vabs.hf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vfneg.sf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vabs.sf.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.hf.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vadd.sf.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vsub.sf.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.acc.128B(<32 x i32>, <32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32>, <32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vassign.fp.128B(<32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #4

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5
