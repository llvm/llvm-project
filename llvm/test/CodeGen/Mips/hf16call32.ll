; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=stel

@x = common global float 0.000000e+00, align 4
@y = common global float 0.000000e+00, align 4
@xd = common global double 0.000000e+00, align 8
@yd = common global double 0.000000e+00, align 8
@xy = common global { float, float } zeroinitializer, align 4
@xyd = common global { double, double } zeroinitializer, align 8
@ret_sf = common global float 0.000000e+00, align 4
@ret_df = common global double 0.000000e+00, align 8
@ret_sc = common global { float, float } zeroinitializer, align 4
@ret_dc = common global { double, double } zeroinitializer, align 8
@lx = common global float 0.000000e+00, align 4
@ly = common global float 0.000000e+00, align 4
@lxd = common global double 0.000000e+00, align 8
@lyd = common global double 0.000000e+00, align 8
@lxy = common global { float, float } zeroinitializer, align 4
@lxyd = common global { double, double } zeroinitializer, align 8
@lret_sf = common global float 0.000000e+00, align 4
@lret_df = common global double 0.000000e+00, align 8
@lret_sc = common global { float, float } zeroinitializer, align 4
@lret_dc = common global { double, double } zeroinitializer, align 8
@.str = private unnamed_addr constant [10 x i8] c"%f %f %i\0A\00", align 1
@.str1 = private unnamed_addr constant [16 x i8] c"%f=%f %f=%f %i\0A\00", align 1
@.str2 = private unnamed_addr constant [22 x i8] c"%f=%f %f=%f %f=%f %i\0A\00", align 1
@.str3 = private unnamed_addr constant [18 x i8] c"%f+%fi=%f+%fi %i\0A\00", align 1
@.str4 = private unnamed_addr constant [24 x i8] c"%f+%fi=%f+%fi %f=%f %i\0A\00", align 1

; Function Attrs: nounwind
define void @clear() #0 {
entry:
  store float 1.000000e+00, float* @x, align 4
  store float 1.000000e+00, float* @y, align 4
  store double 1.000000e+00, double* @xd, align 8
  store double 1.000000e+00, double* @yd, align 8
  store float 1.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @xy, i32 0, i32 0)
  store float 0.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @xy, i32 0, i32 1)
  store double 1.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @xyd, i32 0, i32 0)
  store double 0.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @xyd, i32 0, i32 1)
  store float 1.000000e+00, float* @ret_sf, align 4
  store double 1.000000e+00, double* @ret_df, align 8
  store float 1.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  store float 0.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  store double 1.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  store double 0.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  store float 0.000000e+00, float* @lx, align 4
  store float 0.000000e+00, float* @ly, align 4
  store double 0.000000e+00, double* @lxd, align 8
  store double 0.000000e+00, double* @lyd, align 8
  store float 0.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @lxy, i32 0, i32 0)
  store float 0.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @lxy, i32 0, i32 1)
  store double 0.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @lxyd, i32 0, i32 0)
  store double 0.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @lxyd, i32 0, i32 1)
  store float 0.000000e+00, float* @lret_sf, align 4
  store double 0.000000e+00, double* @lret_df, align 8
  store float 0.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  store float 0.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  store double 0.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  store double 0.000000e+00, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  ret void
}

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  call void @clear()
  store float 1.500000e+00, float* @lx, align 4
  %0 = load float, float* @lx, align 4
  call void @v_sf(float %0)
  %1 = load float, float* @x, align 4
  %conv = fpext float %1 to double
  %2 = load float, float* @lx, align 4
  %conv1 = fpext float %2 to double
  %3 = load float, float* @x, align 4
  %4 = load float, float* @lx, align 4
  %cmp = fcmp oeq float %3, %4
  %conv2 = zext i1 %cmp to i32
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), double %conv, double %conv1, i32 %conv2)
  call void @clear()
  store double 0x41678C29C0000000, double* @lxd, align 8
  %5 = load double, double* @lxd, align 8
  call void @v_df(double %5)
  %6 = load double, double* @xd, align 8
  %7 = load double, double* @lxd, align 8
  %8 = load double, double* @xd, align 8
  %9 = load double, double* @lxd, align 8
  %cmp3 = fcmp oeq double %8, %9
  %conv4 = zext i1 %cmp3 to i32
  %call5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), double %6, double %7, i32 %conv4)
  call void @clear()
  store float 9.000000e+00, float* @lx, align 4
  store float 1.000000e+01, float* @ly, align 4
  %10 = load float, float* @lx, align 4
  %11 = load float, float* @ly, align 4
  call void @v_sf_sf(float %10, float %11)
  %12 = load float, float* @x, align 4
  %conv6 = fpext float %12 to double
  %13 = load float, float* @lx, align 4
  %conv7 = fpext float %13 to double
  %14 = load float, float* @y, align 4
  %conv8 = fpext float %14 to double
  %15 = load float, float* @ly, align 4
  %conv9 = fpext float %15 to double
  %16 = load float, float* @x, align 4
  %17 = load float, float* @lx, align 4
  %cmp10 = fcmp oeq float %16, %17
  br i1 %cmp10, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
  %18 = load float, float* @y, align 4
  %19 = load float, float* @ly, align 4
  %cmp12 = fcmp oeq float %18, %19
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %20 = phi i1 [ false, %entry ], [ %cmp12, %land.rhs ]
  %land.ext = zext i1 %20 to i32
  %call14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %conv6, double %conv7, double %conv8, double %conv9, i32 %land.ext)
  call void @clear()
  store float 0x3FFE666660000000, float* @lx, align 4
  store double 0x4007E613249FF279, double* @lyd, align 8
  %21 = load float, float* @lx, align 4
  %22 = load double, double* @lyd, align 8
  call void @v_sf_df(float %21, double %22)
  %23 = load float, float* @x, align 4
  %conv15 = fpext float %23 to double
  %24 = load float, float* @lx, align 4
  %conv16 = fpext float %24 to double
  %25 = load double, double* @yd, align 8
  %26 = load double, double* @lyd, align 8
  %27 = load float, float* @x, align 4
  %28 = load float, float* @lx, align 4
  %cmp17 = fcmp oeq float %27, %28
  %conv18 = zext i1 %cmp17 to i32
  %29 = load double, double* @yd, align 8
  %30 = load double, double* @lyd, align 8
  %cmp19 = fcmp oeq double %29, %30
  %conv20 = zext i1 %cmp19 to i32
  %and = and i32 %conv18, %conv20
  %call21 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %conv15, double %conv16, double %25, double %26, i32 %and)
  call void @clear()
  store double 0x4194E54F94000000, double* @lxd, align 8
  store float 7.600000e+01, float* @ly, align 4
  %31 = load double, double* @lxd, align 8
  %32 = load float, float* @ly, align 4
  call void @v_df_sf(double %31, float %32)
  %33 = load double, double* @xd, align 8
  %34 = load double, double* @lxd, align 8
  %35 = load float, float* @y, align 4
  %conv22 = fpext float %35 to double
  %36 = load float, float* @ly, align 4
  %conv23 = fpext float %36 to double
  %37 = load double, double* @xd, align 8
  %38 = load double, double* @lxd, align 8
  %cmp24 = fcmp oeq double %37, %38
  %conv25 = zext i1 %cmp24 to i32
  %39 = load float, float* @y, align 4
  %40 = load float, float* @ly, align 4
  %cmp26 = fcmp oeq float %39, %40
  %conv27 = zext i1 %cmp26 to i32
  %and28 = and i32 %conv25, %conv27
  %call29 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %33, double %34, double %conv22, double %conv23, i32 %and28)
  call void @clear()
  store double 7.365198e+07, double* @lxd, align 8
  store double 0x416536CD80000000, double* @lyd, align 8
  %41 = load double, double* @lxd, align 8
  %42 = load double, double* @lyd, align 8
  call void @v_df_df(double %41, double %42)
  %43 = load double, double* @xd, align 8
  %44 = load double, double* @lxd, align 8
  %45 = load double, double* @yd, align 8
  %46 = load double, double* @lyd, align 8
  %47 = load double, double* @xd, align 8
  %48 = load double, double* @lxd, align 8
  %cmp30 = fcmp oeq double %47, %48
  %conv31 = zext i1 %cmp30 to i32
  %49 = load double, double* @yd, align 8
  %50 = load double, double* @lyd, align 8
  %cmp32 = fcmp oeq double %49, %50
  %conv33 = zext i1 %cmp32 to i32
  %and34 = and i32 %conv31, %conv33
  %call35 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %43, double %44, double %45, double %46, i32 %and34)
  call void @clear()
  store float 0x4016666660000000, float* @ret_sf, align 4
  %call36 = call float @sf_v()
  store float %call36, float* @lret_sf, align 4
  %51 = load float, float* @ret_sf, align 4
  %conv37 = fpext float %51 to double
  %52 = load float, float* @lret_sf, align 4
  %conv38 = fpext float %52 to double
  %53 = load float, float* @ret_sf, align 4
  %54 = load float, float* @lret_sf, align 4
  %cmp39 = fcmp oeq float %53, %54
  %conv40 = zext i1 %cmp39 to i32
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), double %conv37, double %conv38, i32 %conv40)
  call void @clear()
  store float 4.587300e+06, float* @ret_sf, align 4
  store float 3.420000e+02, float* @lx, align 4
  %55 = load float, float* @lx, align 4
  %call42 = call float @sf_sf(float %55)
  store float %call42, float* @lret_sf, align 4
  %56 = load float, float* @ret_sf, align 4
  %conv43 = fpext float %56 to double
  %57 = load float, float* @lret_sf, align 4
  %conv44 = fpext float %57 to double
  %58 = load float, float* @x, align 4
  %conv45 = fpext float %58 to double
  %59 = load float, float* @lx, align 4
  %conv46 = fpext float %59 to double
  %60 = load float, float* @ret_sf, align 4
  %61 = load float, float* @lret_sf, align 4
  %cmp47 = fcmp oeq float %60, %61
  %conv48 = zext i1 %cmp47 to i32
  %62 = load float, float* @x, align 4
  %63 = load float, float* @lx, align 4
  %cmp49 = fcmp oeq float %62, %63
  %conv50 = zext i1 %cmp49 to i32
  %and51 = and i32 %conv48, %conv50
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %conv43, double %conv44, double %conv45, double %conv46, i32 %and51)
  call void @clear()
  store float 4.445910e+06, float* @ret_sf, align 4
  store double 0x419A7DB294000000, double* @lxd, align 8
  %64 = load double, double* @lxd, align 8
  %call53 = call float @sf_df(double %64)
  store float %call53, float* @lret_sf, align 4
  %65 = load float, float* @ret_sf, align 4
  %conv54 = fpext float %65 to double
  %66 = load float, float* @lret_sf, align 4
  %conv55 = fpext float %66 to double
  %67 = load double, double* @xd, align 8
  %68 = load double, double* @lxd, align 8
  %69 = load float, float* @ret_sf, align 4
  %70 = load float, float* @lret_sf, align 4
  %cmp56 = fcmp oeq float %69, %70
  %conv57 = zext i1 %cmp56 to i32
  %71 = load double, double* @xd, align 8
  %72 = load double, double* @lxd, align 8
  %cmp58 = fcmp oeq double %71, %72
  %conv59 = zext i1 %cmp58 to i32
  %and60 = and i32 %conv57, %conv59
  %call61 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %conv54, double %conv55, double %67, double %68, i32 %and60)
  call void @clear()
  store float 0x3FFF4BC6A0000000, float* @ret_sf, align 4
  store float 4.445500e+03, float* @lx, align 4
  store float 0x4068ACCCC0000000, float* @ly, align 4
  %73 = load float, float* @lx, align 4
  %74 = load float, float* @ly, align 4
  %call62 = call float @sf_sf_sf(float %73, float %74)
  store float %call62, float* @lret_sf, align 4
  %75 = load float, float* @ret_sf, align 4
  %conv63 = fpext float %75 to double
  %76 = load float, float* @lret_sf, align 4
  %conv64 = fpext float %76 to double
  %77 = load float, float* @x, align 4
  %conv65 = fpext float %77 to double
  %78 = load float, float* @lx, align 4
  %conv66 = fpext float %78 to double
  %79 = load float, float* @y, align 4
  %conv67 = fpext float %79 to double
  %80 = load float, float* @ly, align 4
  %conv68 = fpext float %80 to double
  %81 = load float, float* @ret_sf, align 4
  %82 = load float, float* @lret_sf, align 4
  %cmp69 = fcmp oeq float %81, %82
  br i1 %cmp69, label %land.lhs.true, label %land.end76

land.lhs.true:                                    ; preds = %land.end
  %83 = load float, float* @x, align 4
  %84 = load float, float* @lx, align 4
  %cmp71 = fcmp oeq float %83, %84
  br i1 %cmp71, label %land.rhs73, label %land.end76

land.rhs73:                                       ; preds = %land.lhs.true
  %85 = load float, float* @y, align 4
  %86 = load float, float* @ly, align 4
  %cmp74 = fcmp oeq float %85, %86
  br label %land.end76

land.end76:                                       ; preds = %land.rhs73, %land.lhs.true, %land.end
  %87 = phi i1 [ false, %land.lhs.true ], [ false, %land.end ], [ %cmp74, %land.rhs73 ]
  %land.ext77 = zext i1 %87 to i32
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %conv63, double %conv64, double %conv65, double %conv66, double %conv67, double %conv68, i32 %land.ext77)
  call void @clear()
  store float 9.991300e+04, float* @ret_sf, align 4
  store float 1.114500e+04, float* @lx, align 4
  store double 9.994445e+07, double* @lyd, align 8
  %88 = load float, float* @lx, align 4
  %89 = load double, double* @lyd, align 8
  %call79 = call float @sf_sf_df(float %88, double %89)
  store float %call79, float* @lret_sf, align 4
  %90 = load float, float* @ret_sf, align 4
  %conv80 = fpext float %90 to double
  %91 = load float, float* @lret_sf, align 4
  %conv81 = fpext float %91 to double
  %92 = load float, float* @x, align 4
  %conv82 = fpext float %92 to double
  %93 = load float, float* @lx, align 4
  %conv83 = fpext float %93 to double
  %94 = load double, double* @yd, align 8
  %95 = load double, double* @lyd, align 8
  %96 = load float, float* @ret_sf, align 4
  %97 = load float, float* @lret_sf, align 4
  %cmp84 = fcmp oeq float %96, %97
  br i1 %cmp84, label %land.lhs.true86, label %land.end92

land.lhs.true86:                                  ; preds = %land.end76
  %98 = load float, float* @x, align 4
  %99 = load float, float* @lx, align 4
  %cmp87 = fcmp oeq float %98, %99
  br i1 %cmp87, label %land.rhs89, label %land.end92

land.rhs89:                                       ; preds = %land.lhs.true86
  %100 = load double, double* @yd, align 8
  %101 = load double, double* @lyd, align 8
  %cmp90 = fcmp oeq double %100, %101
  br label %land.end92

land.end92:                                       ; preds = %land.rhs89, %land.lhs.true86, %land.end76
  %102 = phi i1 [ false, %land.lhs.true86 ], [ false, %land.end76 ], [ %cmp90, %land.rhs89 ]
  %land.ext93 = zext i1 %102 to i32
  %call94 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %conv80, double %conv81, double %conv82, double %conv83, double %94, double %95, i32 %land.ext93)
  call void @clear()
  store float 0x417CCC7A00000000, float* @ret_sf, align 4
  store double 0x4172034530000000, double* @lxd, align 8
  store float 4.456200e+04, float* @ly, align 4
  %103 = load double, double* @lxd, align 8
  %104 = load float, float* @ly, align 4
  %call95 = call float @sf_df_sf(double %103, float %104)
  store float %call95, float* @lret_sf, align 4
  %105 = load float, float* @ret_sf, align 4
  %conv96 = fpext float %105 to double
  %106 = load float, float* @lret_sf, align 4
  %conv97 = fpext float %106 to double
  %107 = load double, double* @xd, align 8
  %108 = load double, double* @lxd, align 8
  %109 = load float, float* @y, align 4
  %conv98 = fpext float %109 to double
  %110 = load float, float* @ly, align 4
  %conv99 = fpext float %110 to double
  %111 = load float, float* @ret_sf, align 4
  %112 = load float, float* @lret_sf, align 4
  %cmp100 = fcmp oeq float %111, %112
  br i1 %cmp100, label %land.lhs.true102, label %land.end108

land.lhs.true102:                                 ; preds = %land.end92
  %113 = load double, double* @xd, align 8
  %114 = load double, double* @lxd, align 8
  %cmp103 = fcmp oeq double %113, %114
  br i1 %cmp103, label %land.rhs105, label %land.end108

land.rhs105:                                      ; preds = %land.lhs.true102
  %115 = load float, float* @y, align 4
  %116 = load float, float* @ly, align 4
  %cmp106 = fcmp oeq float %115, %116
  br label %land.end108

land.end108:                                      ; preds = %land.rhs105, %land.lhs.true102, %land.end92
  %117 = phi i1 [ false, %land.lhs.true102 ], [ false, %land.end92 ], [ %cmp106, %land.rhs105 ]
  %land.ext109 = zext i1 %117 to i32
  %call110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %conv96, double %conv97, double %107, double %108, double %conv98, double %conv99, i32 %land.ext109)
  call void @clear()
  store float 3.987721e+06, float* @ret_sf, align 4
  store double 0x3FF1F49F6DDDC2D8, double* @lxd, align 8
  store double 0x409129F306A2B170, double* @lyd, align 8
  %118 = load double, double* @lxd, align 8
  %119 = load double, double* @lyd, align 8
  %call111 = call float @sf_df_df(double %118, double %119)
  store float %call111, float* @lret_sf, align 4
  %120 = load float, float* @ret_sf, align 4
  %conv112 = fpext float %120 to double
  %121 = load float, float* @lret_sf, align 4
  %conv113 = fpext float %121 to double
  %122 = load double, double* @xd, align 8
  %123 = load double, double* @lxd, align 8
  %124 = load double, double* @yd, align 8
  %125 = load double, double* @lyd, align 8
  %126 = load float, float* @ret_sf, align 4
  %127 = load float, float* @lret_sf, align 4
  %cmp114 = fcmp oeq float %126, %127
  br i1 %cmp114, label %land.lhs.true116, label %land.end122

land.lhs.true116:                                 ; preds = %land.end108
  %128 = load double, double* @xd, align 8
  %129 = load double, double* @lxd, align 8
  %cmp117 = fcmp oeq double %128, %129
  br i1 %cmp117, label %land.rhs119, label %land.end122

land.rhs119:                                      ; preds = %land.lhs.true116
  %130 = load double, double* @yd, align 8
  %131 = load double, double* @lyd, align 8
  %cmp120 = fcmp oeq double %130, %131
  br label %land.end122

land.end122:                                      ; preds = %land.rhs119, %land.lhs.true116, %land.end108
  %132 = phi i1 [ false, %land.lhs.true116 ], [ false, %land.end108 ], [ %cmp120, %land.rhs119 ]
  %land.ext123 = zext i1 %132 to i32
  %call124 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %conv112, double %conv113, double %122, double %123, double %124, double %125, i32 %land.ext123)
  call void @clear()
  store double 1.561234e+01, double* @ret_df, align 8
  %call125 = call double @df_v()
  store double %call125, double* @lret_df, align 8
  %133 = load double, double* @ret_df, align 8
  %134 = load double, double* @lret_df, align 8
  %135 = load double, double* @ret_df, align 8
  %136 = load double, double* @lret_df, align 8
  %cmp126 = fcmp oeq double %135, %136
  %conv127 = zext i1 %cmp126 to i32
  %call128 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), double %133, double %134, i32 %conv127)
  call void @clear()
  store double 1.345873e+01, double* @ret_df, align 8
  store float 3.434520e+05, float* @lx, align 4
  %137 = load float, float* @lx, align 4
  %call129 = call double @df_sf(float %137)
  store double %call129, double* @lret_df, align 8
  %138 = load double, double* @ret_df, align 8
  %139 = load double, double* @lret_df, align 8
  %140 = load float, float* @x, align 4
  %conv130 = fpext float %140 to double
  %141 = load float, float* @lx, align 4
  %conv131 = fpext float %141 to double
  %142 = load double, double* @ret_df, align 8
  %143 = load double, double* @lret_df, align 8
  %cmp132 = fcmp oeq double %142, %143
  %conv133 = zext i1 %cmp132 to i32
  %144 = load float, float* @x, align 4
  %145 = load float, float* @lx, align 4
  %cmp134 = fcmp oeq float %144, %145
  %conv135 = zext i1 %cmp134 to i32
  %and136 = and i32 %conv133, %conv135
  %call137 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %138, double %139, double %conv130, double %conv131, i32 %and136)
  call void @clear()
  store double 0x4084F3AB7AA25D8D, double* @ret_df, align 8
  store double 0x4114F671D2F1A9FC, double* @lxd, align 8
  %146 = load double, double* @lxd, align 8
  %call138 = call double @df_df(double %146)
  store double %call138, double* @lret_df, align 8
  %147 = load double, double* @ret_df, align 8
  %148 = load double, double* @lret_df, align 8
  %149 = load double, double* @xd, align 8
  %150 = load double, double* @lxd, align 8
  %151 = load double, double* @ret_df, align 8
  %152 = load double, double* @lret_df, align 8
  %cmp139 = fcmp oeq double %151, %152
  %conv140 = zext i1 %cmp139 to i32
  %153 = load double, double* @xd, align 8
  %154 = load double, double* @lxd, align 8
  %cmp141 = fcmp oeq double %153, %154
  %conv142 = zext i1 %cmp141 to i32
  %and143 = and i32 %conv140, %conv142
  %call144 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str1, i32 0, i32 0), double %147, double %148, double %149, double %150, i32 %and143)
  call void @clear()
  store double 6.781956e+03, double* @ret_df, align 8
  store float 4.445500e+03, float* @lx, align 4
  store float 0x4068ACCCC0000000, float* @ly, align 4
  %155 = load float, float* @lx, align 4
  %156 = load float, float* @ly, align 4
  %call145 = call double @df_sf_sf(float %155, float %156)
  store double %call145, double* @lret_df, align 8
  %157 = load double, double* @ret_df, align 8
  %158 = load double, double* @lret_df, align 8
  %159 = load float, float* @x, align 4
  %conv146 = fpext float %159 to double
  %160 = load float, float* @lx, align 4
  %conv147 = fpext float %160 to double
  %161 = load float, float* @y, align 4
  %conv148 = fpext float %161 to double
  %162 = load float, float* @ly, align 4
  %conv149 = fpext float %162 to double
  %163 = load double, double* @ret_df, align 8
  %164 = load double, double* @lret_df, align 8
  %cmp150 = fcmp oeq double %163, %164
  br i1 %cmp150, label %land.lhs.true152, label %land.end158

land.lhs.true152:                                 ; preds = %land.end122
  %165 = load float, float* @x, align 4
  %166 = load float, float* @lx, align 4
  %cmp153 = fcmp oeq float %165, %166
  br i1 %cmp153, label %land.rhs155, label %land.end158

land.rhs155:                                      ; preds = %land.lhs.true152
  %167 = load float, float* @y, align 4
  %168 = load float, float* @ly, align 4
  %cmp156 = fcmp oeq float %167, %168
  br label %land.end158

land.end158:                                      ; preds = %land.rhs155, %land.lhs.true152, %land.end122
  %169 = phi i1 [ false, %land.lhs.true152 ], [ false, %land.end122 ], [ %cmp156, %land.rhs155 ]
  %land.ext159 = zext i1 %169 to i32
  %call160 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %157, double %158, double %conv146, double %conv147, double %conv148, double %conv149, i32 %land.ext159)
  call void @clear()
  store double 1.889130e+05, double* @ret_df, align 8
  store float 9.111450e+05, float* @lx, align 4
  store double 0x4185320A58000000, double* @lyd, align 8
  %170 = load float, float* @lx, align 4
  %171 = load double, double* @lyd, align 8
  %call161 = call double @df_sf_df(float %170, double %171)
  store double %call161, double* @lret_df, align 8
  %172 = load double, double* @ret_df, align 8
  %173 = load double, double* @lret_df, align 8
  %174 = load float, float* @x, align 4
  %conv162 = fpext float %174 to double
  %175 = load float, float* @lx, align 4
  %conv163 = fpext float %175 to double
  %176 = load double, double* @yd, align 8
  %177 = load double, double* @lyd, align 8
  %178 = load double, double* @ret_df, align 8
  %179 = load double, double* @lret_df, align 8
  %cmp164 = fcmp oeq double %178, %179
  br i1 %cmp164, label %land.lhs.true166, label %land.end172

land.lhs.true166:                                 ; preds = %land.end158
  %180 = load float, float* @x, align 4
  %181 = load float, float* @lx, align 4
  %cmp167 = fcmp oeq float %180, %181
  br i1 %cmp167, label %land.rhs169, label %land.end172

land.rhs169:                                      ; preds = %land.lhs.true166
  %182 = load double, double* @yd, align 8
  %183 = load double, double* @lyd, align 8
  %cmp170 = fcmp oeq double %182, %183
  br label %land.end172

land.end172:                                      ; preds = %land.rhs169, %land.lhs.true166, %land.end158
  %184 = phi i1 [ false, %land.lhs.true166 ], [ false, %land.end158 ], [ %cmp170, %land.rhs169 ]
  %land.ext173 = zext i1 %184 to i32
  %call174 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %172, double %173, double %conv162, double %conv163, double %176, double %177, i32 %land.ext173)
  call void @clear()
  store double 0x418B2DB900000000, double* @ret_df, align 8
  store double 0x41B1EF2ED3000000, double* @lxd, align 8
  store float 1.244562e+06, float* @ly, align 4
  %185 = load double, double* @lxd, align 8
  %186 = load float, float* @ly, align 4
  %call175 = call double @df_df_sf(double %185, float %186)
  store double %call175, double* @lret_df, align 8
  %187 = load double, double* @ret_df, align 8
  %188 = load double, double* @lret_df, align 8
  %189 = load double, double* @xd, align 8
  %190 = load double, double* @lxd, align 8
  %191 = load float, float* @y, align 4
  %conv176 = fpext float %191 to double
  %192 = load float, float* @ly, align 4
  %conv177 = fpext float %192 to double
  %193 = load double, double* @ret_df, align 8
  %194 = load double, double* @lret_df, align 8
  %cmp178 = fcmp oeq double %193, %194
  br i1 %cmp178, label %land.lhs.true180, label %land.end186

land.lhs.true180:                                 ; preds = %land.end172
  %195 = load double, double* @xd, align 8
  %196 = load double, double* @lxd, align 8
  %cmp181 = fcmp oeq double %195, %196
  br i1 %cmp181, label %land.rhs183, label %land.end186

land.rhs183:                                      ; preds = %land.lhs.true180
  %197 = load float, float* @y, align 4
  %198 = load float, float* @ly, align 4
  %cmp184 = fcmp oeq float %197, %198
  br label %land.end186

land.end186:                                      ; preds = %land.rhs183, %land.lhs.true180, %land.end172
  %199 = phi i1 [ false, %land.lhs.true180 ], [ false, %land.end172 ], [ %cmp184, %land.rhs183 ]
  %land.ext187 = zext i1 %199 to i32
  %call188 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %187, double %188, double %189, double %190, double %conv176, double %conv177, i32 %land.ext187)
  call void @clear()
  store double 3.987721e+06, double* @ret_df, align 8
  store double 5.223560e+00, double* @lxd, align 8
  store double 0x40B7D37CC1A8AC5C, double* @lyd, align 8
  %200 = load double, double* @lxd, align 8
  %201 = load double, double* @lyd, align 8
  %call189 = call double @df_df_df(double %200, double %201)
  store double %call189, double* @lret_df, align 8
  %202 = load double, double* @ret_df, align 8
  %203 = load double, double* @lret_df, align 8
  %204 = load double, double* @xd, align 8
  %205 = load double, double* @lxd, align 8
  %206 = load double, double* @yd, align 8
  %207 = load double, double* @lyd, align 8
  %208 = load double, double* @ret_df, align 8
  %209 = load double, double* @lret_df, align 8
  %cmp190 = fcmp oeq double %208, %209
  br i1 %cmp190, label %land.lhs.true192, label %land.end198

land.lhs.true192:                                 ; preds = %land.end186
  %210 = load double, double* @xd, align 8
  %211 = load double, double* @lxd, align 8
  %cmp193 = fcmp oeq double %210, %211
  br i1 %cmp193, label %land.rhs195, label %land.end198

land.rhs195:                                      ; preds = %land.lhs.true192
  %212 = load double, double* @yd, align 8
  %213 = load double, double* @lyd, align 8
  %cmp196 = fcmp oeq double %212, %213
  br label %land.end198

land.end198:                                      ; preds = %land.rhs195, %land.lhs.true192, %land.end186
  %214 = phi i1 [ false, %land.lhs.true192 ], [ false, %land.end186 ], [ %cmp196, %land.rhs195 ]
  %land.ext199 = zext i1 %214 to i32
  %call200 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str2, i32 0, i32 0), double %202, double %203, double %204, double %205, double %206, double %207, i32 %land.ext199)
  call void @clear()
  store float 4.500000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  store float 7.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %call201 = call { float, float } @sc_v()
  %215 = extractvalue { float, float } %call201, 0
  %216 = extractvalue { float, float } %call201, 1
  store float %215, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  store float %216, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %ret_sc.real = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  %ret_sc.imag = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %conv202 = fpext float %ret_sc.real to double
  %conv203 = fpext float %ret_sc.imag to double
  %ret_sc.real204 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  %ret_sc.imag205 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %conv206 = fpext float %ret_sc.real204 to double
  %conv207 = fpext float %ret_sc.imag205 to double
  %lret_sc.real = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  %lret_sc.imag = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %conv208 = fpext float %lret_sc.real to double
  %conv209 = fpext float %lret_sc.imag to double
  %lret_sc.real210 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  %lret_sc.imag211 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %conv212 = fpext float %lret_sc.real210 to double
  %conv213 = fpext float %lret_sc.imag211 to double
  %ret_sc.real214 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  %ret_sc.imag215 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %lret_sc.real216 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  %lret_sc.imag217 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %cmp.r = fcmp oeq float %ret_sc.real214, %lret_sc.real216
  %cmp.i = fcmp oeq float %ret_sc.imag215, %lret_sc.imag217
  %and.ri = and i1 %cmp.r, %cmp.i
  %conv218 = zext i1 %and.ri to i32
  %call219 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str3, i32 0, i32 0), double %conv202, double %conv207, double %conv208, double %conv213, i32 %conv218)
  call void @clear()
  store float 0x3FF7A99300000000, float* @lx, align 4
  store float 4.500000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  store float 7.000000e+00, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %217 = load float, float* @lx, align 4
  %call220 = call { float, float } @sc_sf(float %217)
  %218 = extractvalue { float, float } %call220, 0
  %219 = extractvalue { float, float } %call220, 1
  store float %218, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  store float %219, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %ret_sc.real221 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  %ret_sc.imag222 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %conv223 = fpext float %ret_sc.real221 to double
  %conv224 = fpext float %ret_sc.imag222 to double
  %ret_sc.real225 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  %ret_sc.imag226 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %conv227 = fpext float %ret_sc.real225 to double
  %conv228 = fpext float %ret_sc.imag226 to double
  %lret_sc.real229 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  %lret_sc.imag230 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %conv231 = fpext float %lret_sc.real229 to double
  %conv232 = fpext float %lret_sc.imag230 to double
  %lret_sc.real233 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  %lret_sc.imag234 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %conv235 = fpext float %lret_sc.real233 to double
  %conv236 = fpext float %lret_sc.imag234 to double
  %220 = load float, float* @x, align 4
  %conv237 = fpext float %220 to double
  %221 = load float, float* @lx, align 4
  %conv238 = fpext float %221 to double
  %ret_sc.real239 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 0)
  %ret_sc.imag240 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @ret_sc, i32 0, i32 1)
  %lret_sc.real241 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 0)
  %lret_sc.imag242 = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @lret_sc, i32 0, i32 1)
  %cmp.r243 = fcmp oeq float %ret_sc.real239, %lret_sc.real241
  %cmp.i244 = fcmp oeq float %ret_sc.imag240, %lret_sc.imag242
  %and.ri245 = and i1 %cmp.r243, %cmp.i244
  br i1 %and.ri245, label %land.rhs247, label %land.end250

land.rhs247:                                      ; preds = %land.end198
  %222 = load float, float* @x, align 4
  %223 = load float, float* @lx, align 4
  %cmp248 = fcmp oeq float %222, %223
  br label %land.end250

land.end250:                                      ; preds = %land.rhs247, %land.end198
  %224 = phi i1 [ false, %land.end198 ], [ %cmp248, %land.rhs247 ]
  %land.ext251 = zext i1 %224 to i32
  %call252 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str4, i32 0, i32 0), double %conv223, double %conv228, double %conv231, double %conv236, double %conv237, double %conv238, i32 %land.ext251)
  call void @clear()
  store double 1.234500e+03, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  store double 7.677000e+03, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %call253 = call { double, double } @dc_v()
  %225 = extractvalue { double, double } %call253, 0
  %226 = extractvalue { double, double } %call253, 1
  store double %225, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  store double %226, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %ret_dc.real = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  %ret_dc.imag = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %ret_dc.real254 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  %ret_dc.imag255 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %lret_dc.real = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  %lret_dc.imag = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %lret_dc.real256 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  %lret_dc.imag257 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %ret_dc.real258 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  %ret_dc.imag259 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %lret_dc.real260 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  %lret_dc.imag261 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %cmp.r262 = fcmp oeq double %ret_dc.real258, %lret_dc.real260
  %cmp.i263 = fcmp oeq double %ret_dc.imag259, %lret_dc.imag261
  %and.ri264 = and i1 %cmp.r262, %cmp.i263
  %conv265 = zext i1 %and.ri264 to i32
  %call266 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str3, i32 0, i32 0), double %ret_dc.real, double %ret_dc.imag255, double %lret_dc.real, double %lret_dc.imag257, i32 %conv265)
  call void @clear()
  store double 0x40AAF6F532617C1C, double* @lxd, align 8
  store double 4.444500e+03, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  store double 7.888000e+03, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %227 = load float, float* @lx, align 4
  %call267 = call { double, double } @dc_sf(float %227)
  %228 = extractvalue { double, double } %call267, 0
  %229 = extractvalue { double, double } %call267, 1
  store double %228, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  store double %229, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %ret_dc.real268 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  %ret_dc.imag269 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %ret_dc.real270 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  %ret_dc.imag271 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %lret_dc.real272 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  %lret_dc.imag273 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %lret_dc.real274 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  %lret_dc.imag275 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %230 = load float, float* @x, align 4
  %conv276 = fpext float %230 to double
  %231 = load float, float* @lx, align 4
  %conv277 = fpext float %231 to double
  %ret_dc.real278 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 0)
  %ret_dc.imag279 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @ret_dc, i32 0, i32 1)
  %lret_dc.real280 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 0)
  %lret_dc.imag281 = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @lret_dc, i32 0, i32 1)
  %cmp.r282 = fcmp oeq double %ret_dc.real278, %lret_dc.real280
  %cmp.i283 = fcmp oeq double %ret_dc.imag279, %lret_dc.imag281
  %and.ri284 = and i1 %cmp.r282, %cmp.i283
  br i1 %and.ri284, label %land.rhs286, label %land.end289

land.rhs286:                                      ; preds = %land.end250
  %232 = load float, float* @x, align 4
  %233 = load float, float* @lx, align 4
  %cmp287 = fcmp oeq float %232, %233
  br label %land.end289

land.end289:                                      ; preds = %land.rhs286, %land.end250
  %234 = phi i1 [ false, %land.end250 ], [ %cmp287, %land.rhs286 ]
  %land.ext290 = zext i1 %234 to i32
  %call291 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str4, i32 0, i32 0), double %ret_dc.real268, double %ret_dc.imag271, double %lret_dc.real272, double %lret_dc.imag275, double %conv276, double %conv277, i32 %land.ext290)
  %235 = load i32, i32* %retval
  ret i32 %235
}

declare void @v_sf(float) #1
; stel: .section .mips16.call.fp.v_sf,"ax",@progbits
; stel: .ent __call_stub_fp_v_sf
; stel: mtc1 $4, $f12
; stel: lui $25, %hi(v_sf)
; stel: addiu $25, $25, %lo(v_sf)
; stel: jr $25
; stel: .end __call_stub_fp_v_sf

declare i32 @printf(i8*, ...) #1

declare void @v_df(double) #1
; stel: .section .mips16.call.fp.v_df,"ax",@progbits
; stel: .ent __call_stub_fp_v_df
; stel: #APP
; stel: .set reorder
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: lui $25, %hi(v_df)
; stel: addiu $25, $25, %lo(v_df)
; stel: jr $25
; stel: .end __call_stub_fp_v_df

declare void @v_sf_sf(float, float) #1
; stel: .section .mips16.call.fp.v_sf_sf,"ax",@progbits
; stel: .ent __call_stub_fp_v_sf_sf
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f14
; stel: lui $25, %hi(v_sf_sf)
; stel: addiu $25, $25, %lo(v_sf_sf)
; stel: jr $25
; stel: .end __call_stub_fp_v_sf_sf

declare void @v_sf_df(float, double) #1
; stel: .section .mips16.call.fp.v_sf_df,"ax",@progbits
; stel: .ent __call_stub_fp_v_sf_df
; stel: mtc1 $4, $f12
; stel: mtc1 $6, $f14
; stel: mtc1 $7, $f15
; stel: lui $25, %hi(v_sf_df)
; stel: addiu $25, $25, %lo(v_sf_df)
; stel: jr $25
; stel: .end __call_stub_fp_v_sf_df

declare void @v_df_sf(double, float) #1
; stel: .section .mips16.call.fp.v_df_sf,"ax",@progbits
; stel: .ent __call_stub_fp_v_df_sf
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: mtc1 $6, $f14
; stel: lui $25, %hi(v_df_sf)
; stel: addiu $25, $25, %lo(v_df_sf)
; stel: jr $25
; stel: .end __call_stub_fp_v_df_sf

declare void @v_df_df(double, double) #1
; stel: .section .mips16.call.fp.v_df_df,"ax",@progbits
; stel: .ent __call_stub_fp_v_df_df
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: mtc1 $6, $f14
; stel: mtc1 $7, $f15
; stel: lui $25, %hi(v_df_df)
; stel: addiu $25, $25, %lo(v_df_df)
; stel: jr $25
; stel: .end __call_stub_fp_v_df_df

declare float @sf_v() #1
; stel: .section .mips16.call.fp.sf_v,"ax",@progbits
; stel: .ent __call_stub_fp_sf_v
; stel: move $18, $31
; stel: jal sf_v
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_v

declare float @sf_sf(float) #1
; stel: .section .mips16.call.fp.sf_sf,"ax",@progbits
; stel: .ent __call_stub_fp_sf_sf
; stel: mtc1 $4, $f12
; stel: move $18, ${{31|ra}}
; stel: jal sf_sf
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_sf

declare float @sf_df(double) #1
; stel: .section .mips16.call.fp.sf_df,"ax",@progbits
; stel: .ent __call_stub_fp_sf_df
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: move $18, ${{31|ra}}
; stel: jal sf_df
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_df

declare float @sf_sf_sf(float, float) #1
; stel: .section .mips16.call.fp.sf_sf_sf,"ax",@progbits
; stel: .ent __call_stub_fp_sf_sf_sf
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f14
; stel: move $18, ${{31|ra}}
; stel: jal sf_sf_sf
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_sf_sf

declare float @sf_sf_df(float, double) #1
; stel: .section .mips16.call.fp.sf_sf_df,"ax",@progbits
; stel: .ent __call_stub_fp_sf_sf_df
; stel: mtc1 $4, $f12
; stel: mtc1 $6, $f14
; stel: mtc1 $7, $f15
; stel: move $18, ${{31|ra}}
; stel: jal sf_sf_df
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_sf_df

declare float @sf_df_sf(double, float) #1
; stel: .section .mips16.call.fp.sf_df_sf,"ax",@progbits
; stel: .ent __call_stub_fp_sf_df_sf
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: mtc1 $6, $f14
; stel: move $18, ${{31|ra}}
; stel: jal sf_df_sf
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_df_sf

declare float @sf_df_df(double, double) #1
; stel: .section .mips16.call.fp.sf_df_df,"ax",@progbits
; stel: .ent __call_stub_fp_sf_df_df
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: mtc1 $6, $f14
; stel: mtc1 $7, $f15
; stel: move $18, ${{31|ra}}
; stel: jal sf_df_df
; stel: mfc1 $2, $f0
; stel: jr $18
; stel: .end __call_stub_fp_sf_df_df

declare double @df_v() #1
; stel: .section .mips16.call.fp.df_v,"ax",@progbits
; stel: .ent __call_stub_fp_df_v
; stel: move $18, $31
; stel: jal df_v
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_v

declare double @df_sf(float) #1
; stel: .section .mips16.call.fp.df_sf,"ax",@progbits
; stel: .ent __call_stub_fp_df_sf
; stel: mtc1 $4, $f12
; stel: move $18, ${{31|ra}}
; stel: jal df_sf
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_sf

declare double @df_df(double) #1
; stel: .section .mips16.call.fp.df_df,"ax",@progbits
; stel: .ent __call_stub_fp_df_df
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: move $18, ${{31|ra}}
; stel: jal df_df
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_df

declare double @df_sf_sf(float, float) #1
; stel: .section .mips16.call.fp.df_sf_sf,"ax",@progbits
; stel: .ent __call_stub_fp_df_sf_sf
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f14
; stel: move $18, ${{31|ra}}
; stel: jal df_sf_sf
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_sf_sf

declare double @df_sf_df(float, double) #1
; stel: .section .mips16.call.fp.df_sf_df,"ax",@progbits
; stel: .ent __call_stub_fp_df_sf_df
; stel: mtc1 $4, $f12
; stel: mtc1 $6, $f14
; stel: mtc1 $7, $f15
; stel: move $18, ${{31|ra}}
; stel: jal df_sf_df
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_sf_df

declare double @df_df_sf(double, float) #1
; stel: .section .mips16.call.fp.df_df_sf,"ax",@progbits
; stel: .ent __call_stub_fp_df_df_sf
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: mtc1 $6, $f14
; stel: move $18, ${{31|ra}}
; stel: jal df_df_sf
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_df_sf

declare double @df_df_df(double, double) #1
; stel: .section .mips16.call.fp.df_df_df,"ax",@progbits
; stel: .ent __call_stub_fp_df_df_df
; stel: mtc1 $4, $f12
; stel: mtc1 $5, $f13
; stel: mtc1 $6, $f14
; stel: mtc1 $7, $f15
; stel: move $18, ${{31|ra}}
; stel: jal df_df_df
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_df_df_df

declare { float, float } @sc_v() #1
; stel: .section .mips16.call.fp.sc_v,"ax",@progbits
; stel: .ent __call_stub_fp_sc_v
; stel: move $18, $31
; stel: jal sc_v
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f2
; stel: jr $18
; stel: .end __call_stub_fp_sc_v

declare { float, float } @sc_sf(float) #1
; stel: .section .mips16.call.fp.sc_sf,"ax",@progbits
; stel: .ent __call_stub_fp_sc_sf
; stel: mtc1 $4, $f12
; stel: move $18, ${{31|ra}}
; stel: jal sc_sf
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f2
; stel: jr $18
; stel: .end __call_stub_fp_sc_sf

declare { double, double } @dc_v() #1
; stel: .section .mips16.call.fp.dc_v,"ax",@progbits
; stel: .ent __call_stub_fp_dc_v
; stel: move $18, $31
; stel: jal dc_v
; stel: mfc1 $4, $f2
; stel: mfc1 $5, $f3
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_dc_v

declare { double, double } @dc_sf(float) #1
; stel: .section .mips16.call.fp.dc_sf,"ax",@progbits
; stel: .ent __call_stub_fp_dc_sf
; stel: mtc1 $4, $f12
; stel: move $18, ${{31|ra}}
; stel: jal dc_sf
; stel: mfc1 $4, $f2
; stel: mfc1 $5, $f3
; stel: mfc1 $2, $f0
; stel: mfc1 $3, $f1
; stel: jr $18
; stel: .end __call_stub_fp_dc_sf

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
