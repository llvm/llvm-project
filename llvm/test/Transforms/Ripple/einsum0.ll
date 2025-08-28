; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S %s | FileCheck %s --implicit-check-not="warning:"

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z22nkctv_nkvw_nctw_dram_0PKhS0_Phfififimmmmmm(ptr noundef %aptr, ptr noundef %bptr, ptr noundef %cptr, float noundef %a_scale, i32 noundef %a_offset, float noundef %b_scale, i32 noundef %b_offset, float noundef %c_scale, i32 noundef %c_offset, i64 noundef %N, i64 noundef %K, i64 noundef %C, i64 noundef %T, i64 noundef %V, i64 noundef %W) #0 {
entry:
  %retval.i206 = alloca float, align 4
  %x.addr.i207 = alloca float, align 4
  %lb.addr.i208 = alloca float, align 4
  %ub.addr.i209 = alloca float, align 4
  %retval.i195 = alloca float, align 4
  %x.addr.i196 = alloca float, align 4
  %lb.addr.i197 = alloca float, align 4
  %ub.addr.i198 = alloca float, align 4
  %retval.i = alloca float, align 4
  %x.addr.i = alloca float, align 4
  %lb.addr.i = alloca float, align 4
  %ub.addr.i = alloca float, align 4
  %aptr.addr = alloca ptr, align 8
  %bptr.addr = alloca ptr, align 8
  %cptr.addr = alloca ptr, align 8
  %a_scale.addr = alloca float, align 4
  %a_offset.addr = alloca i32, align 4
  %b_scale.addr = alloca float, align 4
  %b_offset.addr = alloca i32, align 4
  %c_scale.addr = alloca float, align 4
  %c_offset.addr = alloca i32, align 4
  %N.addr = alloca i64, align 8
  %K.addr = alloca i64, align 8
  %C.addr = alloca i64, align 8
  %T.addr = alloca i64, align 8
  %V.addr = alloca i64, align 8
  %W.addr = alloca i64, align 8
  %v0 = alloca i64, align 8
  %v1 = alloca i64, align 8
  %nv0 = alloca i64, align 8
  %nv1 = alloca i64, align 8
  %n = alloca i64, align 8
  %c = alloca i64, align 8
  %t = alloca i64, align 8
  %w = alloca i64, align 8
  %acc = alloca float, align 4
  %k = alloca i64, align 8
  %v = alloca i64, align 8
  %acc_clamped = alloca float, align 4
  %w63 = alloca i64, align 8
  %acc68 = alloca float, align 4
  %k69 = alloca i64, align 8
  %v73 = alloca i64, align 8
  %acc_clamped110 = alloca float, align 4
  %acc131 = alloca float, align 4
  %k132 = alloca i64, align 8
  %v136 = alloca i64, align 8
  %acc_clamped173 = alloca float, align 4
  store ptr %aptr, ptr %aptr.addr, align 8
  store ptr %bptr, ptr %bptr.addr, align 8
  store ptr %cptr, ptr %cptr.addr, align 8
  store float %a_scale, ptr %a_scale.addr, align 4
  store i32 %a_offset, ptr %a_offset.addr, align 4
  store float %b_scale, ptr %b_scale.addr, align 4
  store i32 %b_offset, ptr %b_offset.addr, align 4
  store float %c_scale, ptr %c_scale.addr, align 4
  store i32 %c_offset, ptr %c_offset.addr, align 4
  store i64 %N, ptr %N.addr, align 8
  store i64 %K, ptr %K.addr, align 8
  store i64 %C, ptr %C.addr, align 8
  store i64 %T, ptr %T.addr, align 8
  store i64 %V, ptr %V.addr, align 8
  store i64 %W, ptr %W.addr, align 8
  %BS = call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %0 = call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  store i64 %0, ptr %v0, align 8
  %1 = call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 1)
  store i64 %1, ptr %v1, align 8
  %2 = call i64 @llvm.ripple.block.getsize.i64(ptr %BS, i64 0)
  store i64 %2, ptr %nv0, align 8
  %3 = call i64 @llvm.ripple.block.getsize.i64(ptr %BS, i64 1)
  store i64 %3, ptr %nv1, align 8
  store i64 0, ptr %n, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc192, %entry
  %4 = load i64, ptr %n, align 8
  %5 = load i64, ptr %N.addr, align 8
  %cmp = icmp ult i64 %4, %5
  br i1 %cmp, label %for.body, label %for.end194

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %c, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc189, %for.body
  %6 = load i64, ptr %c, align 8
  %7 = load i64, ptr %C.addr, align 8
  %cmp2 = icmp ult i64 %6, %7
  br i1 %cmp2, label %for.body3, label %for.end191

for.body3:                                        ; preds = %for.cond1
  store i64 0, ptr %t, align 8
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc58, %for.body3
  %8 = load i64, ptr %t, align 8
  %9 = load i64, ptr %nv1, align 8
  %add = add i64 %8, %9
  %10 = load i64, ptr %T.addr, align 8
  %cmp5 = icmp ule i64 %add, %10
  br i1 %cmp5, label %for.body6, label %for.end60

for.body6:                                        ; preds = %for.cond4
  store i64 0, ptr %w, align 8
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc55, %for.body6
  %11 = load i64, ptr %w, align 8
  %12 = load i64, ptr %nv0, align 8
  %add8 = add i64 %11, %12
  %13 = load i64, ptr %W.addr, align 8
  %cmp9 = icmp ule i64 %add8, %13
  br i1 %cmp9, label %for.body10, label %for.end57

for.body10:                                       ; preds = %for.cond7
  store float 0.000000e+00, ptr %acc, align 4
  store i64 0, ptr %k, align 8
  br label %for.cond11

for.cond11:                                       ; preds = %for.inc40, %for.body10
  %14 = load i64, ptr %k, align 8
  %15 = load i64, ptr %K.addr, align 8
  %cmp12 = icmp ult i64 %14, %15
  br i1 %cmp12, label %for.body13, label %for.end42

for.body13:                                       ; preds = %for.cond11
  store i64 0, ptr %v, align 8
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc, %for.body13
  %16 = load i64, ptr %v, align 8
  %17 = load i64, ptr %V.addr, align 8
  %cmp15 = icmp ult i64 %16, %17
  br i1 %cmp15, label %for.body16, label %for.end

for.body16:                                       ; preds = %for.cond14
  %18 = load float, ptr %a_scale.addr, align 4
  %19 = load ptr, ptr %aptr.addr, align 8
  %20 = load i64, ptr %v, align 8
  %21 = load i64, ptr %V.addr, align 8
  %22 = load i64, ptr %t, align 8
  %23 = load i64, ptr %v1, align 8
  %add17 = add i64 %22, %23
  %24 = load i64, ptr %T.addr, align 8
  %25 = load i64, ptr %c, align 8
  %26 = load i64, ptr %C.addr, align 8
  %27 = load i64, ptr %k, align 8
  %28 = load i64, ptr %K.addr, align 8
  %29 = load i64, ptr %n, align 8
  %mul = mul i64 %28, %29
  %add18 = add i64 %27, %mul
  %mul19 = mul i64 %26, %add18
  %add20 = add i64 %25, %mul19
  %mul21 = mul i64 %24, %add20
  %add22 = add i64 %add17, %mul21
  %mul23 = mul i64 %21, %add22
  %add24 = add i64 %20, %mul23
  %arrayidx = getelementptr inbounds nuw i8, ptr %19, i64 %add24
  %30 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %30 to i32
  %31 = load i32, ptr %a_offset.addr, align 4
  %sub = sub nsw i32 %conv, %31
  %conv25 = sitofp i32 %sub to float
  %mul26 = fmul float %18, %conv25
  %32 = load float, ptr %b_scale.addr, align 4
  %33 = load ptr, ptr %bptr.addr, align 8
  %34 = load i64, ptr %w, align 8
  %35 = load i64, ptr %v0, align 8
  %add27 = add i64 %34, %35
  %36 = load i64, ptr %W.addr, align 8
  %37 = load i64, ptr %v, align 8
  %38 = load i64, ptr %V.addr, align 8
  %39 = load i64, ptr %k, align 8
  %40 = load i64, ptr %K.addr, align 8
  %41 = load i64, ptr %n, align 8
  %mul28 = mul i64 %40, %41
  %add29 = add i64 %39, %mul28
  %mul30 = mul i64 %38, %add29
  %add31 = add i64 %37, %mul30
  %mul32 = mul i64 %36, %add31
  %add33 = add i64 %add27, %mul32
  %arrayidx34 = getelementptr inbounds nuw i8, ptr %33, i64 %add33
  %42 = load i8, ptr %arrayidx34, align 1
  %conv35 = zext i8 %42 to i32
  %43 = load i32, ptr %b_offset.addr, align 4
  %sub36 = sub nsw i32 %conv35, %43
  %conv37 = sitofp i32 %sub36 to float
  %mul38 = fmul float %32, %conv37
  %44 = load float, ptr %acc, align 4
  %45 = call float @llvm.fmuladd.f32(float %mul26, float %mul38, float %44)
  store float %45, ptr %acc, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body16
  %46 = load i64, ptr %v, align 8
  %inc = add i64 %46, 1
  store i64 %inc, ptr %v, align 8
  br label %for.cond14, !llvm.loop !6

for.end:                                          ; preds = %for.cond14
  br label %for.inc40

for.inc40:                                        ; preds = %for.end
  %47 = load i64, ptr %k, align 8
  %inc41 = add i64 %47, 1
  store i64 %inc41, ptr %k, align 8
  br label %for.cond11, !llvm.loop !8

for.end42:                                        ; preds = %for.cond11
  %48 = load float, ptr %acc, align 4
  %49 = load float, ptr %c_scale.addr, align 4
  %div = fdiv float %48, %49
  %50 = load i32, ptr %c_offset.addr, align 4
  %conv43 = sitofp i32 %50 to float
  %add44 = fadd float %div, %conv43
  store float %add44, ptr %x.addr.i207, align 4
  store float 0.000000e+00, ptr %lb.addr.i208, align 4
  store float 2.550000e+02, ptr %ub.addr.i209, align 4
  %51 = load float, ptr %x.addr.i207, align 4
  %52 = load float, ptr %lb.addr.i208, align 4
  %cmp.i210 = fcmp olt float %51, %52
  br i1 %cmp.i210, label %if.then.i215, label %if.end.i211

if.then.i215:                                     ; preds = %for.end42
  %53 = load float, ptr %lb.addr.i208, align 4
  store float %53, ptr %retval.i206, align 4
  br label %_ZL5clampfff.exit216

if.end.i211:                                      ; preds = %for.end42
  %54 = load float, ptr %x.addr.i207, align 4
  %55 = load float, ptr %ub.addr.i209, align 4
  %cmp1.i212 = fcmp ogt float %54, %55
  br i1 %cmp1.i212, label %if.then2.i214, label %if.end3.i213

if.then2.i214:                                    ; preds = %if.end.i211
  %56 = load float, ptr %ub.addr.i209, align 4
  store float %56, ptr %retval.i206, align 4
  br label %_ZL5clampfff.exit216

if.end3.i213:                                     ; preds = %if.end.i211
  %57 = load float, ptr %x.addr.i207, align 4
  store float %57, ptr %retval.i206, align 4
  br label %_ZL5clampfff.exit216

_ZL5clampfff.exit216:                             ; preds = %if.then.i215, %if.then2.i214, %if.end3.i213
  %58 = load float, ptr %retval.i206, align 4
  store float %58, ptr %acc_clamped, align 4
  %59 = load float, ptr %acc_clamped, align 4
  %conv45 = fptoui float %59 to i8
  %60 = load ptr, ptr %cptr.addr, align 8
  %61 = load i64, ptr %w, align 8
  %62 = load i64, ptr %v0, align 8
  %add46 = add i64 %61, %62
  %63 = load i64, ptr %W.addr, align 8
  %64 = load i64, ptr %t, align 8
  %65 = load i64, ptr %v1, align 8
  %add47 = add i64 %64, %65
  %66 = load i64, ptr %T.addr, align 8
  %67 = load i64, ptr %c, align 8
  %68 = load i64, ptr %C.addr, align 8
  %69 = load i64, ptr %n, align 8
  %mul48 = mul i64 %68, %69
  %add49 = add i64 %67, %mul48
  %mul50 = mul i64 %66, %add49
  %add51 = add i64 %add47, %mul50
  %mul52 = mul i64 %63, %add51
  %add53 = add i64 %add46, %mul52
  %arrayidx54 = getelementptr inbounds nuw i8, ptr %60, i64 %add53
  store i8 %conv45, ptr %arrayidx54, align 1
  br label %for.inc55

for.inc55:                                        ; preds = %_ZL5clampfff.exit216
  %70 = load i64, ptr %nv0, align 8
  %71 = load i64, ptr %w, align 8
  %add56 = add i64 %71, %70
  store i64 %add56, ptr %w, align 8
  br label %for.cond7, !llvm.loop !9

for.end57:                                        ; preds = %for.cond7
  br label %for.inc58

for.inc58:                                        ; preds = %for.end57
  %72 = load i64, ptr %nv1, align 8
  %73 = load i64, ptr %t, align 8
  %add59 = add i64 %73, %72
  store i64 %add59, ptr %t, align 8
  br label %for.cond4, !llvm.loop !10

for.end60:                                        ; preds = %for.cond4
  %74 = load i64, ptr %t, align 8
  %75 = load i64, ptr %v1, align 8
  %add61 = add i64 %74, %75
  %76 = load i64, ptr %T.addr, align 8
  %cmp62 = icmp ult i64 %add61, %76
  br i1 %cmp62, label %if.then, label %if.end188

if.then:                                          ; preds = %for.end60
  store i64 0, ptr %w63, align 8
  br label %for.cond64

for.cond64:                                       ; preds = %for.inc125, %if.then
  %77 = load i64, ptr %w63, align 8
  %78 = load i64, ptr %nv0, align 8
  %add65 = add i64 %77, %78
  %79 = load i64, ptr %W.addr, align 8
  %cmp66 = icmp ule i64 %add65, %79
  br i1 %cmp66, label %for.body67, label %for.end127

for.body67:                                       ; preds = %for.cond64
  store float 0.000000e+00, ptr %acc68, align 4
  store i64 0, ptr %k69, align 8
  br label %for.cond70

for.cond70:                                       ; preds = %for.inc107, %for.body67
  %80 = load i64, ptr %k69, align 8
  %81 = load i64, ptr %K.addr, align 8
  %cmp71 = icmp ult i64 %80, %81
  br i1 %cmp71, label %for.body72, label %for.end109

for.body72:                                       ; preds = %for.cond70
  store i64 0, ptr %v73, align 8
  br label %for.cond74

for.cond74:                                       ; preds = %for.inc104, %for.body72
  %82 = load i64, ptr %v73, align 8
  %83 = load i64, ptr %V.addr, align 8
  %cmp75 = icmp ult i64 %82, %83
  br i1 %cmp75, label %for.body76, label %for.end106

for.body76:                                       ; preds = %for.cond74
  %84 = load float, ptr %a_scale.addr, align 4
  %85 = load ptr, ptr %aptr.addr, align 8
  %86 = load i64, ptr %v73, align 8
  %87 = load i64, ptr %V.addr, align 8
  %88 = load i64, ptr %t, align 8
  %89 = load i64, ptr %v1, align 8
  %add77 = add i64 %88, %89
  %90 = load i64, ptr %T.addr, align 8
  %91 = load i64, ptr %c, align 8
  %92 = load i64, ptr %C.addr, align 8
  %93 = load i64, ptr %k69, align 8
  %94 = load i64, ptr %K.addr, align 8
  %95 = load i64, ptr %n, align 8
  %mul78 = mul i64 %94, %95
  %add79 = add i64 %93, %mul78
  %mul80 = mul i64 %92, %add79
  %add81 = add i64 %91, %mul80
  %mul82 = mul i64 %90, %add81
  %add83 = add i64 %add77, %mul82
  %mul84 = mul i64 %87, %add83
  %add85 = add i64 %86, %mul84
  %arrayidx86 = getelementptr inbounds nuw i8, ptr %85, i64 %add85
  %96 = load i8, ptr %arrayidx86, align 1
  %conv87 = zext i8 %96 to i32
  %97 = load i32, ptr %a_offset.addr, align 4
  %sub88 = sub nsw i32 %conv87, %97
  %conv89 = sitofp i32 %sub88 to float
  %mul90 = fmul float %84, %conv89
  %98 = load float, ptr %b_scale.addr, align 4
  %99 = load ptr, ptr %bptr.addr, align 8
  %100 = load i64, ptr %w63, align 8
  %101 = load i64, ptr %v0, align 8
  %add91 = add i64 %100, %101
  %102 = load i64, ptr %W.addr, align 8
  %103 = load i64, ptr %v73, align 8
  %104 = load i64, ptr %V.addr, align 8
  %105 = load i64, ptr %k69, align 8
  %106 = load i64, ptr %K.addr, align 8
  %107 = load i64, ptr %n, align 8
  %mul92 = mul i64 %106, %107
  %add93 = add i64 %105, %mul92
  %mul94 = mul i64 %104, %add93
  %add95 = add i64 %103, %mul94
  %mul96 = mul i64 %102, %add95
  %add97 = add i64 %add91, %mul96
  %arrayidx98 = getelementptr inbounds nuw i8, ptr %99, i64 %add97
  %108 = load i8, ptr %arrayidx98, align 1
  %conv99 = zext i8 %108 to i32
  %109 = load i32, ptr %b_offset.addr, align 4
  %sub100 = sub nsw i32 %conv99, %109
  %conv101 = sitofp i32 %sub100 to float
  %mul102 = fmul float %98, %conv101
  %110 = load float, ptr %acc68, align 4
  %111 = call float @llvm.fmuladd.f32(float %mul90, float %mul102, float %110)
  store float %111, ptr %acc68, align 4
  br label %for.inc104

for.inc104:                                       ; preds = %for.body76
  %112 = load i64, ptr %v73, align 8
  %inc105 = add i64 %112, 1
  store i64 %inc105, ptr %v73, align 8
  br label %for.cond74, !llvm.loop !11

for.end106:                                       ; preds = %for.cond74
  br label %for.inc107

for.inc107:                                       ; preds = %for.end106
  %113 = load i64, ptr %k69, align 8
  %inc108 = add i64 %113, 1
  store i64 %inc108, ptr %k69, align 8
  br label %for.cond70, !llvm.loop !12

for.end109:                                       ; preds = %for.cond70
  %114 = load float, ptr %acc68, align 4
  %115 = load float, ptr %c_scale.addr, align 4
  %div111 = fdiv float %114, %115
  %116 = load i32, ptr %c_offset.addr, align 4
  %conv112 = sitofp i32 %116 to float
  %add113 = fadd float %div111, %conv112
  store float %add113, ptr %x.addr.i196, align 4
  store float 0.000000e+00, ptr %lb.addr.i197, align 4
  store float 2.550000e+02, ptr %ub.addr.i198, align 4
  %117 = load float, ptr %x.addr.i196, align 4
  %118 = load float, ptr %lb.addr.i197, align 4
  %cmp.i199 = fcmp olt float %117, %118
  br i1 %cmp.i199, label %if.then.i204, label %if.end.i200

if.then.i204:                                     ; preds = %for.end109
  %119 = load float, ptr %lb.addr.i197, align 4
  store float %119, ptr %retval.i195, align 4
  br label %_ZL5clampfff.exit205

if.end.i200:                                      ; preds = %for.end109
  %120 = load float, ptr %x.addr.i196, align 4
  %121 = load float, ptr %ub.addr.i198, align 4
  %cmp1.i201 = fcmp ogt float %120, %121
  br i1 %cmp1.i201, label %if.then2.i203, label %if.end3.i202

if.then2.i203:                                    ; preds = %if.end.i200
  %122 = load float, ptr %ub.addr.i198, align 4
  store float %122, ptr %retval.i195, align 4
  br label %_ZL5clampfff.exit205

if.end3.i202:                                     ; preds = %if.end.i200
  %123 = load float, ptr %x.addr.i196, align 4
  store float %123, ptr %retval.i195, align 4
  br label %_ZL5clampfff.exit205

_ZL5clampfff.exit205:                             ; preds = %if.then.i204, %if.then2.i203, %if.end3.i202
  %124 = load float, ptr %retval.i195, align 4
  store float %124, ptr %acc_clamped110, align 4
  %125 = load float, ptr %acc_clamped110, align 4
  %conv115 = fptoui float %125 to i8
  %126 = load ptr, ptr %cptr.addr, align 8
  %127 = load i64, ptr %w63, align 8
  %128 = load i64, ptr %v0, align 8
  %add116 = add i64 %127, %128
  %129 = load i64, ptr %W.addr, align 8
  %130 = load i64, ptr %t, align 8
  %131 = load i64, ptr %v1, align 8
  %add117 = add i64 %130, %131
  %132 = load i64, ptr %T.addr, align 8
  %133 = load i64, ptr %c, align 8
  %134 = load i64, ptr %C.addr, align 8
  %135 = load i64, ptr %n, align 8
  %mul118 = mul i64 %134, %135
  %add119 = add i64 %133, %mul118
  %mul120 = mul i64 %132, %add119
  %add121 = add i64 %add117, %mul120
  %mul122 = mul i64 %129, %add121
  %add123 = add i64 %add116, %mul122
  %arrayidx124 = getelementptr inbounds nuw i8, ptr %126, i64 %add123
  store i8 %conv115, ptr %arrayidx124, align 1
  br label %for.inc125

for.inc125:                                       ; preds = %_ZL5clampfff.exit205
  %136 = load i64, ptr %nv0, align 8
  %137 = load i64, ptr %w63, align 8
  %add126 = add i64 %137, %136
  store i64 %add126, ptr %w63, align 8
  br label %for.cond64, !llvm.loop !13

for.end127:                                       ; preds = %for.cond64
  %138 = load i64, ptr %w63, align 8
  %139 = load i64, ptr %v0, align 8
  %add128 = add i64 %138, %139
  %140 = load i64, ptr %W.addr, align 8
  %cmp129 = icmp ult i64 %add128, %140
  br i1 %cmp129, label %if.then130, label %if.end

if.then130:                                       ; preds = %for.end127
  store float 0.000000e+00, ptr %acc131, align 4
  store i64 0, ptr %k132, align 8
  br label %for.cond133

for.cond133:                                      ; preds = %for.inc170, %if.then130
  %141 = load i64, ptr %k132, align 8
  %142 = load i64, ptr %K.addr, align 8
  %cmp134 = icmp ult i64 %141, %142
  br i1 %cmp134, label %for.body135, label %for.end172

for.body135:                                      ; preds = %for.cond133
  store i64 0, ptr %v136, align 8
  br label %for.cond137

for.cond137:                                      ; preds = %for.inc167, %for.body135
  %143 = load i64, ptr %v136, align 8
  %144 = load i64, ptr %V.addr, align 8
  %cmp138 = icmp ult i64 %143, %144
  br i1 %cmp138, label %for.body139, label %for.end169

for.body139:                                      ; preds = %for.cond137
  %145 = load float, ptr %a_scale.addr, align 4
  %146 = load ptr, ptr %aptr.addr, align 8
  %147 = load i64, ptr %v136, align 8
  %148 = load i64, ptr %V.addr, align 8
  %149 = load i64, ptr %t, align 8
  %150 = load i64, ptr %v1, align 8
  %add140 = add i64 %149, %150
  %151 = load i64, ptr %T.addr, align 8
  %152 = load i64, ptr %c, align 8
  %153 = load i64, ptr %C.addr, align 8
  %154 = load i64, ptr %k132, align 8
  %155 = load i64, ptr %K.addr, align 8
  %156 = load i64, ptr %n, align 8
  %mul141 = mul i64 %155, %156
  %add142 = add i64 %154, %mul141
  %mul143 = mul i64 %153, %add142
  %add144 = add i64 %152, %mul143
  %mul145 = mul i64 %151, %add144
  %add146 = add i64 %add140, %mul145
  %mul147 = mul i64 %148, %add146
  %add148 = add i64 %147, %mul147
  %arrayidx149 = getelementptr inbounds nuw i8, ptr %146, i64 %add148
  %157 = load i8, ptr %arrayidx149, align 1
  %conv150 = zext i8 %157 to i32
  %158 = load i32, ptr %a_offset.addr, align 4
  %sub151 = sub nsw i32 %conv150, %158
  %conv152 = sitofp i32 %sub151 to float
  %mul153 = fmul float %145, %conv152
  %159 = load float, ptr %b_scale.addr, align 4
  %160 = load ptr, ptr %bptr.addr, align 8
  %161 = load i64, ptr %w63, align 8
  %162 = load i64, ptr %v0, align 8
  %add154 = add i64 %161, %162
  %163 = load i64, ptr %W.addr, align 8
  %164 = load i64, ptr %v136, align 8
  %165 = load i64, ptr %V.addr, align 8
  %166 = load i64, ptr %k132, align 8
  %167 = load i64, ptr %K.addr, align 8
  %168 = load i64, ptr %n, align 8
  %mul155 = mul i64 %167, %168
  %add156 = add i64 %166, %mul155
  %mul157 = mul i64 %165, %add156
  %add158 = add i64 %164, %mul157
  %mul159 = mul i64 %163, %add158
  %add160 = add i64 %add154, %mul159
  %arrayidx161 = getelementptr inbounds nuw i8, ptr %160, i64 %add160
  %169 = load i8, ptr %arrayidx161, align 1
  %conv162 = zext i8 %169 to i32
  %170 = load i32, ptr %b_offset.addr, align 4
  %sub163 = sub nsw i32 %conv162, %170
  %conv164 = sitofp i32 %sub163 to float
  %mul165 = fmul float %159, %conv164
  %171 = load float, ptr %acc131, align 4
  %172 = call float @llvm.fmuladd.f32(float %mul153, float %mul165, float %171)
  store float %172, ptr %acc131, align 4
  br label %for.inc167

for.inc167:                                       ; preds = %for.body139
  %173 = load i64, ptr %v136, align 8
  %inc168 = add i64 %173, 1
  store i64 %inc168, ptr %v136, align 8
  br label %for.cond137, !llvm.loop !14

for.end169:                                       ; preds = %for.cond137
  br label %for.inc170

for.inc170:                                       ; preds = %for.end169
  %174 = load i64, ptr %k132, align 8
  %inc171 = add i64 %174, 1
  store i64 %inc171, ptr %k132, align 8
  br label %for.cond133, !llvm.loop !15

for.end172:                                       ; preds = %for.cond133
  %175 = load float, ptr %acc131, align 4
  %176 = load float, ptr %c_scale.addr, align 4
  %div174 = fdiv float %175, %176
  %177 = load i32, ptr %c_offset.addr, align 4
  %conv175 = sitofp i32 %177 to float
  %add176 = fadd float %div174, %conv175
  store float %add176, ptr %x.addr.i, align 4
  store float 0.000000e+00, ptr %lb.addr.i, align 4
  store float 2.550000e+02, ptr %ub.addr.i, align 4
  %178 = load float, ptr %x.addr.i, align 4
  %179 = load float, ptr %lb.addr.i, align 4
  %cmp.i = fcmp olt float %178, %179
  br i1 %cmp.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.end172
  %180 = load float, ptr %lb.addr.i, align 4
  store float %180, ptr %retval.i, align 4
  br label %_ZL5clampfff.exit

if.end.i:                                         ; preds = %for.end172
  %181 = load float, ptr %x.addr.i, align 4
  %182 = load float, ptr %ub.addr.i, align 4
  %cmp1.i = fcmp ogt float %181, %182
  br i1 %cmp1.i, label %if.then2.i, label %if.end3.i

if.then2.i:                                       ; preds = %if.end.i
  %183 = load float, ptr %ub.addr.i, align 4
  store float %183, ptr %retval.i, align 4
  br label %_ZL5clampfff.exit

if.end3.i:                                        ; preds = %if.end.i
  %184 = load float, ptr %x.addr.i, align 4
  store float %184, ptr %retval.i, align 4
  br label %_ZL5clampfff.exit

_ZL5clampfff.exit:                                ; preds = %if.then.i, %if.then2.i, %if.end3.i
  %185 = load float, ptr %retval.i, align 4
  store float %185, ptr %acc_clamped173, align 4
  %186 = load float, ptr %acc_clamped173, align 4
  %conv178 = fptoui float %186 to i8
  %187 = load ptr, ptr %cptr.addr, align 8
  %188 = load i64, ptr %w63, align 8
  %189 = load i64, ptr %v0, align 8
  %add179 = add i64 %188, %189
  %190 = load i64, ptr %W.addr, align 8
  %191 = load i64, ptr %t, align 8
  %192 = load i64, ptr %v1, align 8
  %add180 = add i64 %191, %192
  %193 = load i64, ptr %T.addr, align 8
  %194 = load i64, ptr %c, align 8
  %195 = load i64, ptr %C.addr, align 8
  %196 = load i64, ptr %n, align 8
  %mul181 = mul i64 %195, %196
  %add182 = add i64 %194, %mul181
  %mul183 = mul i64 %193, %add182
  %add184 = add i64 %add180, %mul183
  %mul185 = mul i64 %190, %add184
  %add186 = add i64 %add179, %mul185
  %arrayidx187 = getelementptr inbounds nuw i8, ptr %187, i64 %add186
  store i8 %conv178, ptr %arrayidx187, align 1
  br label %if.end

if.end:                                           ; preds = %_ZL5clampfff.exit, %for.end127
  br label %if.end188

if.end188:                                        ; preds = %if.end, %for.end60
  br label %for.inc189

for.inc189:                                       ; preds = %if.end188
  %197 = load i64, ptr %c, align 8
  %inc190 = add i64 %197, 1
  store i64 %inc190, ptr %c, align 8
  br label %for.cond1, !llvm.loop !16

for.end191:                                       ; preds = %for.cond1
  br label %for.inc192

for.inc192:                                       ; preds = %for.end191
  %198 = load i64, ptr %n, align 8
  %inc193 = add i64 %198, 1
  store i64 %inc193, ptr %n, align 8
  br label %for.cond, !llvm.loop !17

for.end194:                                       ; preds = %for.cond
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.getsize.i64(ptr, i64 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
!16 = distinct !{!16, !7}
!17 = distinct !{!17, !7}
