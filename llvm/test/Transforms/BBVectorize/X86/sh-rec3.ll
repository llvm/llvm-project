target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -basicaa -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -S | FileCheck %s

%struct.gsm_state.2.8.39.44.45.55.56.57.58.59.62.63.64.65.74.75.76.77.80.87.92.93.94.95.96.97.110.111.112.113.114.128.130.135.136.137.138.139.140.141.142.143.144.145.148.149.150.151.152.169.170.177.178.179.184.185.186.187.188.201.208.209.219.220.221.223.224.225.230.231.232.233.235.236.237.238.245.246.248.249.272.274.279.280.281.282.283.286.293.298.299.314.315.316.317.318.319.320.321.322.323.324.325.326.327.328.329.330.331.332.333.334.335.336.337.338.339.340.341.342.343.344.345.346.347.348.349.350.351.352.353.565 = type { [280 x i16], i16, i64, i32, [8 x i16], [2 x [8 x i16]], i16, i16, [9 x i16], i16, i8, i8 }

define void @gsm_encode(%struct.gsm_state.2.8.39.44.45.55.56.57.58.59.62.63.64.65.74.75.76.77.80.87.92.93.94.95.96.97.110.111.112.113.114.128.130.135.136.137.138.139.140.141.142.143.144.145.148.149.150.151.152.169.170.177.178.179.184.185.186.187.188.201.208.209.219.220.221.223.224.225.230.231.232.233.235.236.237.238.245.246.248.249.272.274.279.280.281.282.283.286.293.298.299.314.315.316.317.318.319.320.321.322.323.324.325.326.327.328.329.330.331.332.333.334.335.336.337.338.339.340.341.342.343.344.345.346.347.348.349.350.351.352.353.565* %s, i16* %source, i8* %c) nounwind uwtable {
entry:
  %LARc28 = alloca [2 x i64], align 16
  %LARc28.sub = getelementptr inbounds [2 x i64]* %LARc28, i64 0, i64 0
  %tmpcast = bitcast [2 x i64]* %LARc28 to [8 x i16]*
  %Nc = alloca [4 x i16], align 2
  %Mc = alloca [4 x i16], align 2
  %bc = alloca [4 x i16], align 2
  %xmc = alloca [52 x i16], align 16
  %arraydecay = bitcast [2 x i64]* %LARc28 to i16*
  %arraydecay1 = getelementptr inbounds [4 x i16]* %Nc, i64 0, i64 0
  %arraydecay2 = getelementptr inbounds [4 x i16]* %bc, i64 0, i64 0
  %arraydecay3 = getelementptr inbounds [4 x i16]* %Mc, i64 0, i64 0
  %arraydecay5 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 0
  call void @Gsm_Coder(%struct.gsm_state.2.8.39.44.45.55.56.57.58.59.62.63.64.65.74.75.76.77.80.87.92.93.94.95.96.97.110.111.112.113.114.128.130.135.136.137.138.139.140.141.142.143.144.145.148.149.150.151.152.169.170.177.178.179.184.185.186.187.188.201.208.209.219.220.221.223.224.225.230.231.232.233.235.236.237.238.245.246.248.249.272.274.279.280.281.282.283.286.293.298.299.314.315.316.317.318.319.320.321.322.323.324.325.326.327.328.329.330.331.332.333.334.335.336.337.338.339.340.341.342.343.344.345.346.347.348.349.350.351.352.353.565* %s, i16* %source, i16* %arraydecay, i16* %arraydecay1, i16* %arraydecay2, i16* %arraydecay3, i16* undef, i16* %arraydecay5) nounwind
  %0 = load i64* %LARc28.sub, align 16
  %1 = trunc i64 %0 to i32
  %conv1 = lshr i32 %1, 2
  %and = and i32 %conv1, 15
  %or = or i32 %and, 208
  %conv6 = trunc i32 %or to i8
  %incdec.ptr = getelementptr inbounds i8* %c, i64 1
  store i8 %conv6, i8* %c, align 1
  %conv84 = trunc i64 %0 to i8
  %and9 = shl i8 %conv84, 6
  %incdec.ptr15 = getelementptr inbounds i8* %c, i64 2
  store i8 %and9, i8* %incdec.ptr, align 1
  %2 = lshr i64 %0, 50
  %shr226.tr = trunc i64 %2 to i8
  %conv25 = and i8 %shr226.tr, 7
  %incdec.ptr26 = getelementptr inbounds i8* %c, i64 3
  store i8 %conv25, i8* %incdec.ptr15, align 1
  %incdec.ptr42 = getelementptr inbounds i8* %c, i64 4
  store i8 0, i8* %incdec.ptr26, align 1
  %arrayidx52 = getelementptr inbounds [8 x i16]* %tmpcast, i64 0, i64 7
  %3 = load i16* %arrayidx52, align 2
  %conv537 = trunc i16 %3 to i8
  %and54 = and i8 %conv537, 7
  %incdec.ptr57 = getelementptr inbounds i8* %c, i64 5
  store i8 %and54, i8* %incdec.ptr42, align 1
  %incdec.ptr68 = getelementptr inbounds i8* %c, i64 6
  store i8 0, i8* %incdec.ptr57, align 1
  %4 = load i16* %arraydecay3, align 2
  %conv748 = trunc i16 %4 to i8
  %and75 = shl i8 %conv748, 5
  %shl76 = and i8 %and75, 96
  %incdec.ptr84 = getelementptr inbounds i8* %c, i64 7
  store i8 %shl76, i8* %incdec.ptr68, align 1
  %arrayidx94 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 1
  %5 = load i16* %arrayidx94, align 2
  %conv959 = trunc i16 %5 to i8
  %and96 = shl i8 %conv959, 1
  %shl97 = and i8 %and96, 14
  %or103 = or i8 %shl97, 1
  %incdec.ptr105 = getelementptr inbounds i8* %c, i64 8
  store i8 %or103, i8* %incdec.ptr84, align 1
  %arrayidx115 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 4
  %6 = bitcast i16* %arrayidx115 to i32*
  %7 = load i32* %6, align 8
  %conv11610 = trunc i32 %7 to i8
  %and117 = and i8 %conv11610, 7
  %incdec.ptr120 = getelementptr inbounds i8* %c, i64 9
  store i8 %and117, i8* %incdec.ptr105, align 1
  %8 = lshr i32 %7, 16
  %and12330 = shl nuw nsw i32 %8, 5
  %and123 = trunc i32 %and12330 to i8
  %incdec.ptr136 = getelementptr inbounds i8* %c, i64 10
  store i8 %and123, i8* %incdec.ptr120, align 1
  %incdec.ptr157 = getelementptr inbounds i8* %c, i64 11
  store i8 0, i8* %incdec.ptr136, align 1
  %incdec.ptr172 = getelementptr inbounds i8* %c, i64 12
  store i8 0, i8* %incdec.ptr157, align 1
  %arrayidx173 = getelementptr inbounds [4 x i16]* %Nc, i64 0, i64 1
  %9 = load i16* %arrayidx173, align 2
  %conv17412 = zext i16 %9 to i32
  %and175 = shl nuw nsw i32 %conv17412, 1
  %arrayidx177 = getelementptr inbounds [4 x i16]* %bc, i64 0, i64 1
  %10 = load i16* %arrayidx177, align 2
  %conv17826 = zext i16 %10 to i32
  %shr17913 = lshr i32 %conv17826, 1
  %and180 = and i32 %shr17913, 1
  %or181 = or i32 %and175, %and180
  %conv182 = trunc i32 %or181 to i8
  %incdec.ptr183 = getelementptr inbounds i8* %c, i64 13
  store i8 %conv182, i8* %incdec.ptr172, align 1
  %arrayidx188 = getelementptr inbounds [4 x i16]* %Mc, i64 0, i64 1
  %11 = load i16* %arrayidx188, align 2
  %conv18914 = trunc i16 %11 to i8
  %and190 = shl i8 %conv18914, 5
  %shl191 = and i8 %and190, 96
  %incdec.ptr199 = getelementptr inbounds i8* %c, i64 14
  store i8 %shl191, i8* %incdec.ptr183, align 1
  %arrayidx209 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 14
  %12 = load i16* %arrayidx209, align 4
  %conv21015 = trunc i16 %12 to i8
  %and211 = shl i8 %conv21015, 1
  %shl212 = and i8 %and211, 14
  %or218 = or i8 %shl212, 1
  %incdec.ptr220 = getelementptr inbounds i8* %c, i64 15
  store i8 %or218, i8* %incdec.ptr199, align 1
  %arrayidx225 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 16
  %13 = bitcast i16* %arrayidx225 to i64*
  %14 = load i64* %13, align 16
  %conv22616 = trunc i64 %14 to i8
  %and227 = shl i8 %conv22616, 3
  %shl228 = and i8 %and227, 56
  %incdec.ptr235 = getelementptr inbounds i8* %c, i64 16
  store i8 %shl228, i8* %incdec.ptr220, align 1
  %15 = lshr i64 %14, 32
  %and23832 = shl nuw nsw i64 %15, 5
  %and238 = trunc i64 %and23832 to i8
  %incdec.ptr251 = getelementptr inbounds i8* %c, i64 17
  store i8 %and238, i8* %incdec.ptr235, align 1
  %arrayidx266 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 23
  %incdec.ptr272 = getelementptr inbounds i8* %c, i64 18
  store i8 0, i8* %incdec.ptr251, align 1
  %16 = load i16* %arrayidx266, align 2
  %conv27418 = trunc i16 %16 to i8
  %and275 = shl i8 %conv27418, 6
  %incdec.ptr287 = getelementptr inbounds i8* %c, i64 19
  store i8 %and275, i8* %incdec.ptr272, align 1
  %arrayidx288 = getelementptr inbounds [4 x i16]* %Nc, i64 0, i64 2
  %17 = load i16* %arrayidx288, align 2
  %conv28919 = zext i16 %17 to i32
  %and290 = shl nuw nsw i32 %conv28919, 1
  %arrayidx292 = getelementptr inbounds [4 x i16]* %bc, i64 0, i64 2
  %18 = load i16* %arrayidx292, align 2
  %conv29327 = zext i16 %18 to i32
  %shr29420 = lshr i32 %conv29327, 1
  %and295 = and i32 %shr29420, 1
  %or296 = or i32 %and290, %and295
  %conv297 = trunc i32 %or296 to i8
  %incdec.ptr298 = getelementptr inbounds i8* %c, i64 20
  store i8 %conv297, i8* %incdec.ptr287, align 1
  %conv30021 = trunc i16 %18 to i8
  %and301 = shl i8 %conv30021, 7
  %incdec.ptr314 = getelementptr inbounds i8* %c, i64 21
  store i8 %and301, i8* %incdec.ptr298, align 1
  %incdec.ptr335 = getelementptr inbounds i8* %c, i64 22
  store i8 0, i8* %incdec.ptr314, align 1
  %arrayidx340 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 29
  %19 = load i16* %arrayidx340, align 2
  %conv34122 = trunc i16 %19 to i8
  %and342 = shl i8 %conv34122, 3
  %shl343 = and i8 %and342, 56
  %incdec.ptr350 = getelementptr inbounds i8* %c, i64 23
  store i8 %shl343, i8* %incdec.ptr335, align 1
  %arrayidx355 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 32
  %20 = bitcast i16* %arrayidx355 to i32*
  %21 = load i32* %20, align 16
  %conv35623 = shl i32 %21, 2
  %shl358 = and i32 %conv35623, 28
  %22 = lshr i32 %21, 17
  %and363 = and i32 %22, 3
  %or364 = or i32 %shl358, %and363
  %conv365 = trunc i32 %or364 to i8
  store i8 %conv365, i8* %incdec.ptr350, align 1
  unreachable
; CHECK: @gsm_encode
}

declare void @Gsm_Coder(%struct.gsm_state.2.8.39.44.45.55.56.57.58.59.62.63.64.65.74.75.76.77.80.87.92.93.94.95.96.97.110.111.112.113.114.128.130.135.136.137.138.139.140.141.142.143.144.145.148.149.150.151.152.169.170.177.178.179.184.185.186.187.188.201.208.209.219.220.221.223.224.225.230.231.232.233.235.236.237.238.245.246.248.249.272.274.279.280.281.282.283.286.293.298.299.314.315.316.317.318.319.320.321.322.323.324.325.326.327.328.329.330.331.332.333.334.335.336.337.338.339.340.341.342.343.344.345.346.347.348.349.350.351.352.353.565*, i16*, i16*, i16*, i16*, i16*, i16*, i16*)

declare void @llvm.trap() noreturn nounwind
