; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s

; Due to a bug in resolveFrameIndex we ended up with invalid addresses
; containing a base register 0.  Verify that this no longer happens.
; CHECK-NOT: (0)

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.Info = type { i32, i32, i8*, i8*, i8*, [32 x i8*], i64, [32 x i64], i64, i64, i64, [32 x i64] }
%struct.S1998 = type { [2 x i32*], i64, i64, double, i16, i32, [29 x %struct.anon], i16, i8, i32, [8 x i8] }
%struct.anon = type { [16 x double], i32, i16, i32, [3 x i8], [6 x i8], [4 x i32], i8 }

@info = global %struct.Info zeroinitializer, align 8
@fails = global i32 0, align 4
@intarray = global [256 x i32] zeroinitializer, align 4
@s1998 = global %struct.S1998 zeroinitializer, align 16
@a1998 = external global [5 x %struct.S1998]

define void @test1998() {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %tmp = alloca i32, align 4
  %agg.tmp = alloca %struct.S1998, align 16
  %agg.tmp111 = alloca %struct.S1998, align 16
  %agg.tmp112 = alloca %struct.S1998, align 16
  %agg.tmp113 = alloca %struct.S1998, align 16
  %agg.tmp114 = alloca %struct.S1998, align 16
  %agg.tmp115 = alloca %struct.S1998, align 16
  %agg.tmp116 = alloca %struct.S1998, align 16
  %agg.tmp117 = alloca %struct.S1998, align 16
  %agg.tmp118 = alloca %struct.S1998, align 16
  %agg.tmp119 = alloca %struct.S1998, align 16
  call void @llvm.memset.p0i8.i64(i8* bitcast (%struct.S1998* @s1998 to i8*), i8 0, i64 5168, i1 false)
  call void @llvm.memset.p0i8.i64(i8* bitcast ([5 x %struct.S1998]* @a1998 to i8*), i8 0, i64 25840, i1 false)
  call void @llvm.memset.p0i8.i64(i8* bitcast (%struct.Info* @info to i8*), i8 0, i64 832, i1 false)
  store i8* bitcast (%struct.S1998* @s1998 to i8*), i8** getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 2), align 8
  store i8* bitcast ([5 x %struct.S1998]* @a1998 to i8*), i8** getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 3), align 8
  store i8* bitcast (%struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 3) to i8*), i8** getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 4), align 8
  store i64 5168, i64* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 6), align 8
  store i64 16, i64* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 8), align 8
  store i64 16, i64* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 9), align 8
  store i64 16, i64* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 10), align 8
  %0 = load i64, i64* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 8), align 8
  %sub = sub i64 %0, 1
  %and = and i64 ptrtoint (%struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 3) to i64), %sub
  %tobool = icmp ne i64 %and, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load i32, i32* @fails, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* @fails, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store i32 0, i32* %i, align 4
  store i32 0, i32* %j, align 4
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom
  store i8* bitcast (i32** getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 0, i64 1) to i8*), i8** %arrayidx, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %3 to i64
  %arrayidx2 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom1
  store i64 8, i64* %arrayidx2, align 8
  %4 = load i32, i32* %i, align 4
  %idxprom3 = sext i32 %4 to i64
  %arrayidx4 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom3
  store i64 8, i64* %arrayidx4, align 8
  store i32* getelementptr inbounds ([256 x i32], [256 x i32]* @intarray, i32 0, i64 190), i32** getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 0, i64 1), align 8
  store i32* getelementptr inbounds ([256 x i32], [256 x i32]* @intarray, i32 0, i64 241), i32** getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 0, i64 1), align 8
  %5 = load i32, i32* %i, align 4
  %inc5 = add nsw i32 %5, 1
  store i32 %inc5, i32* %i, align 4
  %6 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %6 to i64
  %arrayidx7 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom6
  store i8* bitcast (i64* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 1) to i8*), i8** %arrayidx7, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom8 = sext i32 %7 to i64
  %arrayidx9 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom8
  store i64 8, i64* %arrayidx9, align 8
  %8 = load i32, i32* %i, align 4
  %idxprom10 = sext i32 %8 to i64
  %arrayidx11 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom10
  store i64 8, i64* %arrayidx11, align 8
  store i64 -3866974208859106459, i64* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 1), align 8
  store i64 -185376695371304091, i64* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 1), align 8
  %9 = load i32, i32* %i, align 4
  %inc12 = add nsw i32 %9, 1
  store i32 %inc12, i32* %i, align 4
  %10 = load i32, i32* %i, align 4
  %idxprom13 = sext i32 %10 to i64
  %arrayidx14 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom13
  store i8* bitcast (i64* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 2) to i8*), i8** %arrayidx14, align 8
  %11 = load i32, i32* %i, align 4
  %idxprom15 = sext i32 %11 to i64
  %arrayidx16 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom15
  store i64 8, i64* %arrayidx16, align 8
  %12 = load i32, i32* %i, align 4
  %idxprom17 = sext i32 %12 to i64
  %arrayidx18 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom17
  store i64 8, i64* %arrayidx18, align 8
  store i64 -963638028680427187, i64* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 2), align 8
  store i64 7510542175772455554, i64* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 2), align 8
  %13 = load i32, i32* %i, align 4
  %inc19 = add nsw i32 %13, 1
  store i32 %inc19, i32* %i, align 4
  %14 = load i32, i32* %i, align 4
  %idxprom20 = sext i32 %14 to i64
  %arrayidx21 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom20
  store i8* bitcast (double* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 3) to i8*), i8** %arrayidx21, align 8
  %15 = load i32, i32* %i, align 4
  %idxprom22 = sext i32 %15 to i64
  %arrayidx23 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom22
  store i64 8, i64* %arrayidx23, align 8
  %16 = load i32, i32* %i, align 4
  %idxprom24 = sext i32 %16 to i64
  %arrayidx25 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom24
  store i64 16, i64* %arrayidx25, align 8
  store double 0xC0F8783300000000, double* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 3), align 16
  store double 0xC10DF3CCC0000000, double* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 3), align 16
  %17 = load i32, i32* %i, align 4
  %inc26 = add nsw i32 %17, 1
  store i32 %inc26, i32* %i, align 4
  %18 = load i32, i32* %i, align 4
  %idxprom27 = sext i32 %18 to i64
  %arrayidx28 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom27
  store i8* bitcast (i16* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 4) to i8*), i8** %arrayidx28, align 8
  %19 = load i32, i32* %i, align 4
  %idxprom29 = sext i32 %19 to i64
  %arrayidx30 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom29
  store i64 2, i64* %arrayidx30, align 8
  %20 = load i32, i32* %i, align 4
  %idxprom31 = sext i32 %20 to i64
  %arrayidx32 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom31
  store i64 2, i64* %arrayidx32, align 8
  store i16 -15897, i16* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 4), align 2
  store i16 30935, i16* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 4), align 2
  %21 = load i32, i32* %i, align 4
  %inc33 = add nsw i32 %21, 1
  store i32 %inc33, i32* %i, align 4
  store i32 -419541644, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 5), align 4
  store i32 2125926812, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 5), align 4
  %22 = load i32, i32* %j, align 4
  %inc34 = add nsw i32 %22, 1
  store i32 %inc34, i32* %j, align 4
  %23 = load i32, i32* %i, align 4
  %idxprom35 = sext i32 %23 to i64
  %arrayidx36 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom35
  store i8* bitcast (double* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 0, i64 0) to i8*), i8** %arrayidx36, align 8
  %24 = load i32, i32* %i, align 4
  %idxprom37 = sext i32 %24 to i64
  %arrayidx38 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom37
  store i64 8, i64* %arrayidx38, align 8
  %25 = load i32, i32* %i, align 4
  %idxprom39 = sext i32 %25 to i64
  %arrayidx40 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom39
  store i64 8, i64* %arrayidx40, align 8
  store double 0xC0FC765780000000, double* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 0, i64 0), align 8
  store double 0xC1025CD7A0000000, double* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 0, i64 0), align 8
  %26 = load i32, i32* %i, align 4
  %inc41 = add nsw i32 %26, 1
  store i32 %inc41, i32* %i, align 4
  %bf.load = load i32, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 1), align 8
  %bf.clear = and i32 %bf.load, 7
  %bf.set = or i32 %bf.clear, 16
  store i32 %bf.set, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 1), align 8
  %bf.load42 = load i32, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 1), align 8
  %bf.clear43 = and i32 %bf.load42, 7
  %bf.set44 = or i32 %bf.clear43, 24
  store i32 %bf.set44, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 1), align 8
  %27 = load i32, i32* %j, align 4
  %inc45 = add nsw i32 %27, 1
  store i32 %inc45, i32* %j, align 4
  %bf.load46 = load i16, i16* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 2), align 4
  %bf.clear47 = and i16 %bf.load46, 127
  store i16 %bf.clear47, i16* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 2), align 4
  %bf.load48 = load i16, i16* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 2), align 4
  %bf.clear49 = and i16 %bf.load48, 127
  store i16 %bf.clear49, i16* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 2), align 4
  %28 = load i32, i32* %j, align 4
  %inc50 = add nsw i32 %28, 1
  store i32 %inc50, i32* %j, align 4
  %bf.load51 = load i32, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 3), align 8
  %bf.clear52 = and i32 %bf.load51, 63
  store i32 %bf.clear52, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 3), align 8
  %bf.load53 = load i32, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 3), align 8
  %bf.clear54 = and i32 %bf.load53, 63
  %bf.set55 = or i32 %bf.clear54, 64
  store i32 %bf.set55, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 3), align 8
  %29 = load i32, i32* %j, align 4
  %inc56 = add nsw i32 %29, 1
  store i32 %inc56, i32* %j, align 4
  %bf.load57 = load i24, i24* bitcast ([3 x i8]* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 4) to i24*), align 4
  %bf.clear58 = and i24 %bf.load57, 63
  store i24 %bf.clear58, i24* bitcast ([3 x i8]* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 4) to i24*), align 4
  %bf.load59 = load i24, i24* bitcast ([3 x i8]* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 4) to i24*), align 4
  %bf.clear60 = and i24 %bf.load59, 63
  store i24 %bf.clear60, i24* bitcast ([3 x i8]* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 4) to i24*), align 4
  %30 = load i32, i32* %j, align 4
  %inc61 = add nsw i32 %30, 1
  store i32 %inc61, i32* %j, align 4
  %31 = load i32, i32* %i, align 4
  %idxprom62 = sext i32 %31 to i64
  %arrayidx63 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom62
  store i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 5, i64 5), i8** %arrayidx63, align 8
  %32 = load i32, i32* %i, align 4
  %idxprom64 = sext i32 %32 to i64
  %arrayidx65 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom64
  store i64 1, i64* %arrayidx65, align 8
  %33 = load i32, i32* %i, align 4
  %idxprom66 = sext i32 %33 to i64
  %arrayidx67 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom66
  store i64 1, i64* %arrayidx67, align 8
  store i8 -83, i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 5, i64 5), align 1
  store i8 -67, i8* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 5, i64 5), align 1
  %34 = load i32, i32* %i, align 4
  %inc68 = add nsw i32 %34, 1
  store i32 %inc68, i32* %i, align 4
  %35 = load i32, i32* %i, align 4
  %idxprom69 = sext i32 %35 to i64
  %arrayidx70 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom69
  store i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 5, i64 1), i8** %arrayidx70, align 8
  %36 = load i32, i32* %i, align 4
  %idxprom71 = sext i32 %36 to i64
  %arrayidx72 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom71
  store i64 1, i64* %arrayidx72, align 8
  %37 = load i32, i32* %i, align 4
  %idxprom73 = sext i32 %37 to i64
  %arrayidx74 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom73
  store i64 1, i64* %arrayidx74, align 8
  store i8 34, i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 5, i64 1), align 1
  store i8 64, i8* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 5, i64 1), align 1
  %38 = load i32, i32* %i, align 4
  %inc75 = add nsw i32 %38, 1
  store i32 %inc75, i32* %i, align 4
  %39 = load i32, i32* %i, align 4
  %idxprom76 = sext i32 %39 to i64
  %arrayidx77 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom76
  store i8* bitcast (i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 6, i64 3) to i8*), i8** %arrayidx77, align 8
  %40 = load i32, i32* %i, align 4
  %idxprom78 = sext i32 %40 to i64
  %arrayidx79 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom78
  store i64 4, i64* %arrayidx79, align 8
  %41 = load i32, i32* %i, align 4
  %idxprom80 = sext i32 %41 to i64
  %arrayidx81 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom80
  store i64 4, i64* %arrayidx81, align 8
  store i32 -3, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 6, i64 3), align 4
  store i32 -3, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 6, i64 3), align 4
  %42 = load i32, i32* %i, align 4
  %inc82 = add nsw i32 %42, 1
  store i32 %inc82, i32* %i, align 4
  %43 = load i32, i32* %i, align 4
  %idxprom83 = sext i32 %43 to i64
  %arrayidx84 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom83
  store i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 7), i8** %arrayidx84, align 8
  %44 = load i32, i32* %i, align 4
  %idxprom85 = sext i32 %44 to i64
  %arrayidx86 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom85
  store i64 1, i64* %arrayidx86, align 8
  %45 = load i32, i32* %i, align 4
  %idxprom87 = sext i32 %45 to i64
  %arrayidx88 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom87
  store i64 1, i64* %arrayidx88, align 8
  store i8 106, i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 6, i64 4, i32 7), align 1
  store i8 -102, i8* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 6, i64 4, i32 7), align 1
  %46 = load i32, i32* %i, align 4
  %inc89 = add nsw i32 %46, 1
  store i32 %inc89, i32* %i, align 4
  %47 = load i32, i32* %i, align 4
  %idxprom90 = sext i32 %47 to i64
  %arrayidx91 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom90
  store i8* bitcast (i16* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 7) to i8*), i8** %arrayidx91, align 8
  %48 = load i32, i32* %i, align 4
  %idxprom92 = sext i32 %48 to i64
  %arrayidx93 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom92
  store i64 2, i64* %arrayidx93, align 8
  %49 = load i32, i32* %i, align 4
  %idxprom94 = sext i32 %49 to i64
  %arrayidx95 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom94
  store i64 2, i64* %arrayidx95, align 8
  store i16 29665, i16* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 7), align 2
  store i16 7107, i16* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 7), align 2
  %50 = load i32, i32* %i, align 4
  %inc96 = add nsw i32 %50, 1
  store i32 %inc96, i32* %i, align 4
  %51 = load i32, i32* %i, align 4
  %idxprom97 = sext i32 %51 to i64
  %arrayidx98 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom97
  store i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 8), i8** %arrayidx98, align 8
  %52 = load i32, i32* %i, align 4
  %idxprom99 = sext i32 %52 to i64
  %arrayidx100 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom99
  store i64 1, i64* %arrayidx100, align 8
  %53 = load i32, i32* %i, align 4
  %idxprom101 = sext i32 %53 to i64
  %arrayidx102 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom101
  store i64 1, i64* %arrayidx102, align 8
  store i8 52, i8* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 8), align 1
  store i8 -86, i8* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 8), align 1
  %54 = load i32, i32* %i, align 4
  %inc103 = add nsw i32 %54, 1
  store i32 %inc103, i32* %i, align 4
  %55 = load i32, i32* %i, align 4
  %idxprom104 = sext i32 %55 to i64
  %arrayidx105 = getelementptr inbounds [32 x i8*], [32 x i8*]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 5), i32 0, i64 %idxprom104
  store i8* bitcast (i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 9) to i8*), i8** %arrayidx105, align 8
  %56 = load i32, i32* %i, align 4
  %idxprom106 = sext i32 %56 to i64
  %arrayidx107 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 7), i32 0, i64 %idxprom106
  store i64 4, i64* %arrayidx107, align 8
  %57 = load i32, i32* %i, align 4
  %idxprom108 = sext i32 %57 to i64
  %arrayidx109 = getelementptr inbounds [32 x i64], [32 x i64]* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 11), i32 0, i64 %idxprom108
  store i64 4, i64* %arrayidx109, align 8
  store i32 -54118453, i32* getelementptr inbounds (%struct.S1998, %struct.S1998* @s1998, i32 0, i32 9), align 4
  store i32 1668755823, i32* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2, i32 9), align 4
  %58 = load i32, i32* %i, align 4
  %inc110 = add nsw i32 %58, 1
  store i32 %inc110, i32* %i, align 4
  store i32 %inc110, i32* %tmp
  %59 = load i32, i32* %tmp
  %60 = load i32, i32* %i, align 4
  store i32 %60, i32* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 0), align 4
  %61 = load i32, i32* %j, align 4
  store i32 %61, i32* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 1), align 4
  %62 = bitcast %struct.S1998* %agg.tmp111 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %62, i8* bitcast (%struct.S1998* @s1998 to i8*), i64 5168, i1 false)
  %63 = bitcast %struct.S1998* %agg.tmp112 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %63, i8* bitcast (%struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2) to i8*), i64 5168, i1 false)
  call void @check1998(%struct.S1998* sret %agg.tmp, %struct.S1998* byval align 16 %agg.tmp111, %struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 1), %struct.S1998* byval align 16 %agg.tmp112)
  call void @checkx1998(%struct.S1998* byval align 16 %agg.tmp)
  %64 = bitcast %struct.S1998* %agg.tmp113 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %64, i8* bitcast (%struct.S1998* @s1998 to i8*), i64 5168, i1 false)
  %65 = bitcast %struct.S1998* %agg.tmp114 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %65, i8* bitcast (%struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2) to i8*), i64 5168, i1 false)
  %66 = bitcast %struct.S1998* %agg.tmp115 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %66, i8* bitcast (%struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2) to i8*), i64 5168, i1 false)
  call void (i32, ...) @check1998va(i32 signext 1, double 1.000000e+00, %struct.S1998* byval align 16 %agg.tmp113, i64 2, %struct.S1998* byval align 16 %agg.tmp114, %struct.S1998* byval align 16 %agg.tmp115)
  %67 = bitcast %struct.S1998* %agg.tmp116 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %67, i8* bitcast (%struct.S1998* @s1998 to i8*), i64 5168, i1 false)
  %68 = bitcast %struct.S1998* %agg.tmp117 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %68, i8* bitcast (%struct.S1998* @s1998 to i8*), i64 5168, i1 false)
  %69 = bitcast %struct.S1998* %agg.tmp118 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %69, i8* bitcast (%struct.S1998* getelementptr inbounds ([5 x %struct.S1998], [5 x %struct.S1998]* @a1998, i32 0, i64 2) to i8*), i64 5168, i1 false)
  %70 = bitcast %struct.S1998* %agg.tmp119 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %70, i8* bitcast (%struct.S1998* @s1998 to i8*), i64 5168, i1 false)
  call void (i32, ...) @check1998va(i32 signext 2, %struct.S1998* byval align 16 %agg.tmp116, %struct.S1998* byval align 16 %agg.tmp117, ppc_fp128 0xM40000000000000000000000000000000, %struct.S1998* byval align 16 %agg.tmp118, %struct.S1998* byval align 16 %agg.tmp119)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)

declare void @check1998(%struct.S1998* sret, %struct.S1998* byval align 16, %struct.S1998*, %struct.S1998* byval align 16)
declare void @check1998va(i32 signext, ...)
declare void @checkx1998(%struct.S1998* byval align 16 %arg)

