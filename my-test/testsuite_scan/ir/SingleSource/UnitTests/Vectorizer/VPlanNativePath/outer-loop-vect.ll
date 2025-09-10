; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/VPlanNativePath/outer-loop-vect.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/VPlanNativePath/outer-loop-vect.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::mersenne_twister_engine" = type { [624 x i64], i64 }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::uniform_int_distribution" = type { %"struct.std::uniform_int_distribution<>::param_type" }
%"struct.std::uniform_int_distribution<>::param_type" = type { i32, i32 }

$_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

@_ZL3rng = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [32 x i8] c"Checking matrix-multiplication\0A\00", align 1
@.str.1 = private unnamed_addr constant [33 x i8] c"Checking loop with auxiliary IV\0A\00", align 1
@.str.2 = private unnamed_addr constant [45 x i8] c"Checking loop with indirect memory accesses\0A\00", align 1
@.str.3 = private unnamed_addr constant [27 x i8] c"Checking triple-loop-nest\0A\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.4 = private unnamed_addr constant [12 x i8] c"Miscompare\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_outer_loop_vect.cpp, ptr null }]

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::uniform_int_distribution", align 8
  %2 = alloca %"class.std::uniform_int_distribution", align 8
  %3 = alloca %"class.std::uniform_int_distribution", align 8
  %4 = alloca %"class.std::uniform_int_distribution", align 8
  %5 = alloca %"class.std::uniform_int_distribution", align 8
  %6 = alloca %"class.std::mersenne_twister_engine", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #13
  store i64 15, ptr %6, align 8, !tbaa !6
  br label %7

7:                                                ; preds = %7, %0
  %8 = phi i64 [ 15, %0 ], [ %15, %7 ]
  %9 = phi i64 [ 1, %0 ], [ %16, %7 ]
  %10 = getelementptr i64, ptr %6, i64 %9
  %11 = lshr i64 %8, 30
  %12 = xor i64 %11, %8
  %13 = mul nuw nsw i64 %12, 1812433253
  %14 = add nuw i64 %13, %9
  %15 = and i64 %14, 4294967295
  store i64 %15, ptr %10, align 8, !tbaa !6
  %16 = add nuw nsw i64 %9, 1
  %17 = icmp eq i64 %16, 624
  br i1 %17, label %18, label %7, !llvm.loop !10

18:                                               ; preds = %7
  %19 = getelementptr inbounds nuw i8, ptr %6, i64 4992
  store i64 624, ptr %19, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %6, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #13
  %20 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 31)
  %21 = tail call noalias noundef nonnull dereferenceable(40000) ptr @_Znam(i64 noundef 40000) #14
  %22 = invoke noalias noundef nonnull dereferenceable(40000) ptr @_Znam(i64 noundef 40000) #14
          to label %23 unwind label %616

23:                                               ; preds = %18
  %24 = invoke noalias noundef nonnull dereferenceable(40000) ptr @_Znam(i64 noundef 40000) #14
          to label %25 unwind label %618

25:                                               ; preds = %23
  %26 = invoke noalias noundef nonnull dereferenceable(40000) ptr @_Znam(i64 noundef 40000) #14
          to label %27 unwind label %620

27:                                               ; preds = %25
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #13
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %5, align 8, !tbaa !16
  br label %28

28:                                               ; preds = %31, %27
  %29 = phi i64 [ 0, %27 ], [ %33, %31 ]
  %30 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %5)
          to label %31 unwind label %624

31:                                               ; preds = %28
  %32 = getelementptr inbounds nuw i32, ptr %24, i64 %29
  store i32 %30, ptr %32, align 4, !tbaa !16
  %33 = add nuw nsw i64 %29, 1
  %34 = icmp eq i64 %33, 10000
  br i1 %34, label %35, label %28, !llvm.loop !18

35:                                               ; preds = %31
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #13
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %4, align 8, !tbaa !16
  br label %36

36:                                               ; preds = %39, %35
  %37 = phi i64 [ 0, %35 ], [ %41, %39 ]
  %38 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %4)
          to label %39 unwind label %622

39:                                               ; preds = %36
  %40 = getelementptr inbounds nuw i32, ptr %26, i64 %37
  store i32 %38, ptr %40, align 4, !tbaa !16
  %41 = add nuw nsw i64 %37, 1
  %42 = icmp eq i64 %41, 10000
  br i1 %42, label %43, label %36, !llvm.loop !18

43:                                               ; preds = %39
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #13
  call void @llvm.experimental.noalias.scope.decl(metadata !19)
  br label %44

44:                                               ; preds = %78, %43
  %45 = phi i64 [ 0, %43 ], [ %79, %78 ]
  %46 = mul nuw nsw i64 %45, 400
  %47 = getelementptr inbounds nuw i8, ptr %24, i64 %46
  %48 = getelementptr inbounds nuw i8, ptr %21, i64 %46
  br label %49

49:                                               ; preds = %73, %44
  %50 = phi i64 [ 0, %44 ], [ %76, %73 ]
  %51 = getelementptr inbounds nuw i32, ptr %26, i64 %50
  br label %52

52:                                               ; preds = %52, %49
  %53 = phi i64 [ 0, %49 ], [ %71, %52 ]
  %54 = phi i32 [ 0, %49 ], [ %69, %52 ]
  %55 = phi i32 [ 0, %49 ], [ %70, %52 ]
  %56 = or disjoint i64 %53, 1
  %57 = getelementptr inbounds nuw i32, ptr %47, i64 %53
  %58 = getelementptr inbounds nuw i32, ptr %47, i64 %56
  %59 = load i32, ptr %57, align 4, !tbaa !16, !noalias !19
  %60 = load i32, ptr %58, align 4, !tbaa !16, !noalias !19
  %61 = mul nuw nsw i64 %53, 400
  %62 = mul nuw nsw i64 %56, 400
  %63 = getelementptr inbounds nuw i8, ptr %51, i64 %61
  %64 = getelementptr inbounds nuw i8, ptr %51, i64 %62
  %65 = load i32, ptr %63, align 4, !tbaa !16, !noalias !19
  %66 = load i32, ptr %64, align 4, !tbaa !16, !noalias !19
  %67 = mul nsw i32 %65, %59
  %68 = mul nsw i32 %66, %60
  %69 = add i32 %67, %54
  %70 = add i32 %68, %55
  %71 = add nuw i64 %53, 2
  %72 = icmp eq i64 %71, 100
  br i1 %72, label %73, label %52, !llvm.loop !22

73:                                               ; preds = %52
  %74 = add i32 %70, %69
  %75 = getelementptr inbounds nuw i32, ptr %48, i64 %50
  store i32 %74, ptr %75, align 4, !tbaa !16, !alias.scope !19
  %76 = add nuw nsw i64 %50, 1
  %77 = icmp eq i64 %76, 100
  br i1 %77, label %78, label %49, !llvm.loop !25

78:                                               ; preds = %73
  %79 = add nuw nsw i64 %45, 1
  %80 = icmp eq i64 %79, 100
  br i1 %80, label %81, label %44, !llvm.loop !28

81:                                               ; preds = %78
  call void @llvm.experimental.noalias.scope.decl(metadata !29)
  br label %82

82:                                               ; preds = %116, %81
  %83 = phi i64 [ 0, %81 ], [ %117, %116 ]
  %84 = mul nuw nsw i64 %83, 400
  %85 = getelementptr inbounds nuw i8, ptr %24, i64 %84
  %86 = getelementptr inbounds nuw i8, ptr %22, i64 %84
  br label %87

87:                                               ; preds = %111, %82
  %88 = phi i64 [ 0, %82 ], [ %114, %111 ]
  %89 = getelementptr inbounds nuw i32, ptr %26, i64 %88
  br label %90

90:                                               ; preds = %90, %87
  %91 = phi i64 [ 0, %87 ], [ %109, %90 ]
  %92 = phi i32 [ 0, %87 ], [ %107, %90 ]
  %93 = phi i32 [ 0, %87 ], [ %108, %90 ]
  %94 = or disjoint i64 %91, 1
  %95 = getelementptr inbounds nuw i32, ptr %85, i64 %91
  %96 = getelementptr inbounds nuw i32, ptr %85, i64 %94
  %97 = load i32, ptr %95, align 4, !tbaa !16, !noalias !29
  %98 = load i32, ptr %96, align 4, !tbaa !16, !noalias !29
  %99 = mul nuw nsw i64 %91, 400
  %100 = mul nuw nsw i64 %94, 400
  %101 = getelementptr inbounds nuw i8, ptr %89, i64 %99
  %102 = getelementptr inbounds nuw i8, ptr %89, i64 %100
  %103 = load i32, ptr %101, align 4, !tbaa !16, !noalias !29
  %104 = load i32, ptr %102, align 4, !tbaa !16, !noalias !29
  %105 = mul nsw i32 %103, %97
  %106 = mul nsw i32 %104, %98
  %107 = add i32 %105, %92
  %108 = add i32 %106, %93
  %109 = add nuw i64 %91, 2
  %110 = icmp eq i64 %109, 100
  br i1 %110, label %111, label %90, !llvm.loop !32

111:                                              ; preds = %90
  %112 = add i32 %108, %107
  %113 = getelementptr inbounds nuw i32, ptr %86, i64 %88
  store i32 %112, ptr %113, align 4, !tbaa !16, !alias.scope !29
  %114 = add nuw nsw i64 %88, 1
  %115 = icmp eq i64 %114, 100
  br i1 %115, label %116, label %87, !llvm.loop !33

116:                                              ; preds = %111
  %117 = add nuw nsw i64 %83, 1
  %118 = icmp eq i64 %117, 100
  br i1 %118, label %119, label %82, !llvm.loop !35

119:                                              ; preds = %116
  %120 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(40000) %21, ptr noundef nonnull readonly dereferenceable(40000) %22, i64 40000)
  %121 = icmp eq i32 %120, 0
  br i1 %121, label %125, label %122

122:                                              ; preds = %119
  %123 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.4)
          to label %124 unwind label %626

124:                                              ; preds = %122
  call void @exit(i32 noundef 1) #15
  unreachable

125:                                              ; preds = %119
  call void @_ZdaPv(ptr noundef nonnull %26) #16
  call void @_ZdaPv(ptr noundef nonnull %24) #16
  call void @_ZdaPv(ptr noundef nonnull %22) #16
  call void @_ZdaPv(ptr noundef nonnull %21) #16
  %126 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.1, i64 noundef 32)
  %127 = call noalias noundef nonnull dereferenceable(492) ptr @_Znam(i64 noundef 492) #14
  %128 = invoke noalias noundef nonnull dereferenceable(492) ptr @_Znam(i64 noundef 492) #14
          to label %129 unwind label %634

129:                                              ; preds = %125
  %130 = invoke noalias noundef nonnull dereferenceable(492) ptr @_Znam(i64 noundef 492) #14
          to label %131 unwind label %636

131:                                              ; preds = %129
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #13
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %3, align 8, !tbaa !16
  br label %132

132:                                              ; preds = %135, %131
  %133 = phi i64 [ 0, %131 ], [ %137, %135 ]
  %134 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %3, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %3)
          to label %135 unwind label %638

135:                                              ; preds = %132
  %136 = getelementptr inbounds nuw i32, ptr %130, i64 %133
  store i32 %134, ptr %136, align 4, !tbaa !16
  %137 = add nuw nsw i64 %133, 1
  %138 = icmp eq i64 %137, 123
  br i1 %138, label %139, label %132, !llvm.loop !18

139:                                              ; preds = %135
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #13
  call void @llvm.experimental.noalias.scope.decl(metadata !36)
  %140 = getelementptr inbounds nuw i8, ptr %130, i64 480
  %141 = getelementptr inbounds nuw i8, ptr %130, i64 464
  %142 = load <4 x i32>, ptr %141, align 4, !tbaa !16, !noalias !36
  %143 = getelementptr inbounds nuw i8, ptr %130, i64 432
  %144 = load <4 x i32>, ptr %143, align 4, !tbaa !16, !noalias !36
  %145 = getelementptr inbounds nuw i8, ptr %130, i64 400
  %146 = load <4 x i32>, ptr %145, align 4, !tbaa !16, !noalias !36
  %147 = getelementptr inbounds nuw i8, ptr %130, i64 368
  %148 = load <4 x i32>, ptr %147, align 4, !tbaa !16, !noalias !36
  %149 = getelementptr inbounds nuw i8, ptr %130, i64 336
  %150 = load <4 x i32>, ptr %149, align 4, !tbaa !16, !noalias !36
  %151 = getelementptr inbounds nuw i8, ptr %130, i64 304
  %152 = load <4 x i32>, ptr %151, align 4, !tbaa !16, !noalias !36
  %153 = getelementptr inbounds nuw i8, ptr %130, i64 272
  %154 = load <4 x i32>, ptr %153, align 4, !tbaa !16, !noalias !36
  %155 = getelementptr inbounds nuw i8, ptr %130, i64 240
  %156 = load <4 x i32>, ptr %155, align 4, !tbaa !16, !noalias !36
  %157 = getelementptr inbounds nuw i8, ptr %130, i64 208
  %158 = load <4 x i32>, ptr %157, align 4, !tbaa !16, !noalias !36
  %159 = getelementptr inbounds nuw i8, ptr %130, i64 176
  %160 = load <4 x i32>, ptr %159, align 4, !tbaa !16, !noalias !36
  %161 = getelementptr inbounds nuw i8, ptr %130, i64 144
  %162 = load <4 x i32>, ptr %161, align 4, !tbaa !16, !noalias !36
  %163 = getelementptr inbounds nuw i8, ptr %130, i64 112
  %164 = load <4 x i32>, ptr %163, align 4, !tbaa !16, !noalias !36
  %165 = getelementptr inbounds nuw i8, ptr %130, i64 80
  %166 = load <4 x i32>, ptr %165, align 4, !tbaa !16, !noalias !36
  %167 = getelementptr inbounds nuw i8, ptr %130, i64 48
  %168 = load <4 x i32>, ptr %167, align 4, !tbaa !16, !noalias !36
  %169 = getelementptr inbounds nuw i8, ptr %130, i64 16
  %170 = load <4 x i32>, ptr %169, align 4, !tbaa !16, !noalias !36
  %171 = add <4 x i32> %168, %170
  %172 = add <4 x i32> %166, %171
  %173 = add <4 x i32> %164, %172
  %174 = add <4 x i32> %162, %173
  %175 = add <4 x i32> %160, %174
  %176 = add <4 x i32> %158, %175
  %177 = add <4 x i32> %156, %176
  %178 = add <4 x i32> %154, %177
  %179 = add <4 x i32> %152, %178
  %180 = add <4 x i32> %150, %179
  %181 = add <4 x i32> %148, %180
  %182 = add <4 x i32> %146, %181
  %183 = add <4 x i32> %144, %182
  %184 = add <4 x i32> %142, %183
  %185 = getelementptr inbounds nuw i8, ptr %130, i64 448
  %186 = load <4 x i32>, ptr %185, align 4, !tbaa !16, !noalias !36
  %187 = getelementptr inbounds nuw i8, ptr %130, i64 416
  %188 = load <4 x i32>, ptr %187, align 4, !tbaa !16, !noalias !36
  %189 = getelementptr inbounds nuw i8, ptr %130, i64 384
  %190 = load <4 x i32>, ptr %189, align 4, !tbaa !16, !noalias !36
  %191 = getelementptr inbounds nuw i8, ptr %130, i64 352
  %192 = load <4 x i32>, ptr %191, align 4, !tbaa !16, !noalias !36
  %193 = getelementptr inbounds nuw i8, ptr %130, i64 320
  %194 = load <4 x i32>, ptr %193, align 4, !tbaa !16, !noalias !36
  %195 = getelementptr inbounds nuw i8, ptr %130, i64 288
  %196 = load <4 x i32>, ptr %195, align 4, !tbaa !16, !noalias !36
  %197 = getelementptr inbounds nuw i8, ptr %130, i64 256
  %198 = load <4 x i32>, ptr %197, align 4, !tbaa !16, !noalias !36
  %199 = getelementptr inbounds nuw i8, ptr %130, i64 224
  %200 = load <4 x i32>, ptr %199, align 4, !tbaa !16, !noalias !36
  %201 = getelementptr inbounds nuw i8, ptr %130, i64 192
  %202 = load <4 x i32>, ptr %201, align 4, !tbaa !16, !noalias !36
  %203 = getelementptr inbounds nuw i8, ptr %130, i64 160
  %204 = load <4 x i32>, ptr %203, align 4, !tbaa !16, !noalias !36
  %205 = getelementptr inbounds nuw i8, ptr %130, i64 128
  %206 = load <4 x i32>, ptr %205, align 4, !tbaa !16, !noalias !36
  %207 = getelementptr inbounds nuw i8, ptr %130, i64 96
  %208 = load <4 x i32>, ptr %207, align 4, !tbaa !16, !noalias !36
  %209 = getelementptr inbounds nuw i8, ptr %130, i64 64
  %210 = load <4 x i32>, ptr %209, align 4, !tbaa !16, !noalias !36
  %211 = getelementptr inbounds nuw i8, ptr %130, i64 32
  %212 = load <4 x i32>, ptr %211, align 4, !tbaa !16, !noalias !36
  %213 = load <4 x i32>, ptr %130, align 4, !tbaa !16, !noalias !36
  %214 = load i32, ptr %140, align 4, !tbaa !16, !noalias !36
  %215 = getelementptr inbounds nuw i8, ptr %130, i64 484
  %216 = load i32, ptr %215, align 4, !tbaa !16, !noalias !36
  %217 = getelementptr inbounds nuw i8, ptr %130, i64 488
  %218 = load i32, ptr %217, align 4, !tbaa !16, !noalias !36
  br label %219

219:                                              ; preds = %139, %219
  %220 = phi i32 [ 333, %139 ], [ %269, %219 ]
  %221 = phi i64 [ 0, %139 ], [ %268, %219 ]
  %222 = insertelement <4 x i32> poison, i32 %220, i64 0
  %223 = shufflevector <4 x i32> %222, <4 x i32> poison, <4 x i32> zeroinitializer
  %224 = mul <4 x i32> %184, %223
  %225 = mul <4 x i32> %186, %223
  %226 = mul <4 x i32> %188, %223
  %227 = mul <4 x i32> %190, %223
  %228 = mul <4 x i32> %192, %223
  %229 = mul <4 x i32> %194, %223
  %230 = mul <4 x i32> %196, %223
  %231 = mul <4 x i32> %198, %223
  %232 = mul <4 x i32> %200, %223
  %233 = mul <4 x i32> %202, %223
  %234 = mul <4 x i32> %204, %223
  %235 = mul <4 x i32> %206, %223
  %236 = mul <4 x i32> %208, %223
  %237 = mul <4 x i32> %210, %223
  %238 = mul <4 x i32> %212, %223
  %239 = mul <4 x i32> %213, %223
  %240 = getelementptr inbounds nuw i32, ptr %130, i64 %221
  %241 = load i32, ptr %240, align 4, !tbaa !16, !noalias !36
  %242 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %241, i64 0
  %243 = add <4 x i32> %239, %242
  %244 = add <4 x i32> %238, %243
  %245 = add <4 x i32> %237, %244
  %246 = add <4 x i32> %236, %245
  %247 = add <4 x i32> %235, %246
  %248 = add <4 x i32> %234, %247
  %249 = add <4 x i32> %233, %248
  %250 = add <4 x i32> %232, %249
  %251 = add <4 x i32> %231, %250
  %252 = add <4 x i32> %230, %251
  %253 = add <4 x i32> %229, %252
  %254 = add <4 x i32> %228, %253
  %255 = add <4 x i32> %227, %254
  %256 = add <4 x i32> %226, %255
  %257 = add <4 x i32> %225, %256
  %258 = add <4 x i32> %224, %257
  %259 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %258)
  %260 = mul i32 %214, %220
  %261 = add i32 %260, %259
  %262 = mul i32 %216, %220
  %263 = add i32 %262, %261
  %264 = mul i32 %218, %220
  %265 = add i32 %264, %263
  %266 = add i32 %265, %220
  %267 = getelementptr inbounds nuw i32, ptr %127, i64 %221
  store i32 %266, ptr %267, align 4, !tbaa !16, !alias.scope !36
  %268 = add nuw nsw i64 %221, 1
  %269 = add nuw nsw i32 %220, 12
  %270 = icmp eq i64 %268, 123
  br i1 %270, label %271, label %219, !llvm.loop !39

271:                                              ; preds = %219
  call void @llvm.experimental.noalias.scope.decl(metadata !40)
  %272 = getelementptr inbounds nuw i8, ptr %130, i64 480
  %273 = getelementptr inbounds nuw i8, ptr %130, i64 464
  %274 = load <4 x i32>, ptr %273, align 4, !tbaa !16, !noalias !40
  %275 = getelementptr inbounds nuw i8, ptr %130, i64 432
  %276 = load <4 x i32>, ptr %275, align 4, !tbaa !16, !noalias !40
  %277 = getelementptr inbounds nuw i8, ptr %130, i64 400
  %278 = load <4 x i32>, ptr %277, align 4, !tbaa !16, !noalias !40
  %279 = getelementptr inbounds nuw i8, ptr %130, i64 368
  %280 = load <4 x i32>, ptr %279, align 4, !tbaa !16, !noalias !40
  %281 = getelementptr inbounds nuw i8, ptr %130, i64 336
  %282 = load <4 x i32>, ptr %281, align 4, !tbaa !16, !noalias !40
  %283 = getelementptr inbounds nuw i8, ptr %130, i64 304
  %284 = load <4 x i32>, ptr %283, align 4, !tbaa !16, !noalias !40
  %285 = getelementptr inbounds nuw i8, ptr %130, i64 272
  %286 = load <4 x i32>, ptr %285, align 4, !tbaa !16, !noalias !40
  %287 = getelementptr inbounds nuw i8, ptr %130, i64 240
  %288 = load <4 x i32>, ptr %287, align 4, !tbaa !16, !noalias !40
  %289 = getelementptr inbounds nuw i8, ptr %130, i64 208
  %290 = load <4 x i32>, ptr %289, align 4, !tbaa !16, !noalias !40
  %291 = getelementptr inbounds nuw i8, ptr %130, i64 176
  %292 = load <4 x i32>, ptr %291, align 4, !tbaa !16, !noalias !40
  %293 = getelementptr inbounds nuw i8, ptr %130, i64 144
  %294 = load <4 x i32>, ptr %293, align 4, !tbaa !16, !noalias !40
  %295 = getelementptr inbounds nuw i8, ptr %130, i64 112
  %296 = load <4 x i32>, ptr %295, align 4, !tbaa !16, !noalias !40
  %297 = getelementptr inbounds nuw i8, ptr %130, i64 80
  %298 = load <4 x i32>, ptr %297, align 4, !tbaa !16, !noalias !40
  %299 = getelementptr inbounds nuw i8, ptr %130, i64 48
  %300 = load <4 x i32>, ptr %299, align 4, !tbaa !16, !noalias !40
  %301 = getelementptr inbounds nuw i8, ptr %130, i64 16
  %302 = load <4 x i32>, ptr %301, align 4, !tbaa !16, !noalias !40
  %303 = add <4 x i32> %300, %302
  %304 = add <4 x i32> %298, %303
  %305 = add <4 x i32> %296, %304
  %306 = add <4 x i32> %294, %305
  %307 = add <4 x i32> %292, %306
  %308 = add <4 x i32> %290, %307
  %309 = add <4 x i32> %288, %308
  %310 = add <4 x i32> %286, %309
  %311 = add <4 x i32> %284, %310
  %312 = add <4 x i32> %282, %311
  %313 = add <4 x i32> %280, %312
  %314 = add <4 x i32> %278, %313
  %315 = add <4 x i32> %276, %314
  %316 = add <4 x i32> %274, %315
  %317 = getelementptr inbounds nuw i8, ptr %130, i64 448
  %318 = load <4 x i32>, ptr %317, align 4, !tbaa !16, !noalias !40
  %319 = getelementptr inbounds nuw i8, ptr %130, i64 416
  %320 = load <4 x i32>, ptr %319, align 4, !tbaa !16, !noalias !40
  %321 = getelementptr inbounds nuw i8, ptr %130, i64 384
  %322 = load <4 x i32>, ptr %321, align 4, !tbaa !16, !noalias !40
  %323 = getelementptr inbounds nuw i8, ptr %130, i64 352
  %324 = load <4 x i32>, ptr %323, align 4, !tbaa !16, !noalias !40
  %325 = getelementptr inbounds nuw i8, ptr %130, i64 320
  %326 = load <4 x i32>, ptr %325, align 4, !tbaa !16, !noalias !40
  %327 = getelementptr inbounds nuw i8, ptr %130, i64 288
  %328 = load <4 x i32>, ptr %327, align 4, !tbaa !16, !noalias !40
  %329 = getelementptr inbounds nuw i8, ptr %130, i64 256
  %330 = load <4 x i32>, ptr %329, align 4, !tbaa !16, !noalias !40
  %331 = getelementptr inbounds nuw i8, ptr %130, i64 224
  %332 = load <4 x i32>, ptr %331, align 4, !tbaa !16, !noalias !40
  %333 = getelementptr inbounds nuw i8, ptr %130, i64 192
  %334 = load <4 x i32>, ptr %333, align 4, !tbaa !16, !noalias !40
  %335 = getelementptr inbounds nuw i8, ptr %130, i64 160
  %336 = load <4 x i32>, ptr %335, align 4, !tbaa !16, !noalias !40
  %337 = getelementptr inbounds nuw i8, ptr %130, i64 128
  %338 = load <4 x i32>, ptr %337, align 4, !tbaa !16, !noalias !40
  %339 = getelementptr inbounds nuw i8, ptr %130, i64 96
  %340 = load <4 x i32>, ptr %339, align 4, !tbaa !16, !noalias !40
  %341 = getelementptr inbounds nuw i8, ptr %130, i64 64
  %342 = load <4 x i32>, ptr %341, align 4, !tbaa !16, !noalias !40
  %343 = getelementptr inbounds nuw i8, ptr %130, i64 32
  %344 = load <4 x i32>, ptr %343, align 4, !tbaa !16, !noalias !40
  %345 = load <4 x i32>, ptr %130, align 4, !tbaa !16, !noalias !40
  %346 = load i32, ptr %272, align 4, !tbaa !16, !noalias !40
  %347 = getelementptr inbounds nuw i8, ptr %130, i64 484
  %348 = load i32, ptr %347, align 4, !tbaa !16, !noalias !40
  %349 = getelementptr inbounds nuw i8, ptr %130, i64 488
  %350 = load i32, ptr %349, align 4, !tbaa !16, !noalias !40
  br label %351

351:                                              ; preds = %271, %351
  %352 = phi i32 [ 333, %271 ], [ %401, %351 ]
  %353 = phi i64 [ 0, %271 ], [ %400, %351 ]
  %354 = insertelement <4 x i32> poison, i32 %352, i64 0
  %355 = shufflevector <4 x i32> %354, <4 x i32> poison, <4 x i32> zeroinitializer
  %356 = mul <4 x i32> %316, %355
  %357 = mul <4 x i32> %318, %355
  %358 = mul <4 x i32> %320, %355
  %359 = mul <4 x i32> %322, %355
  %360 = mul <4 x i32> %324, %355
  %361 = mul <4 x i32> %326, %355
  %362 = mul <4 x i32> %328, %355
  %363 = mul <4 x i32> %330, %355
  %364 = mul <4 x i32> %332, %355
  %365 = mul <4 x i32> %334, %355
  %366 = mul <4 x i32> %336, %355
  %367 = mul <4 x i32> %338, %355
  %368 = mul <4 x i32> %340, %355
  %369 = mul <4 x i32> %342, %355
  %370 = mul <4 x i32> %344, %355
  %371 = mul <4 x i32> %345, %355
  %372 = getelementptr inbounds nuw i32, ptr %130, i64 %353
  %373 = load i32, ptr %372, align 4, !tbaa !16, !noalias !40
  %374 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %373, i64 0
  %375 = add <4 x i32> %371, %374
  %376 = add <4 x i32> %370, %375
  %377 = add <4 x i32> %369, %376
  %378 = add <4 x i32> %368, %377
  %379 = add <4 x i32> %367, %378
  %380 = add <4 x i32> %366, %379
  %381 = add <4 x i32> %365, %380
  %382 = add <4 x i32> %364, %381
  %383 = add <4 x i32> %363, %382
  %384 = add <4 x i32> %362, %383
  %385 = add <4 x i32> %361, %384
  %386 = add <4 x i32> %360, %385
  %387 = add <4 x i32> %359, %386
  %388 = add <4 x i32> %358, %387
  %389 = add <4 x i32> %357, %388
  %390 = add <4 x i32> %356, %389
  %391 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %390)
  %392 = mul i32 %346, %352
  %393 = add i32 %392, %391
  %394 = mul i32 %348, %352
  %395 = add i32 %394, %393
  %396 = mul i32 %350, %352
  %397 = add i32 %396, %395
  %398 = add i32 %397, %352
  %399 = getelementptr inbounds nuw i32, ptr %128, i64 %353
  store i32 %398, ptr %399, align 4, !tbaa !16, !alias.scope !40
  %400 = add nuw nsw i64 %353, 1
  %401 = add nuw nsw i32 %352, 12
  %402 = icmp eq i64 %400, 123
  br i1 %402, label %403, label %351, !llvm.loop !43

403:                                              ; preds = %351
  %404 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(492) %127, ptr noundef nonnull readonly dereferenceable(492) %128, i64 492)
  %405 = icmp eq i32 %404, 0
  br i1 %405, label %409, label %406

406:                                              ; preds = %403
  %407 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.4)
          to label %408 unwind label %640

408:                                              ; preds = %406
  call void @exit(i32 noundef 1) #15
  unreachable

409:                                              ; preds = %403
  call void @_ZdaPv(ptr noundef nonnull %130) #16
  call void @_ZdaPv(ptr noundef nonnull %128) #16
  call void @_ZdaPv(ptr noundef nonnull %127) #16
  %410 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.2, i64 noundef 44)
  %411 = call noalias noundef nonnull dereferenceable(492) ptr @_Znam(i64 noundef 492) #14
  %412 = invoke noalias noundef nonnull dereferenceable(492) ptr @_Znam(i64 noundef 492) #14
          to label %413 unwind label %646

413:                                              ; preds = %409
  %414 = invoke noalias noundef nonnull dereferenceable(492) ptr @_Znam(i64 noundef 492) #14
          to label %415 unwind label %648

415:                                              ; preds = %413
  %416 = invoke noalias noundef nonnull dereferenceable(1824) ptr @_Znam(i64 noundef 1824) #14
          to label %417 unwind label %650

417:                                              ; preds = %415
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #13
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %2, align 8, !tbaa !16
  br label %418

418:                                              ; preds = %421, %417
  %419 = phi i64 [ 0, %417 ], [ %423, %421 ]
  %420 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %2)
          to label %421 unwind label %654

421:                                              ; preds = %418
  %422 = getelementptr inbounds nuw i32, ptr %414, i64 %419
  store i32 %420, ptr %422, align 4, !tbaa !16
  %423 = add nuw nsw i64 %419, 1
  %424 = icmp eq i64 %423, 123
  br i1 %424, label %425, label %418, !llvm.loop !18

425:                                              ; preds = %421
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #13
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %1, align 8, !tbaa !16
  br label %426

426:                                              ; preds = %429, %425
  %427 = phi i64 [ 0, %425 ], [ %431, %429 ]
  %428 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %1)
          to label %429 unwind label %652

429:                                              ; preds = %426
  %430 = getelementptr inbounds nuw i32, ptr %416, i64 %427
  store i32 %428, ptr %430, align 4, !tbaa !16
  %431 = add nuw nsw i64 %427, 1
  %432 = icmp eq i64 %431, 456
  br i1 %432, label %433, label %426, !llvm.loop !18

433:                                              ; preds = %429
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #13
  call void @llvm.experimental.noalias.scope.decl(metadata !44)
  br label %434

434:                                              ; preds = %433, %459
  %435 = phi i64 [ 0, %433 ], [ %462, %459 ]
  br label %436

436:                                              ; preds = %436, %434
  %437 = phi i64 [ 0, %434 ], [ %457, %436 ]
  %438 = phi i32 [ 0, %434 ], [ %455, %436 ]
  %439 = phi i32 [ 0, %434 ], [ %456, %436 ]
  %440 = shl nuw nsw i64 %437, 3
  %441 = shl i64 %437, 3
  %442 = getelementptr inbounds nuw i8, ptr %416, i64 %440
  %443 = getelementptr inbounds nuw i8, ptr %416, i64 %441
  %444 = getelementptr inbounds nuw i8, ptr %443, i64 8
  %445 = load i32, ptr %442, align 4, !tbaa !16, !noalias !44
  %446 = load i32, ptr %444, align 4, !tbaa !16, !noalias !44
  %447 = sext i32 %445 to i64
  %448 = sext i32 %446 to i64
  %449 = urem i64 %447, 123
  %450 = urem i64 %448, 123
  %451 = getelementptr inbounds nuw i32, ptr %414, i64 %449
  %452 = getelementptr inbounds nuw i32, ptr %414, i64 %450
  %453 = load i32, ptr %451, align 4, !tbaa !16, !noalias !44
  %454 = load i32, ptr %452, align 4, !tbaa !16, !noalias !44
  %455 = add i32 %453, %438
  %456 = add i32 %454, %439
  %457 = add nuw i64 %437, 2
  %458 = icmp eq i64 %457, 228
  br i1 %458, label %459, label %436, !llvm.loop !47

459:                                              ; preds = %436
  %460 = add i32 %456, %455
  %461 = getelementptr inbounds nuw i32, ptr %411, i64 %435
  store i32 %460, ptr %461, align 4, !tbaa !16, !alias.scope !44
  %462 = add nuw nsw i64 %435, 1
  %463 = icmp eq i64 %462, 123
  br i1 %463, label %464, label %434, !llvm.loop !48

464:                                              ; preds = %459
  call void @llvm.experimental.noalias.scope.decl(metadata !49)
  br label %465

465:                                              ; preds = %464, %490
  %466 = phi i64 [ 0, %464 ], [ %493, %490 ]
  br label %467

467:                                              ; preds = %467, %465
  %468 = phi i64 [ 0, %465 ], [ %488, %467 ]
  %469 = phi i32 [ 0, %465 ], [ %486, %467 ]
  %470 = phi i32 [ 0, %465 ], [ %487, %467 ]
  %471 = shl nuw nsw i64 %468, 3
  %472 = shl i64 %468, 3
  %473 = getelementptr inbounds nuw i8, ptr %416, i64 %471
  %474 = getelementptr inbounds nuw i8, ptr %416, i64 %472
  %475 = getelementptr inbounds nuw i8, ptr %474, i64 8
  %476 = load i32, ptr %473, align 4, !tbaa !16, !noalias !49
  %477 = load i32, ptr %475, align 4, !tbaa !16, !noalias !49
  %478 = sext i32 %476 to i64
  %479 = sext i32 %477 to i64
  %480 = urem i64 %478, 123
  %481 = urem i64 %479, 123
  %482 = getelementptr inbounds nuw i32, ptr %414, i64 %480
  %483 = getelementptr inbounds nuw i32, ptr %414, i64 %481
  %484 = load i32, ptr %482, align 4, !tbaa !16, !noalias !49
  %485 = load i32, ptr %483, align 4, !tbaa !16, !noalias !49
  %486 = add i32 %484, %469
  %487 = add i32 %485, %470
  %488 = add nuw i64 %468, 2
  %489 = icmp eq i64 %488, 228
  br i1 %489, label %490, label %467, !llvm.loop !52

490:                                              ; preds = %467
  %491 = add i32 %487, %486
  %492 = getelementptr inbounds nuw i32, ptr %412, i64 %466
  store i32 %491, ptr %492, align 4, !tbaa !16, !alias.scope !49
  %493 = add nuw nsw i64 %466, 1
  %494 = icmp eq i64 %493, 123
  br i1 %494, label %495, label %465, !llvm.loop !53

495:                                              ; preds = %490
  %496 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(492) %411, ptr noundef nonnull readonly dereferenceable(492) %412, i64 492)
  %497 = icmp eq i32 %496, 0
  br i1 %497, label %501, label %498

498:                                              ; preds = %495
  %499 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.4)
          to label %500 unwind label %656

500:                                              ; preds = %498
  call void @exit(i32 noundef 1) #15
  unreachable

501:                                              ; preds = %495
  call void @_ZdaPv(ptr noundef nonnull %416) #16
  call void @_ZdaPv(ptr noundef nonnull %414) #16
  call void @_ZdaPv(ptr noundef nonnull %412) #16
  call void @_ZdaPv(ptr noundef nonnull %411) #16
  %502 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3, i64 noundef 26)
  %503 = call noalias noundef nonnull dereferenceable(1483380) ptr @_Znam(i64 noundef 1483380) #14
  %504 = invoke noalias noundef nonnull dereferenceable(1483380) ptr @_Znam(i64 noundef 1483380) #14
          to label %505 unwind label %664

505:                                              ; preds = %501, %554
  %506 = phi i64 [ %555, %554 ], [ 0, %501 ]
  %507 = mul nuw nsw i64 %506, 12060
  %508 = getelementptr inbounds nuw i8, ptr %503, i64 %507
  br label %509

509:                                              ; preds = %509, %505
  %510 = phi i64 [ 0, %505 ], [ %552, %509 ]
  %511 = mul nuw nsw i64 %510, %506
  %512 = mul nuw nsw i64 %510, 268
  %513 = getelementptr inbounds nuw i8, ptr %508, i64 %512
  store i32 0, ptr %513, align 4, !tbaa !16, !alias.scope !54
  %514 = trunc i64 %511 to i32
  %515 = getelementptr inbounds nuw i8, ptr %513, i64 4
  store i32 %514, ptr %515, align 4, !tbaa !16, !alias.scope !54
  %516 = getelementptr inbounds nuw i8, ptr %513, i64 8
  %517 = insertelement <4 x i32> poison, i32 %514, i64 0
  %518 = shufflevector <4 x i32> %517, <4 x i32> poison, <4 x i32> zeroinitializer
  %519 = mul <4 x i32> %518, <i32 2, i32 3, i32 4, i32 5>
  store <4 x i32> %519, ptr %516, align 4, !tbaa !16, !alias.scope !54
  %520 = getelementptr inbounds nuw i8, ptr %513, i64 24
  %521 = mul <4 x i32> %518, <i32 6, i32 7, i32 8, i32 9>
  store <4 x i32> %521, ptr %520, align 4, !tbaa !16, !alias.scope !54
  %522 = getelementptr inbounds nuw i8, ptr %513, i64 40
  %523 = mul <4 x i32> %518, <i32 10, i32 11, i32 12, i32 13>
  store <4 x i32> %523, ptr %522, align 4, !tbaa !16, !alias.scope !54
  %524 = getelementptr inbounds nuw i8, ptr %513, i64 56
  %525 = mul <4 x i32> %518, <i32 14, i32 15, i32 16, i32 17>
  store <4 x i32> %525, ptr %524, align 4, !tbaa !16, !alias.scope !54
  %526 = getelementptr inbounds nuw i8, ptr %513, i64 72
  %527 = mul <4 x i32> %518, <i32 18, i32 19, i32 20, i32 21>
  store <4 x i32> %527, ptr %526, align 4, !tbaa !16, !alias.scope !54
  %528 = getelementptr inbounds nuw i8, ptr %513, i64 88
  %529 = mul <4 x i32> %518, <i32 22, i32 23, i32 24, i32 25>
  store <4 x i32> %529, ptr %528, align 4, !tbaa !16, !alias.scope !54
  %530 = getelementptr inbounds nuw i8, ptr %513, i64 104
  %531 = mul <4 x i32> %518, <i32 26, i32 27, i32 28, i32 29>
  store <4 x i32> %531, ptr %530, align 4, !tbaa !16, !alias.scope !54
  %532 = getelementptr inbounds nuw i8, ptr %513, i64 120
  %533 = mul <4 x i32> %518, <i32 30, i32 31, i32 32, i32 33>
  store <4 x i32> %533, ptr %532, align 4, !tbaa !16, !alias.scope !54
  %534 = getelementptr inbounds nuw i8, ptr %513, i64 136
  %535 = mul <4 x i32> %518, <i32 34, i32 35, i32 36, i32 37>
  store <4 x i32> %535, ptr %534, align 4, !tbaa !16, !alias.scope !54
  %536 = getelementptr inbounds nuw i8, ptr %513, i64 152
  %537 = mul <4 x i32> %518, <i32 38, i32 39, i32 40, i32 41>
  store <4 x i32> %537, ptr %536, align 4, !tbaa !16, !alias.scope !54
  %538 = getelementptr inbounds nuw i8, ptr %513, i64 168
  %539 = mul <4 x i32> %518, <i32 42, i32 43, i32 44, i32 45>
  store <4 x i32> %539, ptr %538, align 4, !tbaa !16, !alias.scope !54
  %540 = getelementptr inbounds nuw i8, ptr %513, i64 184
  %541 = mul <4 x i32> %518, <i32 46, i32 47, i32 48, i32 49>
  store <4 x i32> %541, ptr %540, align 4, !tbaa !16, !alias.scope !54
  %542 = getelementptr inbounds nuw i8, ptr %513, i64 200
  %543 = mul <4 x i32> %518, <i32 50, i32 51, i32 52, i32 53>
  store <4 x i32> %543, ptr %542, align 4, !tbaa !16, !alias.scope !54
  %544 = getelementptr inbounds nuw i8, ptr %513, i64 216
  %545 = mul <4 x i32> %518, <i32 54, i32 55, i32 56, i32 57>
  store <4 x i32> %545, ptr %544, align 4, !tbaa !16, !alias.scope !54
  %546 = getelementptr inbounds nuw i8, ptr %513, i64 232
  %547 = mul <4 x i32> %518, <i32 58, i32 59, i32 60, i32 61>
  store <4 x i32> %547, ptr %546, align 4, !tbaa !16, !alias.scope !54
  %548 = getelementptr inbounds nuw i8, ptr %513, i64 248
  %549 = mul <4 x i32> %518, <i32 62, i32 63, i32 64, i32 65>
  store <4 x i32> %549, ptr %548, align 4, !tbaa !16, !alias.scope !54
  %550 = mul i32 %514, 66
  %551 = getelementptr inbounds nuw i8, ptr %513, i64 264
  store i32 %550, ptr %551, align 4, !tbaa !16, !alias.scope !54
  %552 = add nuw nsw i64 %510, 1
  %553 = icmp eq i64 %552, 45
  br i1 %553, label %554, label %509, !llvm.loop !57

554:                                              ; preds = %509
  %555 = add nuw nsw i64 %506, 1
  %556 = icmp eq i64 %555, 123
  br i1 %556, label %557, label %505, !llvm.loop !58

557:                                              ; preds = %554, %606
  %558 = phi i64 [ %607, %606 ], [ 0, %554 ]
  %559 = mul nuw nsw i64 %558, 12060
  %560 = getelementptr inbounds nuw i8, ptr %504, i64 %559
  br label %561

561:                                              ; preds = %561, %557
  %562 = phi i64 [ 0, %557 ], [ %604, %561 ]
  %563 = mul nuw nsw i64 %562, %558
  %564 = mul nuw nsw i64 %562, 268
  %565 = getelementptr inbounds nuw i8, ptr %560, i64 %564
  store i32 0, ptr %565, align 4, !tbaa !16, !alias.scope !59
  %566 = trunc i64 %563 to i32
  %567 = getelementptr inbounds nuw i8, ptr %565, i64 4
  store i32 %566, ptr %567, align 4, !tbaa !16, !alias.scope !59
  %568 = getelementptr inbounds nuw i8, ptr %565, i64 8
  %569 = insertelement <4 x i32> poison, i32 %566, i64 0
  %570 = shufflevector <4 x i32> %569, <4 x i32> poison, <4 x i32> zeroinitializer
  %571 = mul <4 x i32> %570, <i32 2, i32 3, i32 4, i32 5>
  store <4 x i32> %571, ptr %568, align 4, !tbaa !16, !alias.scope !59
  %572 = getelementptr inbounds nuw i8, ptr %565, i64 24
  %573 = mul <4 x i32> %570, <i32 6, i32 7, i32 8, i32 9>
  store <4 x i32> %573, ptr %572, align 4, !tbaa !16, !alias.scope !59
  %574 = getelementptr inbounds nuw i8, ptr %565, i64 40
  %575 = mul <4 x i32> %570, <i32 10, i32 11, i32 12, i32 13>
  store <4 x i32> %575, ptr %574, align 4, !tbaa !16, !alias.scope !59
  %576 = getelementptr inbounds nuw i8, ptr %565, i64 56
  %577 = mul <4 x i32> %570, <i32 14, i32 15, i32 16, i32 17>
  store <4 x i32> %577, ptr %576, align 4, !tbaa !16, !alias.scope !59
  %578 = getelementptr inbounds nuw i8, ptr %565, i64 72
  %579 = mul <4 x i32> %570, <i32 18, i32 19, i32 20, i32 21>
  store <4 x i32> %579, ptr %578, align 4, !tbaa !16, !alias.scope !59
  %580 = getelementptr inbounds nuw i8, ptr %565, i64 88
  %581 = mul <4 x i32> %570, <i32 22, i32 23, i32 24, i32 25>
  store <4 x i32> %581, ptr %580, align 4, !tbaa !16, !alias.scope !59
  %582 = getelementptr inbounds nuw i8, ptr %565, i64 104
  %583 = mul <4 x i32> %570, <i32 26, i32 27, i32 28, i32 29>
  store <4 x i32> %583, ptr %582, align 4, !tbaa !16, !alias.scope !59
  %584 = getelementptr inbounds nuw i8, ptr %565, i64 120
  %585 = mul <4 x i32> %570, <i32 30, i32 31, i32 32, i32 33>
  store <4 x i32> %585, ptr %584, align 4, !tbaa !16, !alias.scope !59
  %586 = getelementptr inbounds nuw i8, ptr %565, i64 136
  %587 = mul <4 x i32> %570, <i32 34, i32 35, i32 36, i32 37>
  store <4 x i32> %587, ptr %586, align 4, !tbaa !16, !alias.scope !59
  %588 = getelementptr inbounds nuw i8, ptr %565, i64 152
  %589 = mul <4 x i32> %570, <i32 38, i32 39, i32 40, i32 41>
  store <4 x i32> %589, ptr %588, align 4, !tbaa !16, !alias.scope !59
  %590 = getelementptr inbounds nuw i8, ptr %565, i64 168
  %591 = mul <4 x i32> %570, <i32 42, i32 43, i32 44, i32 45>
  store <4 x i32> %591, ptr %590, align 4, !tbaa !16, !alias.scope !59
  %592 = getelementptr inbounds nuw i8, ptr %565, i64 184
  %593 = mul <4 x i32> %570, <i32 46, i32 47, i32 48, i32 49>
  store <4 x i32> %593, ptr %592, align 4, !tbaa !16, !alias.scope !59
  %594 = getelementptr inbounds nuw i8, ptr %565, i64 200
  %595 = mul <4 x i32> %570, <i32 50, i32 51, i32 52, i32 53>
  store <4 x i32> %595, ptr %594, align 4, !tbaa !16, !alias.scope !59
  %596 = getelementptr inbounds nuw i8, ptr %565, i64 216
  %597 = mul <4 x i32> %570, <i32 54, i32 55, i32 56, i32 57>
  store <4 x i32> %597, ptr %596, align 4, !tbaa !16, !alias.scope !59
  %598 = getelementptr inbounds nuw i8, ptr %565, i64 232
  %599 = mul <4 x i32> %570, <i32 58, i32 59, i32 60, i32 61>
  store <4 x i32> %599, ptr %598, align 4, !tbaa !16, !alias.scope !59
  %600 = getelementptr inbounds nuw i8, ptr %565, i64 248
  %601 = mul <4 x i32> %570, <i32 62, i32 63, i32 64, i32 65>
  store <4 x i32> %601, ptr %600, align 4, !tbaa !16, !alias.scope !59
  %602 = mul i32 %566, 66
  %603 = getelementptr inbounds nuw i8, ptr %565, i64 264
  store i32 %602, ptr %603, align 4, !tbaa !16, !alias.scope !59
  %604 = add nuw nsw i64 %562, 1
  %605 = icmp eq i64 %604, 45
  br i1 %605, label %606, label %561, !llvm.loop !62

606:                                              ; preds = %561
  %607 = add nuw nsw i64 %558, 1
  %608 = icmp eq i64 %607, 123
  br i1 %608, label %609, label %557, !llvm.loop !63

609:                                              ; preds = %606
  %610 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(1483380) %503, ptr noundef nonnull readonly dereferenceable(1483380) %504, i64 1483380)
  %611 = icmp eq i32 %610, 0
  br i1 %611, label %615, label %612

612:                                              ; preds = %609
  %613 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.4)
          to label %614 unwind label %666

614:                                              ; preds = %612
  call void @exit(i32 noundef 1) #15
  unreachable

615:                                              ; preds = %609
  call void @_ZdaPv(ptr noundef nonnull %504) #16
  call void @_ZdaPv(ptr noundef nonnull %503) #16
  ret i32 0

616:                                              ; preds = %18
  %617 = landingpad { ptr, i32 }
          cleanup
  br label %668

618:                                              ; preds = %23
  %619 = landingpad { ptr, i32 }
          cleanup
  br label %632

620:                                              ; preds = %25
  %621 = landingpad { ptr, i32 }
          cleanup
  br label %630

622:                                              ; preds = %36
  %623 = landingpad { ptr, i32 }
          cleanup
  br label %628

624:                                              ; preds = %28
  %625 = landingpad { ptr, i32 }
          cleanup
  br label %628

626:                                              ; preds = %122
  %627 = landingpad { ptr, i32 }
          cleanup
  br label %628

628:                                              ; preds = %624, %626, %622
  %629 = phi { ptr, i32 } [ %623, %622 ], [ %625, %624 ], [ %627, %626 ]
  call void @_ZdaPv(ptr noundef nonnull %26) #16
  br label %630

630:                                              ; preds = %628, %620
  %631 = phi { ptr, i32 } [ %629, %628 ], [ %621, %620 ]
  call void @_ZdaPv(ptr noundef nonnull %24) #16
  br label %632

632:                                              ; preds = %630, %618
  %633 = phi { ptr, i32 } [ %631, %630 ], [ %619, %618 ]
  call void @_ZdaPv(ptr noundef nonnull %22) #16
  br label %668

634:                                              ; preds = %125
  %635 = landingpad { ptr, i32 }
          cleanup
  br label %668

636:                                              ; preds = %129
  %637 = landingpad { ptr, i32 }
          cleanup
  br label %644

638:                                              ; preds = %132
  %639 = landingpad { ptr, i32 }
          cleanup
  br label %642

640:                                              ; preds = %406
  %641 = landingpad { ptr, i32 }
          cleanup
  br label %642

642:                                              ; preds = %640, %638
  %643 = phi { ptr, i32 } [ %639, %638 ], [ %641, %640 ]
  call void @_ZdaPv(ptr noundef nonnull %130) #16
  br label %644

644:                                              ; preds = %642, %636
  %645 = phi { ptr, i32 } [ %643, %642 ], [ %637, %636 ]
  call void @_ZdaPv(ptr noundef nonnull %128) #16
  br label %668

646:                                              ; preds = %409
  %647 = landingpad { ptr, i32 }
          cleanup
  br label %668

648:                                              ; preds = %413
  %649 = landingpad { ptr, i32 }
          cleanup
  br label %662

650:                                              ; preds = %415
  %651 = landingpad { ptr, i32 }
          cleanup
  br label %660

652:                                              ; preds = %426
  %653 = landingpad { ptr, i32 }
          cleanup
  br label %658

654:                                              ; preds = %418
  %655 = landingpad { ptr, i32 }
          cleanup
  br label %658

656:                                              ; preds = %498
  %657 = landingpad { ptr, i32 }
          cleanup
  br label %658

658:                                              ; preds = %654, %656, %652
  %659 = phi { ptr, i32 } [ %653, %652 ], [ %655, %654 ], [ %657, %656 ]
  call void @_ZdaPv(ptr noundef nonnull %416) #16
  br label %660

660:                                              ; preds = %658, %650
  %661 = phi { ptr, i32 } [ %659, %658 ], [ %651, %650 ]
  call void @_ZdaPv(ptr noundef nonnull %414) #16
  br label %662

662:                                              ; preds = %660, %648
  %663 = phi { ptr, i32 } [ %661, %660 ], [ %649, %648 ]
  call void @_ZdaPv(ptr noundef nonnull %412) #16
  br label %668

664:                                              ; preds = %501
  %665 = landingpad { ptr, i32 }
          cleanup
  br label %668

666:                                              ; preds = %612
  %667 = landingpad { ptr, i32 }
          cleanup
  call void @_ZdaPv(ptr noundef nonnull %504) #16
  br label %668

668:                                              ; preds = %664, %666, %646, %662, %634, %644, %616, %632
  %669 = phi ptr [ %21, %632 ], [ %21, %616 ], [ %127, %644 ], [ %127, %634 ], [ %411, %662 ], [ %411, %646 ], [ %503, %666 ], [ %503, %664 ]
  %670 = phi { ptr, i32 } [ %633, %632 ], [ %617, %616 ], [ %645, %644 ], [ %635, %634 ], [ %663, %662 ], [ %647, %646 ], [ %667, %666 ], [ %665, %664 ]
  call void @_ZdaPv(ptr noundef nonnull %669) #16
  resume { ptr, i32 } %670
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) local_unnamed_addr #3

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #4

declare i32 @__gxx_personality_v0(...)

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #6

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) local_unnamed_addr #7 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<>::param_type", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !64
  %7 = sext i32 %6 to i64
  %8 = load i32, ptr %2, align 4, !tbaa !66
  %9 = sext i32 %8 to i64
  %10 = sub nsw i64 %7, %9
  %11 = icmp ult i64 %10, 4294967295
  br i1 %11, label %12, label %32

12:                                               ; preds = %3
  %13 = trunc nuw i64 %10 to i32
  %14 = add nuw i32 %13, 1
  %15 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %16 = zext i32 %14 to i64
  %17 = mul i64 %15, %16
  %18 = trunc i64 %17 to i32
  %19 = icmp ult i32 %13, %18
  br i1 %19, label %29, label %20

20:                                               ; preds = %12
  %21 = xor i32 %13, -1
  %22 = urem i32 %21, %14
  %23 = icmp ugt i32 %22, %18
  br i1 %23, label %24, label %29

24:                                               ; preds = %20, %24
  %25 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %26 = mul i64 %25, %16
  %27 = trunc i64 %26 to i32
  %28 = icmp ugt i32 %22, %27
  br i1 %28, label %24, label %29, !llvm.loop !67

29:                                               ; preds = %24, %12, %20
  %30 = phi i64 [ %17, %12 ], [ %17, %20 ], [ %26, %24 ]
  %31 = lshr i64 %30, 32
  br label %45

32:                                               ; preds = %3
  %33 = icmp eq i64 %10, 4294967295
  br i1 %33, label %43, label %34

34:                                               ; preds = %32, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #13
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !16
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = sext i32 %35 to i64
  %37 = shl nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #13
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %45, !llvm.loop !68

43:                                               ; preds = %32
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %34, %43, %29
  %46 = phi i64 [ %31, %29 ], [ %44, %43 ], [ %39, %34 ]
  %47 = load i32, ptr %2, align 4, !tbaa !66
  %48 = trunc i64 %46 to i32
  %49 = add i32 %47, %48
  ret i32 %49
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %0) local_unnamed_addr #7 comdat {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4992
  %3 = load i64, ptr %2, align 8, !tbaa !12
  %4 = icmp ugt i64 %3, 623
  br i1 %4, label %5, label %127

5:                                                ; preds = %1
  %6 = load i64, ptr %0, align 8, !tbaa !6
  %7 = insertelement <2 x i64> poison, i64 %6, i64 1
  br label %8

8:                                                ; preds = %8, %5
  %9 = phi i64 [ 0, %5 ], [ %42, %8 ]
  %10 = phi <2 x i64> [ %7, %5 ], [ %16, %8 ]
  %11 = getelementptr inbounds nuw i64, ptr %0, i64 %9
  %12 = getelementptr inbounds nuw i64, ptr %0, i64 %9
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 24
  %15 = load <2 x i64>, ptr %13, align 8, !tbaa !6
  %16 = load <2 x i64>, ptr %14, align 8, !tbaa !6
  %17 = shufflevector <2 x i64> %10, <2 x i64> %15, <2 x i32> <i32 1, i32 2>
  %18 = shufflevector <2 x i64> %15, <2 x i64> %16, <2 x i32> <i32 1, i32 2>
  %19 = and <2 x i64> %17, splat (i64 -2147483648)
  %20 = and <2 x i64> %18, splat (i64 -2147483648)
  %21 = and <2 x i64> %15, splat (i64 2147483646)
  %22 = and <2 x i64> %16, splat (i64 2147483646)
  %23 = or disjoint <2 x i64> %21, %19
  %24 = or disjoint <2 x i64> %22, %20
  %25 = getelementptr inbounds nuw i8, ptr %11, i64 3176
  %26 = getelementptr inbounds nuw i8, ptr %11, i64 3192
  %27 = load <2 x i64>, ptr %25, align 8, !tbaa !6
  %28 = load <2 x i64>, ptr %26, align 8, !tbaa !6
  %29 = lshr exact <2 x i64> %23, splat (i64 1)
  %30 = lshr exact <2 x i64> %24, splat (i64 1)
  %31 = xor <2 x i64> %29, %27
  %32 = xor <2 x i64> %30, %28
  %33 = and <2 x i64> %15, splat (i64 1)
  %34 = and <2 x i64> %16, splat (i64 1)
  %35 = icmp eq <2 x i64> %33, zeroinitializer
  %36 = icmp eq <2 x i64> %34, zeroinitializer
  %37 = select <2 x i1> %35, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %38 = select <2 x i1> %36, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %39 = xor <2 x i64> %31, %37
  %40 = xor <2 x i64> %32, %38
  %41 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store <2 x i64> %39, ptr %11, align 8, !tbaa !6
  store <2 x i64> %40, ptr %41, align 8, !tbaa !6
  %42 = add nuw i64 %9, 4
  %43 = icmp eq i64 %42, 224
  br i1 %43, label %44, label %8, !llvm.loop !69

44:                                               ; preds = %8
  %45 = extractelement <2 x i64> %16, i64 1
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 1792
  %47 = and i64 %45, -2147483648
  %48 = getelementptr inbounds nuw i8, ptr %0, i64 1800
  %49 = load i64, ptr %48, align 8, !tbaa !6
  %50 = and i64 %49, 2147483646
  %51 = or disjoint i64 %50, %47
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 4968
  %53 = load i64, ptr %52, align 8, !tbaa !6
  %54 = lshr exact i64 %51, 1
  %55 = xor i64 %54, %53
  %56 = and i64 %49, 1
  %57 = icmp eq i64 %56, 0
  %58 = select i1 %57, i64 0, i64 2567483615
  %59 = xor i64 %55, %58
  store i64 %59, ptr %46, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 1800
  %61 = and i64 %49, -2147483648
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 1808
  %63 = load i64, ptr %62, align 8, !tbaa !6
  %64 = and i64 %63, 2147483646
  %65 = or disjoint i64 %64, %61
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 4976
  %67 = load i64, ptr %66, align 8, !tbaa !6
  %68 = lshr exact i64 %65, 1
  %69 = xor i64 %68, %67
  %70 = and i64 %63, 1
  %71 = icmp eq i64 %70, 0
  %72 = select i1 %71, i64 0, i64 2567483615
  %73 = xor i64 %69, %72
  store i64 %73, ptr %60, align 8, !tbaa !6
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 1808
  %75 = and i64 %63, -2147483648
  %76 = getelementptr inbounds nuw i8, ptr %0, i64 1816
  %77 = load i64, ptr %76, align 8, !tbaa !6
  %78 = and i64 %77, 2147483646
  %79 = or disjoint i64 %78, %75
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 4984
  %81 = load i64, ptr %80, align 8, !tbaa !6
  %82 = lshr exact i64 %79, 1
  %83 = xor i64 %82, %81
  %84 = and i64 %77, 1
  %85 = icmp eq i64 %84, 0
  %86 = select i1 %85, i64 0, i64 2567483615
  %87 = xor i64 %83, %86
  store i64 %87, ptr %74, align 8, !tbaa !6
  %88 = getelementptr inbounds nuw i8, ptr %0, i64 1816
  %89 = load i64, ptr %88, align 8, !tbaa !6
  %90 = insertelement <2 x i64> poison, i64 %89, i64 1
  br label %91

91:                                               ; preds = %91, %44
  %92 = phi i64 [ 0, %44 ], [ %110, %91 ]
  %93 = phi <2 x i64> [ %90, %44 ], [ %98, %91 ]
  %94 = getelementptr i64, ptr %0, i64 %92
  %95 = getelementptr i8, ptr %94, i64 1816
  %96 = getelementptr i64, ptr %0, i64 %92
  %97 = getelementptr i8, ptr %96, i64 1824
  %98 = load <2 x i64>, ptr %97, align 8, !tbaa !6
  %99 = shufflevector <2 x i64> %93, <2 x i64> %98, <2 x i32> <i32 1, i32 2>
  %100 = and <2 x i64> %99, splat (i64 -2147483648)
  %101 = and <2 x i64> %98, splat (i64 2147483646)
  %102 = or disjoint <2 x i64> %101, %100
  %103 = load <2 x i64>, ptr %94, align 8, !tbaa !6
  %104 = lshr exact <2 x i64> %102, splat (i64 1)
  %105 = xor <2 x i64> %104, %103
  %106 = and <2 x i64> %98, splat (i64 1)
  %107 = icmp eq <2 x i64> %106, zeroinitializer
  %108 = select <2 x i1> %107, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %109 = xor <2 x i64> %105, %108
  store <2 x i64> %109, ptr %95, align 8, !tbaa !6
  %110 = add nuw i64 %92, 2
  %111 = icmp eq i64 %110, 396
  br i1 %111, label %112, label %91, !llvm.loop !70

112:                                              ; preds = %91
  %113 = getelementptr inbounds nuw i8, ptr %0, i64 4984
  %114 = load i64, ptr %113, align 8, !tbaa !6
  %115 = and i64 %114, -2147483648
  %116 = load i64, ptr %0, align 8, !tbaa !6
  %117 = and i64 %116, 2147483646
  %118 = or disjoint i64 %117, %115
  %119 = getelementptr inbounds nuw i8, ptr %0, i64 3168
  %120 = load i64, ptr %119, align 8, !tbaa !6
  %121 = lshr exact i64 %118, 1
  %122 = xor i64 %121, %120
  %123 = and i64 %116, 1
  %124 = icmp eq i64 %123, 0
  %125 = select i1 %124, i64 0, i64 2567483615
  %126 = xor i64 %122, %125
  store i64 %126, ptr %113, align 8, !tbaa !6
  br label %127

127:                                              ; preds = %112, %1
  %128 = phi i64 [ 0, %112 ], [ %3, %1 ]
  %129 = add nuw nsw i64 %128, 1
  store i64 %129, ptr %2, align 8, !tbaa !12
  %130 = getelementptr inbounds nuw i64, ptr %0, i64 %128
  %131 = load i64, ptr %130, align 8, !tbaa !6
  %132 = lshr i64 %131, 11
  %133 = and i64 %132, 4294967295
  %134 = xor i64 %133, %131
  %135 = shl i64 %134, 7
  %136 = and i64 %135, 2636928640
  %137 = xor i64 %136, %134
  %138 = shl i64 %137, 15
  %139 = and i64 %138, 4022730752
  %140 = xor i64 %139, %137
  %141 = lshr i64 %140, 18
  %142 = xor i64 %141, %140
  ret i64 %142
}

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #8

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_outer_loop_vect.cpp() #9 section ".text.startup" {
  store i64 5489, ptr @_ZL3rng, align 8, !tbaa !6
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 5489, %0 ], [ %9, %1 ]
  %3 = phi i64 [ 1, %0 ], [ %10, %1 ]
  %4 = getelementptr i64, ptr @_ZL3rng, i64 %3
  %5 = lshr i64 %2, 30
  %6 = xor i64 %5, %2
  %7 = mul nuw nsw i64 %6, 1812433253
  %8 = add nuw i64 %7, %3
  %9 = and i64 %8, 4294967295
  store i64 %9, ptr %4, align 8, !tbaa !6
  %10 = add nuw nsw i64 %3, 1
  %11 = icmp eq i64 %10, 624
  br i1 %11, label %12, label %1, !llvm.loop !10

12:                                               ; preds = %1
  store i64 624, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4992), align 8, !tbaa !12
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #10

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #12

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #11 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #12 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nounwind }
attributes #14 = { builtin allocsize(0) }
attributes #15 = { cold noreturn nounwind }
attributes #16 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!13, !7, i64 4992}
!13 = !{!"_ZTSSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE", !8, i64 0, !7, i64 4992}
!14 = !{i64 0, i64 4992, !15, i64 4992, i64 8, !6}
!15 = !{!8, !8, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !8, i64 0}
!18 = distinct !{!18, !11}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZZ4mainENK3$_0clEmmmPiPKiS2_: argument 0"}
!21 = distinct !{!21, !"_ZZ4mainENK3$_0clEmmmPiPKiS2_"}
!22 = distinct !{!22, !11, !23, !24}
!23 = !{!"llvm.loop.isvectorized", i32 1}
!24 = !{!"llvm.loop.unroll.runtime.disable"}
!25 = distinct !{!25, !11, !26, !27}
!26 = !{!"llvm.loop.vectorize.width", i32 1}
!27 = !{!"llvm.loop.interleave.count", i32 1}
!28 = distinct !{!28, !11}
!29 = !{!30}
!30 = distinct !{!30, !31, !"_ZZ4mainENK3$_1clEmmmPiPKiS2_: argument 0"}
!31 = distinct !{!31, !"_ZZ4mainENK3$_1clEmmmPiPKiS2_"}
!32 = distinct !{!32, !11, !23, !24}
!33 = distinct !{!33, !11, !34}
!34 = !{!"llvm.loop.vectorize.enable", i1 true}
!35 = distinct !{!35, !11}
!36 = !{!37}
!37 = distinct !{!37, !38, !"_ZZ4mainENK3$_2clEmPiPKi: argument 0"}
!38 = distinct !{!38, !"_ZZ4mainENK3$_2clEmPiPKi"}
!39 = distinct !{!39, !11, !26, !27}
!40 = !{!41}
!41 = distinct !{!41, !42, !"_ZZ4mainENK3$_3clEmPiPKi: argument 0"}
!42 = distinct !{!42, !"_ZZ4mainENK3$_3clEmPiPKi"}
!43 = distinct !{!43, !11, !34}
!44 = !{!45}
!45 = distinct !{!45, !46, !"_ZZ4mainENK3$_4clEmmPiPKiS2_: argument 0"}
!46 = distinct !{!46, !"_ZZ4mainENK3$_4clEmmPiPKiS2_"}
!47 = distinct !{!47, !11, !23, !24}
!48 = distinct !{!48, !11, !26, !27}
!49 = !{!50}
!50 = distinct !{!50, !51, !"_ZZ4mainENK3$_5clEmmPiPKiS2_: argument 0"}
!51 = distinct !{!51, !"_ZZ4mainENK3$_5clEmmPiPKiS2_"}
!52 = distinct !{!52, !11, !23, !24}
!53 = distinct !{!53, !11, !34}
!54 = !{!55}
!55 = distinct !{!55, !56, !"_ZZ4mainENK3$_6clEmmmPi: argument 0"}
!56 = distinct !{!56, !"_ZZ4mainENK3$_6clEmmmPi"}
!57 = distinct !{!57, !11}
!58 = distinct !{!58, !11, !26, !27}
!59 = !{!60}
!60 = distinct !{!60, !61, !"_ZZ4mainENK3$_7clEmmmPi: argument 0"}
!61 = distinct !{!61, !"_ZZ4mainENK3$_7clEmmmPi"}
!62 = distinct !{!62, !11}
!63 = distinct !{!63, !11, !34}
!64 = !{!65, !17, i64 4}
!65 = !{!"_ZTSNSt24uniform_int_distributionIiE10param_typeE", !17, i64 0, !17, i64 4}
!66 = !{!65, !17, i64 0}
!67 = distinct !{!67, !11}
!68 = distinct !{!68, !11}
!69 = distinct !{!69, !11, !23, !24}
!70 = distinct !{!70, !11, !23, !24}
