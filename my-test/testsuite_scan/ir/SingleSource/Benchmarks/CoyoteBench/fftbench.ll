; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/fftbench.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/fftbench.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%class.polynomial = type { ptr, ptr, i64 }
%class.polynomial.0 = type { ptr, ptr, i64 }
%"class.std::complex" = type { { double, double } }

$_ZNK10polynomialIdEmlERKS0_ = comdat any

$_ZN10polynomialIdED2Ev = comdat any

$_ZN10polynomialIdED0Ev = comdat any

$_ZN10polynomialIdE11stretch_fftEv = comdat any

$_ZN10polynomialIdE3fftERKS0_ = comdat any

$_ZN10polynomialIdE11inverse_fftERKS_ISt7complexIdEE = comdat any

$_ZN10polynomialISt7complexIdEED2Ev = comdat any

$_ZN10polynomialIdE4log2Em = comdat any

$_ZN10polynomialISt7complexIdEED0Ev = comdat any

$_ZTV10polynomialIdE = comdat any

$_ZTI10polynomialIdE = comdat any

$_ZTS10polynomialIdE = comdat any

$_ZTV10polynomialISt7complexIdEE = comdat any

$_ZTI10polynomialISt7complexIdEE = comdat any

$_ZTS10polynomialISt7complexIdEE = comdat any

@.str = private unnamed_addr constant [4 x i8] c"-ga\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.1 = private unnamed_addr constant [32 x i8] c"\0Afftbench (Std. C++) run time: \00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"\0A\0A\00", align 1
@_ZZL13random_doublevE4seed = internal unnamed_addr global i64 1325, align 8
@_ZTV10polynomialIdE = linkonce_odr dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI10polynomialIdE, ptr @_ZN10polynomialIdED2Ev, ptr @_ZN10polynomialIdED0Ev] }, comdat, align 8
@_ZTI10polynomialIdE = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS10polynomialIdE }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS10polynomialIdE = linkonce_odr dso_local constant [16 x i8] c"10polynomialIdE\00", comdat, align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"overflow in fft polynomial stretch\00", align 1
@_ZTISt14overflow_error = external constant ptr
@_ZTV10polynomialISt7complexIdEE = linkonce_odr dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI10polynomialISt7complexIdEE, ptr @_ZN10polynomialISt7complexIdEED2Ev, ptr @_ZN10polynomialISt7complexIdEED0Ev] }, comdat, align 8
@_ZTI10polynomialISt7complexIdEE = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS10polynomialISt7complexIdEE }, comdat, align 8
@_ZTS10polynomialISt7complexIdEE = linkonce_odr dso_local constant [28 x i8] c"10polynomialISt7complexIdEE\00", comdat, align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = alloca %class.polynomial, align 8
  %4 = alloca %class.polynomial, align 8
  %5 = alloca %class.polynomial, align 8
  %6 = icmp sgt i32 %0, 1
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !6
  %10 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %9, ptr noundef nonnull dereferenceable(4) @.str) #13
  %11 = icmp eq i32 %10, 0
  br label %12

12:                                               ; preds = %7, %2
  %13 = phi i1 [ false, %2 ], [ %11, %7 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %3, align 8, !tbaa !11
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i64 524288, ptr %15, align 8, !tbaa !13
  %16 = tail call noalias noundef nonnull dereferenceable(4194304) ptr @_Znam(i64 noundef 4194304) #15
  store ptr %16, ptr %14, align 8, !tbaa !17
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  %17 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %18 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 524288, ptr %18, align 8, !tbaa !13
  %19 = invoke noalias noundef nonnull dereferenceable(4194304) ptr @_Znam(i64 noundef 4194304) #15
          to label %20 unwind label %27

20:                                               ; preds = %12
  store ptr %19, ptr %17, align 8, !tbaa !17
  %21 = invoke noalias noundef nonnull dereferenceable(8388600) ptr @_Znam(i64 noundef 8388600) #15
          to label %22 unwind label %29

22:                                               ; preds = %20
  %23 = load i64, ptr @_ZZL13random_doublevE4seed, align 8, !tbaa !18
  %24 = xor i64 %23, 123459876
  br label %31

25:                                               ; preds = %31
  %26 = xor i64 %54, 123459876
  store i64 %26, ptr @_ZZL13random_doublevE4seed, align 8, !tbaa !18
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  invoke void @_ZNK10polynomialIdEmlERKS0_(ptr dead_on_unwind nonnull writable sret(%class.polynomial) align 8 %5, ptr noundef nonnull align 8 dereferenceable(24) %3, ptr noundef nonnull align 8 dereferenceable(24) %4)
          to label %60 unwind label %120

27:                                               ; preds = %12
  %28 = landingpad { ptr, i32 }
          cleanup
  br label %157

29:                                               ; preds = %20
  %30 = landingpad { ptr, i32 }
          cleanup
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  br label %154

31:                                               ; preds = %22, %31
  %32 = phi i64 [ 0, %22 ], [ %58, %31 ]
  %33 = phi i64 [ %24, %22 ], [ %54, %31 ]
  %34 = sdiv i64 %33, 127773
  %35 = mul nsw i64 %34, -127773
  %36 = add i64 %35, %33
  %37 = mul nsw i64 %36, 16807
  %38 = mul nsw i64 %34, -2836
  %39 = add i64 %37, %38
  %40 = icmp slt i64 %39, 0
  %41 = add nsw i64 %39, 2147483647
  %42 = select i1 %40, i64 %41, i64 %39
  %43 = sitofp i64 %42 to double
  %44 = fmul double %43, 0x3E340000002813D9
  %45 = getelementptr inbounds nuw double, ptr %16, i64 %32
  store double %44, ptr %45, align 8, !tbaa !19
  %46 = sdiv i64 %42, 127773
  %47 = mul nsw i64 %46, -127773
  %48 = add i64 %47, %42
  %49 = mul nsw i64 %48, 16807
  %50 = mul nsw i64 %46, -2836
  %51 = add i64 %49, %50
  %52 = icmp slt i64 %51, 0
  %53 = add nsw i64 %51, 2147483647
  %54 = select i1 %52, i64 %53, i64 %51
  %55 = sitofp i64 %54 to double
  %56 = fmul double %55, 0x3E340000002813D9
  %57 = getelementptr inbounds nuw double, ptr %19, i64 %32
  store double %56, ptr %57, align 8, !tbaa !19
  %58 = add nuw nsw i64 %32, 1
  %59 = icmp eq i64 %58, 524288
  br i1 %59, label %25, label %31, !llvm.loop !21

60:                                               ; preds = %25
  %61 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %62 = load i64, ptr %61, align 8, !tbaa !13
  %63 = icmp eq i64 %62, 1048575
  br i1 %63, label %64, label %67

64:                                               ; preds = %60
  %65 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %66 = load ptr, ptr %65, align 8, !tbaa !17
  br label %77

67:                                               ; preds = %60
  call void @_ZdaPv(ptr noundef nonnull %21) #16
  %68 = load i64, ptr %61, align 8, !tbaa !13
  %69 = icmp ugt i64 %68, 2305843009213693951
  %70 = shl i64 %68, 3
  %71 = select i1 %69, i64 -1, i64 %70
  %72 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %71) #15
          to label %73 unwind label %122

73:                                               ; preds = %67
  %74 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %75 = load ptr, ptr %74, align 8, !tbaa !17
  %76 = icmp eq i64 %68, 0
  br i1 %76, label %111, label %77

77:                                               ; preds = %64, %73
  %78 = phi ptr [ %66, %64 ], [ %75, %73 ]
  %79 = phi i64 [ 1048575, %64 ], [ %68, %73 ]
  %80 = phi ptr [ %21, %64 ], [ %72, %73 ]
  %81 = icmp ult i64 %79, 4
  %82 = ptrtoint ptr %78 to i64
  %83 = ptrtoint ptr %80 to i64
  %84 = sub i64 %83, %82
  %85 = icmp ult i64 %84, 32
  %86 = select i1 %81, i1 true, i1 %85
  br i1 %86, label %101, label %87

87:                                               ; preds = %77
  %88 = and i64 %79, -4
  br label %89

89:                                               ; preds = %89, %87
  %90 = phi i64 [ 0, %87 ], [ %97, %89 ]
  %91 = getelementptr inbounds nuw double, ptr %78, i64 %90
  %92 = getelementptr inbounds nuw i8, ptr %91, i64 16
  %93 = load <2 x double>, ptr %91, align 8, !tbaa !19
  %94 = load <2 x double>, ptr %92, align 8, !tbaa !19
  %95 = getelementptr inbounds nuw double, ptr %80, i64 %90
  %96 = getelementptr inbounds nuw i8, ptr %95, i64 16
  store <2 x double> %93, ptr %95, align 8, !tbaa !19
  store <2 x double> %94, ptr %96, align 8, !tbaa !19
  %97 = add nuw i64 %90, 4
  %98 = icmp eq i64 %97, %88
  br i1 %98, label %99, label %89, !llvm.loop !23

99:                                               ; preds = %89
  %100 = icmp eq i64 %79, %88
  br i1 %100, label %110, label %101

101:                                              ; preds = %77, %99
  %102 = phi i64 [ 0, %77 ], [ %88, %99 ]
  br label %103

103:                                              ; preds = %101, %103
  %104 = phi i64 [ %108, %103 ], [ %102, %101 ]
  %105 = getelementptr inbounds nuw double, ptr %78, i64 %104
  %106 = load double, ptr %105, align 8, !tbaa !19
  %107 = getelementptr inbounds nuw double, ptr %80, i64 %104
  store double %106, ptr %107, align 8, !tbaa !19
  %108 = add nuw i64 %104, 1
  %109 = icmp eq i64 %108, %79
  br i1 %109, label %110, label %103, !llvm.loop !26

110:                                              ; preds = %103, %99
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %5, align 8, !tbaa !11
  br label %113

111:                                              ; preds = %73
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %5, align 8, !tbaa !11
  %112 = icmp eq ptr %75, null
  br i1 %112, label %116, label %113

113:                                              ; preds = %110, %111
  %114 = phi ptr [ %80, %110 ], [ %72, %111 ]
  %115 = phi ptr [ %78, %110 ], [ %75, %111 ]
  call void @_ZdaPv(ptr noundef nonnull %115) #16
  br label %116

116:                                              ; preds = %111, %113
  %117 = phi ptr [ %72, %111 ], [ %114, %113 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br i1 %13, label %118, label %132

118:                                              ; preds = %116
  %119 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef 0.000000e+00)
          to label %138 unwind label %130

120:                                              ; preds = %25
  %121 = landingpad { ptr, i32 }
          cleanup
  br label %128

122:                                              ; preds = %67
  %123 = landingpad { ptr, i32 }
          cleanup
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %5, align 8, !tbaa !11
  %124 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %125 = load ptr, ptr %124, align 8, !tbaa !17
  %126 = icmp eq ptr %125, null
  br i1 %126, label %128, label %127

127:                                              ; preds = %122
  call void @_ZdaPv(ptr noundef nonnull %125) #16
  br label %128

128:                                              ; preds = %127, %122, %120
  %129 = phi { ptr, i32 } [ %121, %120 ], [ %123, %122 ], [ %123, %127 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br label %149

130:                                              ; preds = %136, %134, %132, %118, %138
  %131 = landingpad { ptr, i32 }
          cleanup
  br label %149

132:                                              ; preds = %116
  %133 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.1, i64 noundef 31)
          to label %134 unwind label %130

134:                                              ; preds = %132
  %135 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef 0.000000e+00)
          to label %136 unwind label %130

136:                                              ; preds = %134
  %137 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %135, ptr noundef nonnull @.str.2, i64 noundef 2)
          to label %138 unwind label %130

138:                                              ; preds = %136, %118
  %139 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout)
          to label %140 unwind label %130

140:                                              ; preds = %138
  call void @_ZdaPv(ptr noundef nonnull %117) #16
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  %141 = load ptr, ptr %17, align 8, !tbaa !17
  %142 = icmp eq ptr %141, null
  br i1 %142, label %144, label %143

143:                                              ; preds = %140
  call void @_ZdaPv(ptr noundef nonnull %141) #16
  br label %144

144:                                              ; preds = %140, %143
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %3, align 8, !tbaa !11
  %145 = load ptr, ptr %14, align 8, !tbaa !17
  %146 = icmp eq ptr %145, null
  br i1 %146, label %148, label %147

147:                                              ; preds = %144
  call void @_ZdaPv(ptr noundef nonnull %145) #16
  br label %148

148:                                              ; preds = %144, %147
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  ret i32 0

149:                                              ; preds = %128, %130
  %150 = phi ptr [ %117, %130 ], [ %21, %128 ]
  %151 = phi { ptr, i32 } [ %131, %130 ], [ %129, %128 ]
  call void @_ZdaPv(ptr noundef nonnull %150) #16
  %152 = load ptr, ptr %17, align 8, !tbaa !17
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  %153 = icmp eq ptr %152, null
  br i1 %153, label %157, label %154

154:                                              ; preds = %29, %149
  %155 = phi { ptr, i32 } [ %30, %29 ], [ %151, %149 ]
  %156 = phi ptr [ %19, %29 ], [ %152, %149 ]
  call void @_ZdaPv(ptr noundef nonnull %156) #16
  br label %157

157:                                              ; preds = %154, %149, %27
  %158 = phi { ptr, i32 } [ %28, %27 ], [ %151, %149 ], [ %155, %154 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %3, align 8, !tbaa !11
  %159 = load ptr, ptr %14, align 8, !tbaa !17
  %160 = icmp eq ptr %159, null
  br i1 %160, label %162, label %161

161:                                              ; preds = %157
  call void @_ZdaPv(ptr noundef nonnull %159) #16
  br label %162

162:                                              ; preds = %157, %161
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  resume { ptr, i32 } %158
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNK10polynomialIdEmlERKS0_(ptr dead_on_unwind noalias writable sret(%class.polynomial) align 8 %0, ptr noundef nonnull align 8 dereferenceable(24) %1, ptr noundef nonnull align 8 dereferenceable(24) %2) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %4 = alloca %class.polynomial, align 8
  %5 = alloca %class.polynomial, align 8
  %6 = alloca %class.polynomial.0, align 8
  %7 = alloca %class.polynomial.0, align 8
  %8 = alloca %class.polynomial.0, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %12 = load i64, ptr %11, align 8, !tbaa !13
  store i64 %12, ptr %10, align 8, !tbaa !13
  %13 = icmp ugt i64 %12, 2305843009213693951
  %14 = shl i64 %12, 3
  %15 = select i1 %13, i64 -1, i64 %14
  %16 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %15) #15
  store ptr %16, ptr %9, align 8, !tbaa !17
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %18 = load ptr, ptr %17, align 8, !tbaa !17
  %19 = icmp eq i64 %12, 0
  br i1 %19, label %50, label %20

20:                                               ; preds = %3
  %21 = ptrtoint ptr %18 to i64
  %22 = ptrtoint ptr %16 to i64
  %23 = icmp ult i64 %12, 4
  %24 = sub i64 %22, %21
  %25 = icmp ult i64 %24, 32
  %26 = select i1 %23, i1 true, i1 %25
  br i1 %26, label %41, label %27

27:                                               ; preds = %20
  %28 = and i64 %12, -4
  br label %29

29:                                               ; preds = %29, %27
  %30 = phi i64 [ 0, %27 ], [ %37, %29 ]
  %31 = getelementptr inbounds nuw double, ptr %18, i64 %30
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %33 = load <2 x double>, ptr %31, align 8, !tbaa !19
  %34 = load <2 x double>, ptr %32, align 8, !tbaa !19
  %35 = getelementptr inbounds nuw double, ptr %16, i64 %30
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 16
  store <2 x double> %33, ptr %35, align 8, !tbaa !19
  store <2 x double> %34, ptr %36, align 8, !tbaa !19
  %37 = add nuw i64 %30, 4
  %38 = icmp eq i64 %37, %28
  br i1 %38, label %39, label %29, !llvm.loop !27

39:                                               ; preds = %29
  %40 = icmp eq i64 %12, %28
  br i1 %40, label %50, label %41

41:                                               ; preds = %20, %39
  %42 = phi i64 [ 0, %20 ], [ %28, %39 ]
  br label %43

43:                                               ; preds = %41, %43
  %44 = phi i64 [ %48, %43 ], [ %42, %41 ]
  %45 = getelementptr inbounds nuw double, ptr %18, i64 %44
  %46 = load double, ptr %45, align 8, !tbaa !19
  %47 = getelementptr inbounds nuw double, ptr %16, i64 %44
  store double %46, ptr %47, align 8, !tbaa !19
  %48 = add nuw i64 %44, 1
  %49 = icmp eq i64 %48, %12
  br i1 %49, label %50, label %43, !llvm.loop !28

50:                                               ; preds = %43, %39, %3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %5, align 8, !tbaa !11
  %51 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %52 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %53 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %54 = load i64, ptr %53, align 8, !tbaa !13
  store i64 %54, ptr %52, align 8, !tbaa !13
  %55 = icmp ugt i64 %54, 2305843009213693951
  %56 = shl i64 %54, 3
  %57 = select i1 %55, i64 -1, i64 %56
  %58 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %57) #15
          to label %59 unwind label %141

59:                                               ; preds = %50
  store ptr %58, ptr %51, align 8, !tbaa !17
  %60 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %61 = load ptr, ptr %60, align 8, !tbaa !17
  %62 = icmp eq i64 %54, 0
  br i1 %62, label %93, label %63

63:                                               ; preds = %59
  %64 = ptrtoint ptr %61 to i64
  %65 = ptrtoint ptr %58 to i64
  %66 = icmp ult i64 %54, 4
  %67 = sub i64 %65, %64
  %68 = icmp ult i64 %67, 32
  %69 = select i1 %66, i1 true, i1 %68
  br i1 %69, label %84, label %70

70:                                               ; preds = %63
  %71 = and i64 %54, -4
  br label %72

72:                                               ; preds = %72, %70
  %73 = phi i64 [ 0, %70 ], [ %80, %72 ]
  %74 = getelementptr inbounds nuw double, ptr %61, i64 %73
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 16
  %76 = load <2 x double>, ptr %74, align 8, !tbaa !19
  %77 = load <2 x double>, ptr %75, align 8, !tbaa !19
  %78 = getelementptr inbounds nuw double, ptr %58, i64 %73
  %79 = getelementptr inbounds nuw i8, ptr %78, i64 16
  store <2 x double> %76, ptr %78, align 8, !tbaa !19
  store <2 x double> %77, ptr %79, align 8, !tbaa !19
  %80 = add nuw i64 %73, 4
  %81 = icmp eq i64 %80, %71
  br i1 %81, label %82, label %72, !llvm.loop !29

82:                                               ; preds = %72
  %83 = icmp eq i64 %54, %71
  br i1 %83, label %93, label %84

84:                                               ; preds = %63, %82
  %85 = phi i64 [ 0, %63 ], [ %71, %82 ]
  br label %86

86:                                               ; preds = %84, %86
  %87 = phi i64 [ %91, %86 ], [ %85, %84 ]
  %88 = getelementptr inbounds nuw double, ptr %61, i64 %87
  %89 = load double, ptr %88, align 8, !tbaa !19
  %90 = getelementptr inbounds nuw double, ptr %58, i64 %87
  store double %89, ptr %90, align 8, !tbaa !19
  %91 = add nuw i64 %87, 1
  %92 = icmp eq i64 %91, %54
  br i1 %92, label %93, label %86, !llvm.loop !30

93:                                               ; preds = %86, %82, %59
  %94 = icmp ugt i64 %12, %54
  br i1 %94, label %95, label %145

95:                                               ; preds = %93
  %96 = invoke noundef i64 @_ZN10polynomialIdE11stretch_fftEv(ptr noundef nonnull align 8 dereferenceable(24) %4)
          to label %97 unwind label %143

97:                                               ; preds = %95
  %98 = icmp eq i64 %96, 0
  br i1 %98, label %198, label %99

99:                                               ; preds = %97
  %100 = load ptr, ptr %51, align 8, !tbaa !17
  %101 = ptrtoint ptr %100 to i64
  %102 = load i64, ptr %52, align 8, !tbaa !13
  %103 = add i64 %102, %96
  store i64 %103, ptr %52, align 8, !tbaa !13
  %104 = icmp ugt i64 %103, 2305843009213693951
  %105 = shl i64 %103, 3
  %106 = select i1 %104, i64 -1, i64 %105
  %107 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %106) #15
          to label %108 unwind label %143

108:                                              ; preds = %99
  store ptr %107, ptr %51, align 8, !tbaa !17
  %109 = icmp eq i64 %102, 0
  br i1 %109, label %132, label %110

110:                                              ; preds = %108
  %111 = ptrtoint ptr %107 to i64
  %112 = icmp ult i64 %102, 4
  %113 = sub i64 %111, %101
  %114 = icmp ult i64 %113, 32
  %115 = select i1 %112, i1 true, i1 %114
  br i1 %115, label %130, label %116

116:                                              ; preds = %110
  %117 = and i64 %102, -4
  br label %118

118:                                              ; preds = %118, %116
  %119 = phi i64 [ 0, %116 ], [ %126, %118 ]
  %120 = getelementptr inbounds nuw double, ptr %100, i64 %119
  %121 = getelementptr inbounds nuw i8, ptr %120, i64 16
  %122 = load <2 x double>, ptr %120, align 8, !tbaa !19
  %123 = load <2 x double>, ptr %121, align 8, !tbaa !19
  %124 = getelementptr inbounds nuw double, ptr %107, i64 %119
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 16
  store <2 x double> %122, ptr %124, align 8, !tbaa !19
  store <2 x double> %123, ptr %125, align 8, !tbaa !19
  %126 = add nuw i64 %119, 4
  %127 = icmp eq i64 %126, %117
  br i1 %127, label %128, label %118, !llvm.loop !31

128:                                              ; preds = %118
  %129 = icmp eq i64 %102, %117
  br i1 %129, label %132, label %130

130:                                              ; preds = %110, %128
  %131 = phi i64 [ 0, %110 ], [ %117, %128 ]
  br label %134

132:                                              ; preds = %134, %128, %108
  %133 = icmp ult i64 %102, %103
  br i1 %133, label %191, label %198

134:                                              ; preds = %130, %134
  %135 = phi i64 [ %139, %134 ], [ %131, %130 ]
  %136 = getelementptr inbounds nuw double, ptr %100, i64 %135
  %137 = load double, ptr %136, align 8, !tbaa !19
  %138 = getelementptr inbounds nuw double, ptr %107, i64 %135
  store double %137, ptr %138, align 8, !tbaa !19
  %139 = add nuw i64 %135, 1
  %140 = icmp eq i64 %139, %102
  br i1 %140, label %132, label %134, !llvm.loop !32

141:                                              ; preds = %50
  %142 = landingpad { ptr, i32 }
          cleanup
  br label %384

143:                                              ; preds = %149, %99, %145, %95
  %144 = landingpad { ptr, i32 }
          cleanup
  br label %379

145:                                              ; preds = %93
  %146 = invoke noundef i64 @_ZN10polynomialIdE11stretch_fftEv(ptr noundef nonnull align 8 dereferenceable(24) %5)
          to label %147 unwind label %143

147:                                              ; preds = %145
  %148 = icmp eq i64 %146, 0
  br i1 %148, label %198, label %149

149:                                              ; preds = %147
  %150 = load ptr, ptr %9, align 8, !tbaa !17
  %151 = ptrtoint ptr %150 to i64
  %152 = load i64, ptr %10, align 8, !tbaa !13
  %153 = add i64 %152, %146
  store i64 %153, ptr %10, align 8, !tbaa !13
  %154 = icmp ugt i64 %153, 2305843009213693951
  %155 = shl i64 %153, 3
  %156 = select i1 %154, i64 -1, i64 %155
  %157 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %156) #15
          to label %158 unwind label %143

158:                                              ; preds = %149
  store ptr %157, ptr %9, align 8, !tbaa !17
  %159 = icmp eq i64 %152, 0
  br i1 %159, label %182, label %160

160:                                              ; preds = %158
  %161 = ptrtoint ptr %157 to i64
  %162 = icmp ult i64 %152, 4
  %163 = sub i64 %161, %151
  %164 = icmp ult i64 %163, 32
  %165 = select i1 %162, i1 true, i1 %164
  br i1 %165, label %180, label %166

166:                                              ; preds = %160
  %167 = and i64 %152, -4
  br label %168

168:                                              ; preds = %168, %166
  %169 = phi i64 [ 0, %166 ], [ %176, %168 ]
  %170 = getelementptr inbounds nuw double, ptr %150, i64 %169
  %171 = getelementptr inbounds nuw i8, ptr %170, i64 16
  %172 = load <2 x double>, ptr %170, align 8, !tbaa !19
  %173 = load <2 x double>, ptr %171, align 8, !tbaa !19
  %174 = getelementptr inbounds nuw double, ptr %157, i64 %169
  %175 = getelementptr inbounds nuw i8, ptr %174, i64 16
  store <2 x double> %172, ptr %174, align 8, !tbaa !19
  store <2 x double> %173, ptr %175, align 8, !tbaa !19
  %176 = add nuw i64 %169, 4
  %177 = icmp eq i64 %176, %167
  br i1 %177, label %178, label %168, !llvm.loop !33

178:                                              ; preds = %168
  %179 = icmp eq i64 %152, %167
  br i1 %179, label %182, label %180

180:                                              ; preds = %160, %178
  %181 = phi i64 [ 0, %160 ], [ %167, %178 ]
  br label %184

182:                                              ; preds = %184, %178, %158
  %183 = icmp ult i64 %152, %153
  br i1 %183, label %191, label %198

184:                                              ; preds = %180, %184
  %185 = phi i64 [ %189, %184 ], [ %181, %180 ]
  %186 = getelementptr inbounds nuw double, ptr %150, i64 %185
  %187 = load double, ptr %186, align 8, !tbaa !19
  %188 = getelementptr inbounds nuw double, ptr %157, i64 %185
  store double %187, ptr %188, align 8, !tbaa !19
  %189 = add nuw i64 %185, 1
  %190 = icmp eq i64 %189, %152
  br i1 %190, label %182, label %184, !llvm.loop !34

191:                                              ; preds = %182, %132
  %192 = phi i64 [ %102, %132 ], [ %152, %182 ]
  %193 = phi ptr [ %107, %132 ], [ %157, %182 ]
  %194 = phi i64 [ %96, %132 ], [ %146, %182 ]
  %195 = shl i64 %192, 3
  %196 = getelementptr i8, ptr %193, i64 %195
  %197 = shl nuw i64 %194, 3
  call void @llvm.memset.p0.i64(ptr align 8 %196, i8 0, i64 %197, i1 false), !tbaa !19
  br label %198

198:                                              ; preds = %191, %182, %147, %132, %97
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #14
  invoke void @_ZN10polynomialIdE3fftERKS0_(ptr dead_on_unwind nonnull writable sret(%class.polynomial.0) align 8 %6, ptr noundef nonnull align 8 dereferenceable(24) %4)
          to label %199 unwind label %207

199:                                              ; preds = %198
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #14
  invoke void @_ZN10polynomialIdE3fftERKS0_(ptr dead_on_unwind nonnull writable sret(%class.polynomial.0) align 8 %7, ptr noundef nonnull align 8 dereferenceable(24) %5)
          to label %200 unwind label %209

200:                                              ; preds = %199
  %201 = load i64, ptr %10, align 8, !tbaa !13
  %202 = icmp eq i64 %201, 0
  br i1 %202, label %206, label %203

203:                                              ; preds = %200
  %204 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %205 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br label %213

206:                                              ; preds = %238, %200
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #14
  invoke void @_ZN10polynomialIdE11inverse_fftERKS_ISt7complexIdEE(ptr dead_on_unwind nonnull writable sret(%class.polynomial.0) align 8 %8, ptr noundef nonnull align 8 dereferenceable(24) %6)
          to label %243 unwind label %330

207:                                              ; preds = %198
  %208 = landingpad { ptr, i32 }
          cleanup
  br label %377

209:                                              ; preds = %199
  %210 = landingpad { ptr, i32 }
          cleanup
  br label %371

211:                                              ; preds = %286
  %212 = landingpad { ptr, i32 }
          cleanup
  br label %365

213:                                              ; preds = %203, %238
  %214 = phi i64 [ 0, %203 ], [ %241, %238 ]
  %215 = load ptr, ptr %204, align 8, !tbaa !35
  %216 = getelementptr inbounds nuw %"class.std::complex", ptr %215, i64 %214
  %217 = load ptr, ptr %205, align 8, !tbaa !35
  %218 = getelementptr inbounds nuw %"class.std::complex", ptr %217, i64 %214
  %219 = load double, ptr %216, align 8
  %220 = getelementptr inbounds nuw i8, ptr %216, i64 8
  %221 = load double, ptr %220, align 8
  %222 = load double, ptr %218, align 8
  %223 = getelementptr inbounds nuw i8, ptr %218, i64 8
  %224 = load double, ptr %223, align 8
  %225 = fmul double %219, %222
  %226 = fmul double %221, %224
  %227 = fmul double %221, %222
  %228 = fmul double %219, %224
  %229 = fsub double %225, %226
  %230 = fadd double %227, %228
  %231 = fcmp uno double %229, 0.000000e+00
  br i1 %231, label %232, label %238, !prof !38

232:                                              ; preds = %213
  %233 = fcmp uno double %230, 0.000000e+00
  br i1 %233, label %234, label %238, !prof !38

234:                                              ; preds = %232
  %235 = call noundef { double, double } @__muldc3(double noundef %222, double noundef %224, double noundef %219, double noundef %221) #14
  %236 = extractvalue { double, double } %235, 0
  %237 = extractvalue { double, double } %235, 1
  br label %238

238:                                              ; preds = %213, %232, %234
  %239 = phi double [ %229, %213 ], [ %229, %232 ], [ %236, %234 ]
  %240 = phi double [ %230, %213 ], [ %230, %232 ], [ %237, %234 ]
  store double %239, ptr %218, align 8
  store double %240, ptr %223, align 8
  %241 = add nuw i64 %214, 1
  %242 = icmp eq i64 %241, %201
  br i1 %242, label %206, label %213, !llvm.loop !39

243:                                              ; preds = %206
  %244 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %245 = load i64, ptr %244, align 8, !tbaa !40
  %246 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %247 = load i64, ptr %246, align 8, !tbaa !40
  %248 = icmp eq i64 %245, %247
  br i1 %248, label %265, label %249

249:                                              ; preds = %243
  %250 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %251 = load ptr, ptr %250, align 8, !tbaa !35
  %252 = icmp eq ptr %251, null
  br i1 %252, label %255, label %253

253:                                              ; preds = %249
  call void @_ZdaPv(ptr noundef nonnull %251) #16
  %254 = load i64, ptr %246, align 8, !tbaa !40
  br label %255

255:                                              ; preds = %253, %249
  %256 = phi i64 [ %247, %249 ], [ %254, %253 ]
  store i64 %256, ptr %244, align 8, !tbaa !40
  %257 = icmp ugt i64 %256, 1152921504606846975
  %258 = shl i64 %256, 4
  %259 = select i1 %257, i64 -1, i64 %258
  %260 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %259) #15
          to label %261 unwind label %332

261:                                              ; preds = %255
  %262 = icmp eq i64 %256, 0
  br i1 %262, label %264, label %263

263:                                              ; preds = %261
  call void @llvm.memset.p0.i64(ptr nonnull align 8 %260, i8 0, i64 %258, i1 false)
  br label %264

264:                                              ; preds = %263, %261
  store ptr %260, ptr %250, align 8, !tbaa !35
  br label %265

265:                                              ; preds = %264, %243
  %266 = phi i64 [ %256, %264 ], [ %245, %243 ]
  %267 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %268 = load ptr, ptr %267, align 8, !tbaa !35
  %269 = icmp eq i64 %266, 0
  br i1 %269, label %282, label %270

270:                                              ; preds = %265
  %271 = getelementptr inbounds nuw i8, ptr %7, i64 8
  br label %272

272:                                              ; preds = %272, %270
  %273 = phi i64 [ 0, %270 ], [ %277, %272 ]
  %274 = getelementptr inbounds nuw %"class.std::complex", ptr %268, i64 %273
  %275 = load ptr, ptr %271, align 8, !tbaa !35
  %276 = getelementptr inbounds nuw %"class.std::complex", ptr %275, i64 %273
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %276, ptr noundef nonnull align 8 dereferenceable(16) %274, i64 16, i1 false), !tbaa.struct !41
  %277 = add nuw i64 %273, 1
  %278 = load i64, ptr %244, align 8, !tbaa !40
  %279 = icmp ult i64 %277, %278
  br i1 %279, label %272, label %280, !llvm.loop !43

280:                                              ; preds = %272
  %281 = load ptr, ptr %267, align 8, !tbaa !35
  br label %282

282:                                              ; preds = %280, %265
  %283 = phi ptr [ %281, %280 ], [ %268, %265 ]
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %8, align 8, !tbaa !11
  %284 = icmp eq ptr %283, null
  br i1 %284, label %286, label %285

285:                                              ; preds = %282
  call void @_ZdaPv(ptr noundef nonnull %283) #16
  br label %286

286:                                              ; preds = %282, %285
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #14
  %287 = add i64 %201, -1
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %0, align 8, !tbaa !11
  %288 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr null, ptr %288, align 8, !tbaa !17
  %289 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %287, ptr %289, align 8, !tbaa !13
  %290 = icmp ugt i64 %287, 2305843009213693951
  %291 = shl nuw i64 %287, 3
  %292 = select i1 %290, i64 -1, i64 %291
  %293 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %292) #15
          to label %294 unwind label %211

294:                                              ; preds = %286
  store ptr %293, ptr %288, align 8, !tbaa !17
  %295 = icmp eq i64 %287, 0
  %296 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %297 = load ptr, ptr %296, align 8, !tbaa !35
  br i1 %295, label %348, label %298

298:                                              ; preds = %294
  %299 = icmp ult i64 %287, 13
  br i1 %299, label %300, label %302

300:                                              ; preds = %317, %302, %298
  %301 = phi i64 [ 0, %302 ], [ 0, %298 ], [ %316, %317 ]
  br label %340

302:                                              ; preds = %298
  %303 = shl i64 %201, 3
  %304 = getelementptr i8, ptr %293, i64 %303
  %305 = getelementptr i8, ptr %304, i64 -8
  %306 = shl i64 %201, 4
  %307 = getelementptr i8, ptr %297, i64 %306
  %308 = getelementptr i8, ptr %307, i64 -24
  %309 = icmp ult ptr %293, %308
  %310 = icmp ult ptr %297, %305
  %311 = and i1 %309, %310
  br i1 %311, label %300, label %312

312:                                              ; preds = %302
  %313 = and i64 %287, 3
  %314 = icmp eq i64 %313, 0
  %315 = select i1 %314, i64 4, i64 %313
  %316 = sub i64 %287, %315
  br label %317

317:                                              ; preds = %317, %312
  %318 = phi i64 [ 0, %312 ], [ %328, %317 ]
  %319 = getelementptr inbounds nuw %"class.std::complex", ptr %297, i64 %318
  %320 = getelementptr inbounds nuw %"class.std::complex", ptr %297, i64 %318
  %321 = getelementptr inbounds nuw i8, ptr %320, i64 32
  %322 = load <3 x double>, ptr %319, align 8, !tbaa !19, !alias.scope !44
  %323 = shufflevector <3 x double> %322, <3 x double> poison, <2 x i32> <i32 0, i32 2>
  %324 = load <3 x double>, ptr %321, align 8, !tbaa !19, !alias.scope !44
  %325 = shufflevector <3 x double> %324, <3 x double> poison, <2 x i32> <i32 0, i32 2>
  %326 = getelementptr inbounds nuw double, ptr %293, i64 %318
  %327 = getelementptr inbounds nuw i8, ptr %326, i64 16
  store <2 x double> %323, ptr %326, align 8, !tbaa !19, !alias.scope !47, !noalias !44
  store <2 x double> %325, ptr %327, align 8, !tbaa !19, !alias.scope !47, !noalias !44
  %328 = add nuw i64 %318, 4
  %329 = icmp eq i64 %328, %316
  br i1 %329, label %300, label %317, !llvm.loop !49

330:                                              ; preds = %206
  %331 = landingpad { ptr, i32 }
          cleanup
  br label %338

332:                                              ; preds = %255
  %333 = landingpad { ptr, i32 }
          cleanup
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %8, align 8, !tbaa !11
  %334 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %335 = load ptr, ptr %334, align 8, !tbaa !35
  %336 = icmp eq ptr %335, null
  br i1 %336, label %338, label %337

337:                                              ; preds = %332
  call void @_ZdaPv(ptr noundef nonnull %335) #16
  br label %338

338:                                              ; preds = %337, %332, %330
  %339 = phi { ptr, i32 } [ %331, %330 ], [ %333, %332 ], [ %333, %337 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #14
  br label %365

340:                                              ; preds = %300, %340
  %341 = phi i64 [ %345, %340 ], [ %301, %300 ]
  %342 = getelementptr inbounds nuw %"class.std::complex", ptr %297, i64 %341
  %343 = load double, ptr %342, align 8, !tbaa !19
  %344 = getelementptr inbounds nuw double, ptr %293, i64 %341
  store double %343, ptr %344, align 8, !tbaa !19
  %345 = add nuw i64 %341, 1
  %346 = icmp eq i64 %345, %287
  br i1 %346, label %347, label %340, !llvm.loop !50

347:                                              ; preds = %340
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %7, align 8, !tbaa !11
  br label %350

348:                                              ; preds = %294
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %7, align 8, !tbaa !11
  %349 = icmp eq ptr %297, null
  br i1 %349, label %351, label %350

350:                                              ; preds = %347, %348
  call void @_ZdaPv(ptr noundef nonnull %297) #16
  br label %351

351:                                              ; preds = %348, %350
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %6, align 8, !tbaa !11
  %352 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %353 = load ptr, ptr %352, align 8, !tbaa !35
  %354 = icmp eq ptr %353, null
  br i1 %354, label %356, label %355

355:                                              ; preds = %351
  call void @_ZdaPv(ptr noundef nonnull %353) #16
  br label %356

356:                                              ; preds = %351, %355
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %5, align 8, !tbaa !11
  %357 = load ptr, ptr %51, align 8, !tbaa !17
  %358 = icmp eq ptr %357, null
  br i1 %358, label %360, label %359

359:                                              ; preds = %356
  call void @_ZdaPv(ptr noundef nonnull %357) #16
  br label %360

360:                                              ; preds = %356, %359
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  %361 = load ptr, ptr %9, align 8, !tbaa !17
  %362 = icmp eq ptr %361, null
  br i1 %362, label %364, label %363

363:                                              ; preds = %360
  call void @_ZdaPv(ptr noundef nonnull %361) #16
  br label %364

364:                                              ; preds = %360, %363
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  ret void

365:                                              ; preds = %338, %211
  %366 = phi { ptr, i32 } [ %212, %211 ], [ %339, %338 ]
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %7, align 8, !tbaa !11
  %367 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %368 = load ptr, ptr %367, align 8, !tbaa !35
  %369 = icmp eq ptr %368, null
  br i1 %369, label %371, label %370

370:                                              ; preds = %365
  call void @_ZdaPv(ptr noundef nonnull %368) #16
  br label %371

371:                                              ; preds = %370, %365, %209
  %372 = phi { ptr, i32 } [ %210, %209 ], [ %366, %365 ], [ %366, %370 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %6, align 8, !tbaa !11
  %373 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %374 = load ptr, ptr %373, align 8, !tbaa !35
  %375 = icmp eq ptr %374, null
  br i1 %375, label %377, label %376

376:                                              ; preds = %371
  call void @_ZdaPv(ptr noundef nonnull %374) #16
  br label %377

377:                                              ; preds = %376, %371, %207
  %378 = phi { ptr, i32 } [ %208, %207 ], [ %372, %371 ], [ %372, %376 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #14
  br label %379

379:                                              ; preds = %377, %143
  %380 = phi { ptr, i32 } [ %378, %377 ], [ %144, %143 ]
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %5, align 8, !tbaa !11
  %381 = load ptr, ptr %51, align 8, !tbaa !17
  %382 = icmp eq ptr %381, null
  br i1 %382, label %384, label %383

383:                                              ; preds = %379
  call void @_ZdaPv(ptr noundef nonnull %381) #16
  br label %384

384:                                              ; preds = %383, %379, %141
  %385 = phi { ptr, i32 } [ %142, %141 ], [ %380, %379 ], [ %380, %383 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %4, align 8, !tbaa !11
  %386 = load ptr, ptr %9, align 8, !tbaa !17
  %387 = icmp eq ptr %386, null
  br i1 %387, label %389, label %388

388:                                              ; preds = %384
  call void @_ZdaPv(ptr noundef nonnull %386) #16
  br label %389

389:                                              ; preds = %384, %388
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  resume { ptr, i32 } %385
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN10polynomialIdED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #4 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %0, align 8, !tbaa !11
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !tbaa !17
  %4 = icmp eq ptr %3, null
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @_ZdaPv(ptr noundef nonnull %3) #16
  br label %6

6:                                                ; preds = %1, %5
  ret void
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #5

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN10polynomialIdED0Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #4 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialIdE, i64 16), ptr %0, align 8, !tbaa !11
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !tbaa !17
  %4 = icmp eq ptr %3, null
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @_ZdaPv(ptr noundef nonnull %3) #16
  br label %6

6:                                                ; preds = %1, %5
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #16
  ret void
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #6

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #7

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #7

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @_ZN10polynomialIdE11stretch_fftEv(ptr noundef nonnull align 8 dereferenceable(24) %0) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load i64, ptr %2, align 8, !tbaa !13
  %4 = icmp ugt i64 %3, 1
  br i1 %4, label %5, label %136

5:                                                ; preds = %1
  %6 = icmp eq i64 %3, 2
  br i1 %6, label %136, label %7

7:                                                ; preds = %5
  %8 = icmp ugt i64 %3, 4
  br i1 %8, label %9, label %136

9:                                                ; preds = %7
  %10 = icmp ugt i64 %3, 8
  br i1 %10, label %11, label %136

11:                                               ; preds = %9
  %12 = icmp ugt i64 %3, 16
  br i1 %12, label %13, label %136

13:                                               ; preds = %11
  %14 = icmp ugt i64 %3, 32
  br i1 %14, label %15, label %136

15:                                               ; preds = %13
  %16 = icmp ugt i64 %3, 64
  br i1 %16, label %17, label %136

17:                                               ; preds = %15
  %18 = icmp ugt i64 %3, 128
  br i1 %18, label %19, label %136

19:                                               ; preds = %17
  %20 = icmp ugt i64 %3, 256
  br i1 %20, label %21, label %136

21:                                               ; preds = %19
  %22 = icmp ugt i64 %3, 512
  br i1 %22, label %23, label %136

23:                                               ; preds = %21
  %24 = icmp ugt i64 %3, 1024
  br i1 %24, label %25, label %136

25:                                               ; preds = %23
  %26 = icmp ugt i64 %3, 2048
  br i1 %26, label %27, label %136

27:                                               ; preds = %25
  %28 = icmp ugt i64 %3, 4096
  br i1 %28, label %29, label %136

29:                                               ; preds = %27
  %30 = icmp ugt i64 %3, 8192
  br i1 %30, label %31, label %136

31:                                               ; preds = %29
  %32 = icmp ugt i64 %3, 16384
  br i1 %32, label %33, label %136

33:                                               ; preds = %31
  %34 = icmp ugt i64 %3, 32768
  br i1 %34, label %35, label %136

35:                                               ; preds = %33
  %36 = icmp ugt i64 %3, 65536
  br i1 %36, label %37, label %136

37:                                               ; preds = %35
  %38 = icmp ugt i64 %3, 131072
  br i1 %38, label %39, label %136

39:                                               ; preds = %37
  %40 = icmp ugt i64 %3, 262144
  br i1 %40, label %41, label %136

41:                                               ; preds = %39
  %42 = icmp ugt i64 %3, 524288
  br i1 %42, label %43, label %136

43:                                               ; preds = %41
  %44 = icmp ugt i64 %3, 1048576
  br i1 %44, label %45, label %136

45:                                               ; preds = %43
  %46 = icmp ugt i64 %3, 2097152
  br i1 %46, label %47, label %136

47:                                               ; preds = %45
  %48 = icmp ugt i64 %3, 4194304
  br i1 %48, label %49, label %136

49:                                               ; preds = %47
  %50 = icmp ugt i64 %3, 8388608
  br i1 %50, label %51, label %136

51:                                               ; preds = %49
  %52 = icmp ugt i64 %3, 16777216
  br i1 %52, label %53, label %136

53:                                               ; preds = %51
  %54 = icmp ugt i64 %3, 33554432
  br i1 %54, label %55, label %136

55:                                               ; preds = %53
  %56 = icmp ugt i64 %3, 67108864
  br i1 %56, label %57, label %136

57:                                               ; preds = %55
  %58 = icmp ugt i64 %3, 134217728
  br i1 %58, label %59, label %136

59:                                               ; preds = %57
  %60 = icmp ugt i64 %3, 268435456
  br i1 %60, label %61, label %136

61:                                               ; preds = %59
  %62 = icmp ugt i64 %3, 536870912
  br i1 %62, label %63, label %136

63:                                               ; preds = %61
  %64 = icmp ugt i64 %3, 1073741824
  br i1 %64, label %65, label %136

65:                                               ; preds = %63
  %66 = icmp ugt i64 %3, 2147483648
  br i1 %66, label %67, label %136

67:                                               ; preds = %65
  %68 = icmp ugt i64 %3, 4294967296
  br i1 %68, label %69, label %136

69:                                               ; preds = %67
  %70 = icmp ugt i64 %3, 8589934592
  br i1 %70, label %71, label %136

71:                                               ; preds = %69
  %72 = icmp ugt i64 %3, 17179869184
  br i1 %72, label %73, label %136

73:                                               ; preds = %71
  %74 = icmp ugt i64 %3, 34359738368
  br i1 %74, label %75, label %136

75:                                               ; preds = %73
  %76 = icmp ugt i64 %3, 68719476736
  br i1 %76, label %77, label %136

77:                                               ; preds = %75
  %78 = icmp ugt i64 %3, 137438953472
  br i1 %78, label %79, label %136

79:                                               ; preds = %77
  %80 = icmp ugt i64 %3, 274877906944
  br i1 %80, label %81, label %136

81:                                               ; preds = %79
  %82 = icmp ugt i64 %3, 549755813888
  br i1 %82, label %83, label %136

83:                                               ; preds = %81
  %84 = icmp ugt i64 %3, 1099511627776
  br i1 %84, label %85, label %136

85:                                               ; preds = %83
  %86 = icmp ugt i64 %3, 2199023255552
  br i1 %86, label %87, label %136

87:                                               ; preds = %85
  %88 = icmp ugt i64 %3, 4398046511104
  br i1 %88, label %89, label %136

89:                                               ; preds = %87
  %90 = icmp ugt i64 %3, 8796093022208
  br i1 %90, label %91, label %136

91:                                               ; preds = %89
  %92 = icmp ugt i64 %3, 17592186044416
  br i1 %92, label %93, label %136

93:                                               ; preds = %91
  %94 = icmp ugt i64 %3, 35184372088832
  br i1 %94, label %95, label %136

95:                                               ; preds = %93
  %96 = icmp ugt i64 %3, 70368744177664
  br i1 %96, label %97, label %136

97:                                               ; preds = %95
  %98 = icmp ugt i64 %3, 140737488355328
  br i1 %98, label %99, label %136

99:                                               ; preds = %97
  %100 = icmp ugt i64 %3, 281474976710656
  br i1 %100, label %101, label %136

101:                                              ; preds = %99
  %102 = icmp ugt i64 %3, 562949953421312
  br i1 %102, label %103, label %136

103:                                              ; preds = %101
  %104 = icmp ugt i64 %3, 1125899906842624
  br i1 %104, label %105, label %136

105:                                              ; preds = %103
  %106 = icmp ugt i64 %3, 2251799813685248
  br i1 %106, label %107, label %136

107:                                              ; preds = %105
  %108 = icmp ugt i64 %3, 4503599627370496
  br i1 %108, label %109, label %136

109:                                              ; preds = %107
  %110 = icmp ugt i64 %3, 9007199254740992
  br i1 %110, label %111, label %136

111:                                              ; preds = %109
  %112 = icmp ugt i64 %3, 18014398509481984
  br i1 %112, label %113, label %136

113:                                              ; preds = %111
  %114 = icmp ugt i64 %3, 36028797018963968
  br i1 %114, label %115, label %136

115:                                              ; preds = %113
  %116 = icmp ugt i64 %3, 72057594037927936
  br i1 %116, label %117, label %136

117:                                              ; preds = %115
  %118 = icmp ugt i64 %3, 144115188075855872
  br i1 %118, label %119, label %136

119:                                              ; preds = %117
  %120 = icmp ugt i64 %3, 288230376151711744
  br i1 %120, label %121, label %136

121:                                              ; preds = %119
  %122 = icmp ugt i64 %3, 576460752303423488
  br i1 %122, label %123, label %136

123:                                              ; preds = %121
  %124 = icmp ugt i64 %3, 1152921504606846976
  br i1 %124, label %125, label %136

125:                                              ; preds = %123
  %126 = icmp ugt i64 %3, 2305843009213693952
  br i1 %126, label %127, label %136

127:                                              ; preds = %125
  %128 = icmp ugt i64 %3, 4611686018427387904
  br i1 %128, label %129, label %136

129:                                              ; preds = %127
  %130 = icmp ugt i64 %3, -9223372036854775808
  br i1 %130, label %131, label %136

131:                                              ; preds = %129
  %132 = tail call ptr @__cxa_allocate_exception(i64 16) #14
  invoke void @_ZNSt14overflow_errorC1EPKc(ptr noundef nonnull align 8 dereferenceable(16) %132, ptr noundef nonnull @.str.3)
          to label %133 unwind label %134

133:                                              ; preds = %131
  tail call void @__cxa_throw(ptr nonnull %132, ptr nonnull @_ZTISt14overflow_error, ptr nonnull @_ZNSt14overflow_errorD1Ev) #17
  unreachable

134:                                              ; preds = %131
  %135 = landingpad { ptr, i32 }
          cleanup
  tail call void @__cxa_free_exception(ptr nonnull %132) #14
  resume { ptr, i32 } %135

136:                                              ; preds = %129, %127, %125, %123, %121, %119, %117, %115, %113, %111, %109, %107, %105, %103, %101, %99, %97, %95, %93, %91, %89, %87, %85, %83, %81, %79, %77, %75, %73, %71, %69, %67, %65, %63, %61, %59, %57, %55, %53, %51, %49, %47, %45, %43, %41, %39, %37, %35, %33, %31, %29, %27, %25, %23, %21, %19, %17, %15, %13, %11, %9, %7, %5, %1
  %137 = phi i64 [ 1, %1 ], [ 2, %5 ], [ 4, %7 ], [ 8, %9 ], [ 16, %11 ], [ 32, %13 ], [ 64, %15 ], [ 128, %17 ], [ 256, %19 ], [ 512, %21 ], [ 1024, %23 ], [ 2048, %25 ], [ 4096, %27 ], [ 8192, %29 ], [ 16384, %31 ], [ 32768, %33 ], [ 65536, %35 ], [ 131072, %37 ], [ 262144, %39 ], [ 524288, %41 ], [ 1048576, %43 ], [ 2097152, %45 ], [ 4194304, %47 ], [ 8388608, %49 ], [ 16777216, %51 ], [ 33554432, %53 ], [ 67108864, %55 ], [ 134217728, %57 ], [ 268435456, %59 ], [ 536870912, %61 ], [ 1073741824, %63 ], [ 2147483648, %65 ], [ 4294967296, %67 ], [ 8589934592, %69 ], [ 17179869184, %71 ], [ 34359738368, %73 ], [ 68719476736, %75 ], [ 137438953472, %77 ], [ 274877906944, %79 ], [ 549755813888, %81 ], [ 1099511627776, %83 ], [ 2199023255552, %85 ], [ 4398046511104, %87 ], [ 8796093022208, %89 ], [ 17592186044416, %91 ], [ 35184372088832, %93 ], [ 70368744177664, %95 ], [ 140737488355328, %97 ], [ 281474976710656, %99 ], [ 562949953421312, %101 ], [ 1125899906842624, %103 ], [ 2251799813685248, %105 ], [ 4503599627370496, %107 ], [ 9007199254740992, %109 ], [ 18014398509481984, %111 ], [ 36028797018963968, %113 ], [ 72057594037927936, %115 ], [ 144115188075855872, %117 ], [ 288230376151711744, %119 ], [ 576460752303423488, %121 ], [ 1152921504606846976, %123 ], [ 2305843009213693952, %125 ], [ 4611686018427387904, %127 ], [ -9223372036854775808, %129 ]
  %138 = shl i64 %137, 1
  %139 = sub i64 %138, %3
  %140 = icmp eq i64 %138, %3
  br i1 %140, label %185, label %141

141:                                              ; preds = %136
  %142 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %143 = load ptr, ptr %142, align 8, !tbaa !17
  store i64 %138, ptr %2, align 8, !tbaa !13
  %144 = icmp ugt i64 %138, 2305843009213693951
  %145 = shl i64 %137, 4
  %146 = select i1 %144, i64 -1, i64 %145
  %147 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %146) #15
  store ptr %147, ptr %142, align 8, !tbaa !17
  %148 = icmp eq i64 %3, 0
  br i1 %148, label %172, label %149

149:                                              ; preds = %141
  %150 = ptrtoint ptr %147 to i64
  %151 = ptrtoint ptr %143 to i64
  %152 = icmp ult i64 %3, 4
  %153 = sub i64 %150, %151
  %154 = icmp ult i64 %153, 32
  %155 = select i1 %152, i1 true, i1 %154
  br i1 %155, label %170, label %156

156:                                              ; preds = %149
  %157 = and i64 %3, -4
  br label %158

158:                                              ; preds = %158, %156
  %159 = phi i64 [ 0, %156 ], [ %166, %158 ]
  %160 = getelementptr inbounds nuw double, ptr %143, i64 %159
  %161 = getelementptr inbounds nuw i8, ptr %160, i64 16
  %162 = load <2 x double>, ptr %160, align 8, !tbaa !19
  %163 = load <2 x double>, ptr %161, align 8, !tbaa !19
  %164 = getelementptr inbounds nuw double, ptr %147, i64 %159
  %165 = getelementptr inbounds nuw i8, ptr %164, i64 16
  store <2 x double> %162, ptr %164, align 8, !tbaa !19
  store <2 x double> %163, ptr %165, align 8, !tbaa !19
  %166 = add nuw i64 %159, 4
  %167 = icmp eq i64 %166, %157
  br i1 %167, label %168, label %158, !llvm.loop !51

168:                                              ; preds = %158
  %169 = icmp eq i64 %3, %157
  br i1 %169, label %172, label %170

170:                                              ; preds = %149, %168
  %171 = phi i64 [ 0, %149 ], [ %157, %168 ]
  br label %178

172:                                              ; preds = %178, %168, %141
  %173 = icmp ult i64 %3, %138
  br i1 %173, label %174, label %185

174:                                              ; preds = %172
  %175 = shl i64 %3, 3
  %176 = getelementptr i8, ptr %147, i64 %175
  %177 = shl nuw i64 %139, 3
  tail call void @llvm.memset.p0.i64(ptr align 8 %176, i8 0, i64 %177, i1 false), !tbaa !19
  br label %185

178:                                              ; preds = %170, %178
  %179 = phi i64 [ %183, %178 ], [ %171, %170 ]
  %180 = getelementptr inbounds nuw double, ptr %143, i64 %179
  %181 = load double, ptr %180, align 8, !tbaa !19
  %182 = getelementptr inbounds nuw double, ptr %147, i64 %179
  store double %181, ptr %182, align 8, !tbaa !19
  %183 = add nuw i64 %179, 1
  %184 = icmp eq i64 %183, %3
  br i1 %184, label %172, label %178, !llvm.loop !52

185:                                              ; preds = %174, %172, %136
  ret i64 %139
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN10polynomialIdE3fftERKS0_(ptr dead_on_unwind noalias writable sret(%class.polynomial.0) align 8 %0, ptr noundef nonnull align 8 dereferenceable(24) %1) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %4 = load i64, ptr %3, align 8, !tbaa !13
  %5 = tail call noundef i64 @_ZN10polynomialIdE4log2Em(i64 noundef %4)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !53)
  %6 = load i64, ptr %3, align 8, !tbaa !13, !noalias !53
  %7 = tail call noundef i64 @_ZN10polynomialIdE4log2Em(i64 noundef %6), !noalias !53
  %8 = load i64, ptr %3, align 8, !tbaa !13, !noalias !53
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %0, align 8, !tbaa !11, !alias.scope !53
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %8, ptr %10, align 8, !tbaa !40, !alias.scope !53
  %11 = icmp ugt i64 %8, 1152921504606846975
  %12 = shl i64 %8, 4
  %13 = select i1 %11, i64 -1, i64 %12
  %14 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %13) #15, !noalias !53
  %15 = icmp eq i64 %8, 0
  br i1 %15, label %16, label %17

16:                                               ; preds = %2
  store ptr %14, ptr %9, align 8, !tbaa !35, !alias.scope !53
  br label %44

17:                                               ; preds = %2
  tail call void @llvm.memset.p0.i64(ptr nonnull align 8 %14, i8 0, i64 %12, i1 false), !noalias !53
  store ptr %14, ptr %9, align 8, !tbaa !35, !alias.scope !53
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %19 = load ptr, ptr %18, align 8, !tbaa !17, !noalias !53
  %20 = trunc i64 %7 to i32
  %21 = add i32 %20, -1
  %22 = shl nuw i32 1, %21
  %23 = sext i32 %22 to i64
  br label %24

24:                                               ; preds = %39, %17
  %25 = phi i64 [ 0, %17 ], [ %42, %39 ]
  %26 = getelementptr inbounds nuw double, ptr %19, i64 %25
  %27 = load double, ptr %26, align 8, !tbaa !19, !noalias !53
  br label %28

28:                                               ; preds = %28, %24
  %29 = phi i64 [ 0, %24 ], [ %35, %28 ]
  %30 = phi i64 [ 1, %24 ], [ %37, %28 ]
  %31 = phi i64 [ %23, %24 ], [ %36, %28 ]
  %32 = and i64 %30, %25
  %33 = icmp eq i64 %32, 0
  %34 = select i1 %33, i64 0, i64 %31
  %35 = or i64 %34, %29
  %36 = lshr i64 %31, 1
  %37 = shl i64 %30, 1
  %38 = icmp ult i64 %31, 2
  br i1 %38, label %39, label %28, !llvm.loop !56

39:                                               ; preds = %28
  %40 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %35
  %41 = insertelement <2 x double> <double poison, double 0.000000e+00>, double %27, i64 0
  store <2 x double> %41, ptr %40, align 8, !noalias !53
  %42 = add nuw i64 %25, 1
  %43 = icmp eq i64 %42, %8
  br i1 %43, label %44, label %24, !llvm.loop !57

44:                                               ; preds = %39, %16
  %45 = icmp eq i64 %5, 0
  br i1 %45, label %128, label %46

46:                                               ; preds = %44, %123
  %47 = phi i64 [ %124, %123 ], [ 2, %44 ]
  %48 = phi i64 [ %125, %123 ], [ 1, %44 ]
  %49 = phi i64 [ %126, %123 ], [ 0, %44 ]
  %50 = uitofp i64 %47 to double
  %51 = tail call noundef { double, double } @__divdc3(double noundef 0.000000e+00, double noundef 0x401921FB54442D18, double noundef %50, double noundef 0.000000e+00) #14
  %52 = extractvalue { double, double } %51, 0
  %53 = extractvalue { double, double } %51, 1
  %54 = insertvalue [2 x double] poison, double %52, 0
  %55 = insertvalue [2 x double] %54, double %53, 1
  %56 = tail call noundef { double, double } @cexp([2 x double] noundef alignstack(8) %55) #14
  %57 = extractvalue { double, double } %56, 0
  %58 = extractvalue { double, double } %56, 1
  %59 = add i64 %48, -1
  %60 = getelementptr %"class.std::complex", ptr %14, i64 %48
  br label %61

61:                                               ; preds = %46, %118
  %62 = phi i64 [ 0, %46 ], [ %121, %118 ]
  %63 = phi double [ 0.000000e+00, %46 ], [ %120, %118 ]
  %64 = phi double [ 1.000000e+00, %46 ], [ %119, %118 ]
  %65 = load i64, ptr %3, align 8, !tbaa !13
  %66 = add i64 %65, -1
  %67 = icmp ugt i64 %62, %66
  br i1 %67, label %104, label %68

68:                                               ; preds = %61, %89
  %69 = phi i64 [ %90, %89 ], [ %65, %61 ]
  %70 = phi i64 [ %101, %89 ], [ %62, %61 ]
  %71 = getelementptr %"class.std::complex", ptr %60, i64 %70
  %72 = load double, ptr %71, align 8
  %73 = getelementptr inbounds nuw i8, ptr %71, i64 8
  %74 = load double, ptr %73, align 8
  %75 = fmul double %64, %72
  %76 = fmul double %63, %74
  %77 = fmul double %64, %74
  %78 = fmul double %63, %72
  %79 = fsub double %75, %76
  %80 = fadd double %78, %77
  %81 = fcmp uno double %79, 0.000000e+00
  br i1 %81, label %82, label %89, !prof !38

82:                                               ; preds = %68
  %83 = fcmp uno double %80, 0.000000e+00
  br i1 %83, label %84, label %89, !prof !38

84:                                               ; preds = %82
  %85 = tail call noundef { double, double } @__muldc3(double noundef %64, double noundef %63, double noundef %72, double noundef %74) #14
  %86 = extractvalue { double, double } %85, 0
  %87 = extractvalue { double, double } %85, 1
  %88 = load i64, ptr %3, align 8, !tbaa !13
  br label %89

89:                                               ; preds = %84, %82, %68
  %90 = phi i64 [ %69, %68 ], [ %69, %82 ], [ %88, %84 ]
  %91 = phi double [ %79, %68 ], [ %79, %82 ], [ %86, %84 ]
  %92 = phi double [ %80, %68 ], [ %80, %82 ], [ %87, %84 ]
  %93 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %70
  %94 = load double, ptr %93, align 8
  %95 = getelementptr inbounds nuw i8, ptr %93, i64 8
  %96 = load double, ptr %95, align 8, !tbaa !42
  %97 = fadd double %91, %94
  %98 = fadd double %92, %96
  store double %97, ptr %93, align 8
  store double %98, ptr %95, align 8, !tbaa !42
  %99 = fsub double %94, %91
  %100 = fsub double %96, %92
  store double %99, ptr %71, align 8
  store double %100, ptr %73, align 8, !tbaa !42
  %101 = add i64 %70, %47
  %102 = add i64 %90, -1
  %103 = icmp ugt i64 %101, %102
  br i1 %103, label %104, label %68, !llvm.loop !58

104:                                              ; preds = %89, %61
  %105 = fmul double %57, %64
  %106 = fmul double %58, %63
  %107 = fmul double %58, %64
  %108 = fmul double %57, %63
  %109 = fsub double %105, %106
  %110 = fadd double %107, %108
  %111 = fcmp uno double %109, 0.000000e+00
  br i1 %111, label %112, label %118, !prof !38

112:                                              ; preds = %104
  %113 = fcmp uno double %110, 0.000000e+00
  br i1 %113, label %114, label %118, !prof !38

114:                                              ; preds = %112
  %115 = tail call noundef { double, double } @__muldc3(double noundef %64, double noundef %63, double noundef %57, double noundef %58) #14
  %116 = extractvalue { double, double } %115, 0
  %117 = extractvalue { double, double } %115, 1
  br label %118

118:                                              ; preds = %104, %112, %114
  %119 = phi double [ %109, %104 ], [ %109, %112 ], [ %116, %114 ]
  %120 = phi double [ %110, %104 ], [ %110, %112 ], [ %117, %114 ]
  %121 = add i64 %62, 1
  %122 = icmp ugt i64 %121, %59
  br i1 %122, label %123, label %61, !llvm.loop !59

123:                                              ; preds = %118
  %124 = shl i64 %47, 1
  %125 = shl i64 %48, 1
  %126 = add nuw i64 %49, 1
  %127 = icmp eq i64 %126, %5
  br i1 %127, label %128, label %46, !llvm.loop !60

128:                                              ; preds = %123, %44
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN10polynomialIdE11inverse_fftERKS_ISt7complexIdEE(ptr dead_on_unwind noalias writable sret(%class.polynomial.0) align 8 %0, ptr noundef nonnull align 8 dereferenceable(24) %1) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %4 = load i64, ptr %3, align 8, !tbaa !40
  %5 = tail call noundef i64 @_ZN10polynomialIdE4log2Em(i64 noundef %4)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !61)
  %6 = load i64, ptr %3, align 8, !tbaa !40, !noalias !61
  %7 = tail call noundef i64 @_ZN10polynomialIdE4log2Em(i64 noundef %6), !noalias !61
  %8 = load i64, ptr %3, align 8, !tbaa !40, !noalias !61
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %0, align 8, !tbaa !11, !alias.scope !61
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %8, ptr %10, align 8, !tbaa !40, !alias.scope !61
  %11 = icmp ugt i64 %8, 1152921504606846975
  %12 = shl i64 %8, 4
  %13 = select i1 %11, i64 -1, i64 %12
  %14 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %13) #15, !noalias !61
  %15 = icmp eq i64 %8, 0
  br i1 %15, label %16, label %17

16:                                               ; preds = %2
  store ptr %14, ptr %9, align 8, !tbaa !35, !alias.scope !61
  br label %43

17:                                               ; preds = %2
  tail call void @llvm.memset.p0.i64(ptr nonnull align 8 %14, i8 0, i64 %12, i1 false), !noalias !61
  store ptr %14, ptr %9, align 8, !tbaa !35, !alias.scope !61
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %19 = load ptr, ptr %18, align 8, !tbaa !35, !noalias !61
  %20 = trunc i64 %7 to i32
  %21 = add i32 %20, -1
  %22 = shl nuw i32 1, %21
  %23 = sext i32 %22 to i64
  br label %24

24:                                               ; preds = %39, %17
  %25 = phi i64 [ 0, %17 ], [ %41, %39 ]
  %26 = getelementptr inbounds nuw %"class.std::complex", ptr %19, i64 %25
  %27 = load <2 x double>, ptr %26, align 8, !noalias !61
  br label %28

28:                                               ; preds = %28, %24
  %29 = phi i64 [ 0, %24 ], [ %35, %28 ]
  %30 = phi i64 [ 1, %24 ], [ %37, %28 ]
  %31 = phi i64 [ %23, %24 ], [ %36, %28 ]
  %32 = and i64 %30, %25
  %33 = icmp eq i64 %32, 0
  %34 = select i1 %33, i64 0, i64 %31
  %35 = or i64 %34, %29
  %36 = lshr i64 %31, 1
  %37 = shl i64 %30, 1
  %38 = icmp ult i64 %31, 2
  br i1 %38, label %39, label %28, !llvm.loop !56

39:                                               ; preds = %28
  %40 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %35
  store <2 x double> %27, ptr %40, align 8, !noalias !61
  %41 = add nuw i64 %25, 1
  %42 = icmp eq i64 %41, %8
  br i1 %42, label %43, label %24, !llvm.loop !64

43:                                               ; preds = %39, %16
  %44 = icmp eq i64 %5, 0
  br i1 %44, label %47, label %75

45:                                               ; preds = %152
  %46 = load i64, ptr %3, align 8, !tbaa !40
  br label %47

47:                                               ; preds = %45, %43
  %48 = phi i64 [ %46, %45 ], [ %8, %43 ]
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %164, label %50

50:                                               ; preds = %47
  %51 = uitofp i64 %48 to double
  %52 = icmp ult i64 %48, 4
  br i1 %52, label %71, label %53

53:                                               ; preds = %50
  %54 = and i64 %48, -4
  %55 = insertelement <2 x double> poison, double %51, i64 0
  %56 = shufflevector <2 x double> %55, <2 x double> poison, <4 x i32> zeroinitializer
  %57 = shufflevector <2 x double> %55, <2 x double> poison, <4 x i32> zeroinitializer
  br label %58

58:                                               ; preds = %58, %53
  %59 = phi i64 [ 0, %53 ], [ %67, %58 ]
  %60 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %59
  %61 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %59
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 32
  %63 = load <4 x double>, ptr %60, align 8
  %64 = load <4 x double>, ptr %62, align 8
  %65 = fdiv <4 x double> %63, %56
  store <4 x double> %65, ptr %60, align 8
  %66 = fdiv <4 x double> %64, %57
  store <4 x double> %66, ptr %62, align 8
  %67 = add nuw i64 %59, 4
  %68 = icmp eq i64 %67, %54
  br i1 %68, label %69, label %58, !llvm.loop !65

69:                                               ; preds = %58
  %70 = icmp eq i64 %48, %54
  br i1 %70, label %164, label %71

71:                                               ; preds = %50, %69
  %72 = phi i64 [ 0, %50 ], [ %54, %69 ]
  %73 = insertelement <2 x double> poison, double %51, i64 0
  %74 = shufflevector <2 x double> %73, <2 x double> poison, <2 x i32> zeroinitializer
  br label %157

75:                                               ; preds = %43, %152
  %76 = phi i64 [ %153, %152 ], [ 2, %43 ]
  %77 = phi i64 [ %154, %152 ], [ 1, %43 ]
  %78 = phi i64 [ %155, %152 ], [ 0, %43 ]
  %79 = uitofp i64 %76 to double
  %80 = tail call noundef { double, double } @__divdc3(double noundef -0.000000e+00, double noundef 0xC01921FB54442D18, double noundef %79, double noundef 0.000000e+00) #14
  %81 = extractvalue { double, double } %80, 0
  %82 = extractvalue { double, double } %80, 1
  %83 = insertvalue [2 x double] poison, double %81, 0
  %84 = insertvalue [2 x double] %83, double %82, 1
  %85 = tail call noundef { double, double } @cexp([2 x double] noundef alignstack(8) %84) #14
  %86 = extractvalue { double, double } %85, 0
  %87 = extractvalue { double, double } %85, 1
  %88 = add i64 %77, -1
  %89 = getelementptr %"class.std::complex", ptr %14, i64 %77
  br label %90

90:                                               ; preds = %75, %147
  %91 = phi i64 [ 0, %75 ], [ %150, %147 ]
  %92 = phi double [ 0.000000e+00, %75 ], [ %149, %147 ]
  %93 = phi double [ 1.000000e+00, %75 ], [ %148, %147 ]
  %94 = load i64, ptr %3, align 8, !tbaa !40
  %95 = add i64 %94, -1
  %96 = icmp ugt i64 %91, %95
  br i1 %96, label %133, label %97

97:                                               ; preds = %90, %118
  %98 = phi i64 [ %119, %118 ], [ %94, %90 ]
  %99 = phi i64 [ %130, %118 ], [ %91, %90 ]
  %100 = getelementptr %"class.std::complex", ptr %89, i64 %99
  %101 = load double, ptr %100, align 8
  %102 = getelementptr inbounds nuw i8, ptr %100, i64 8
  %103 = load double, ptr %102, align 8
  %104 = fmul double %93, %101
  %105 = fmul double %92, %103
  %106 = fmul double %93, %103
  %107 = fmul double %92, %101
  %108 = fsub double %104, %105
  %109 = fadd double %107, %106
  %110 = fcmp uno double %108, 0.000000e+00
  br i1 %110, label %111, label %118, !prof !38

111:                                              ; preds = %97
  %112 = fcmp uno double %109, 0.000000e+00
  br i1 %112, label %113, label %118, !prof !38

113:                                              ; preds = %111
  %114 = tail call noundef { double, double } @__muldc3(double noundef %93, double noundef %92, double noundef %101, double noundef %103) #14
  %115 = extractvalue { double, double } %114, 0
  %116 = extractvalue { double, double } %114, 1
  %117 = load i64, ptr %3, align 8, !tbaa !40
  br label %118

118:                                              ; preds = %113, %111, %97
  %119 = phi i64 [ %98, %97 ], [ %98, %111 ], [ %117, %113 ]
  %120 = phi double [ %108, %97 ], [ %108, %111 ], [ %115, %113 ]
  %121 = phi double [ %109, %97 ], [ %109, %111 ], [ %116, %113 ]
  %122 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %99
  %123 = load double, ptr %122, align 8
  %124 = getelementptr inbounds nuw i8, ptr %122, i64 8
  %125 = load double, ptr %124, align 8, !tbaa !42
  %126 = fadd double %120, %123
  %127 = fadd double %121, %125
  store double %126, ptr %122, align 8
  store double %127, ptr %124, align 8, !tbaa !42
  %128 = fsub double %123, %120
  %129 = fsub double %125, %121
  store double %128, ptr %100, align 8
  store double %129, ptr %102, align 8, !tbaa !42
  %130 = add i64 %99, %76
  %131 = add i64 %119, -1
  %132 = icmp ugt i64 %130, %131
  br i1 %132, label %133, label %97, !llvm.loop !66

133:                                              ; preds = %118, %90
  %134 = fmul double %86, %93
  %135 = fmul double %87, %92
  %136 = fmul double %87, %93
  %137 = fmul double %86, %92
  %138 = fsub double %134, %135
  %139 = fadd double %136, %137
  %140 = fcmp uno double %138, 0.000000e+00
  br i1 %140, label %141, label %147, !prof !38

141:                                              ; preds = %133
  %142 = fcmp uno double %139, 0.000000e+00
  br i1 %142, label %143, label %147, !prof !38

143:                                              ; preds = %141
  %144 = tail call noundef { double, double } @__muldc3(double noundef %93, double noundef %92, double noundef %86, double noundef %87) #14
  %145 = extractvalue { double, double } %144, 0
  %146 = extractvalue { double, double } %144, 1
  br label %147

147:                                              ; preds = %133, %141, %143
  %148 = phi double [ %138, %133 ], [ %138, %141 ], [ %145, %143 ]
  %149 = phi double [ %139, %133 ], [ %139, %141 ], [ %146, %143 ]
  %150 = add i64 %91, 1
  %151 = icmp ugt i64 %150, %88
  br i1 %151, label %152, label %90, !llvm.loop !67

152:                                              ; preds = %147
  %153 = shl i64 %76, 1
  %154 = shl i64 %77, 1
  %155 = add nuw i64 %78, 1
  %156 = icmp eq i64 %155, %5
  br i1 %156, label %45, label %75, !llvm.loop !68

157:                                              ; preds = %71, %157
  %158 = phi i64 [ %162, %157 ], [ %72, %71 ]
  %159 = getelementptr inbounds nuw %"class.std::complex", ptr %14, i64 %158
  %160 = load <2 x double>, ptr %159, align 8
  %161 = fdiv <2 x double> %160, %74
  store <2 x double> %161, ptr %159, align 8
  %162 = add nuw i64 %158, 1
  %163 = icmp ult i64 %162, %48
  br i1 %163, label %157, label %164, !llvm.loop !69

164:                                              ; preds = %157, %69, %47
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN10polynomialISt7complexIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #4 comdat {
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %0, align 8, !tbaa !11
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !tbaa !35
  %4 = icmp eq ptr %3, null
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @_ZdaPv(ptr noundef nonnull %3) #16
  br label %6

6:                                                ; preds = %1, %5
  ret void
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @_ZNSt14overflow_errorC1EPKc(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef) unnamed_addr #5

declare void @__cxa_free_exception(ptr) local_unnamed_addr

; Function Attrs: nounwind
declare void @_ZNSt14overflow_errorD1Ev(ptr noundef nonnull align 8 dereferenceable(16)) unnamed_addr #8

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #9

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i64 @_ZN10polynomialIdE4log2Em(i64 noundef %0) local_unnamed_addr #4 comdat {
  %2 = icmp ugt i64 %0, 1
  br i1 %2, label %3, label %130

3:                                                ; preds = %1
  %4 = icmp eq i64 %0, 2
  br i1 %4, label %130, label %5

5:                                                ; preds = %3
  %6 = icmp ugt i64 %0, 4
  br i1 %6, label %7, label %130

7:                                                ; preds = %5
  %8 = icmp ugt i64 %0, 8
  br i1 %8, label %9, label %130

9:                                                ; preds = %7
  %10 = icmp ugt i64 %0, 16
  br i1 %10, label %11, label %130

11:                                               ; preds = %9
  %12 = icmp ugt i64 %0, 32
  br i1 %12, label %13, label %130

13:                                               ; preds = %11
  %14 = icmp ugt i64 %0, 64
  br i1 %14, label %15, label %130

15:                                               ; preds = %13
  %16 = icmp ugt i64 %0, 128
  br i1 %16, label %17, label %130

17:                                               ; preds = %15
  %18 = icmp ugt i64 %0, 256
  br i1 %18, label %19, label %130

19:                                               ; preds = %17
  %20 = icmp ugt i64 %0, 512
  br i1 %20, label %21, label %130

21:                                               ; preds = %19
  %22 = icmp ugt i64 %0, 1024
  br i1 %22, label %23, label %130

23:                                               ; preds = %21
  %24 = icmp ugt i64 %0, 2048
  br i1 %24, label %25, label %130

25:                                               ; preds = %23
  %26 = icmp ugt i64 %0, 4096
  br i1 %26, label %27, label %130

27:                                               ; preds = %25
  %28 = icmp ugt i64 %0, 8192
  br i1 %28, label %29, label %130

29:                                               ; preds = %27
  %30 = icmp ugt i64 %0, 16384
  br i1 %30, label %31, label %130

31:                                               ; preds = %29
  %32 = icmp ugt i64 %0, 32768
  br i1 %32, label %33, label %130

33:                                               ; preds = %31
  %34 = icmp ugt i64 %0, 65536
  br i1 %34, label %35, label %130

35:                                               ; preds = %33
  %36 = icmp ugt i64 %0, 131072
  br i1 %36, label %37, label %130

37:                                               ; preds = %35
  %38 = icmp ugt i64 %0, 262144
  br i1 %38, label %39, label %130

39:                                               ; preds = %37
  %40 = icmp ugt i64 %0, 524288
  br i1 %40, label %41, label %130

41:                                               ; preds = %39
  %42 = icmp ugt i64 %0, 1048576
  br i1 %42, label %43, label %130

43:                                               ; preds = %41
  %44 = icmp ugt i64 %0, 2097152
  br i1 %44, label %45, label %130

45:                                               ; preds = %43
  %46 = icmp ugt i64 %0, 4194304
  br i1 %46, label %47, label %130

47:                                               ; preds = %45
  %48 = icmp ugt i64 %0, 8388608
  br i1 %48, label %49, label %130

49:                                               ; preds = %47
  %50 = icmp ugt i64 %0, 16777216
  br i1 %50, label %51, label %130

51:                                               ; preds = %49
  %52 = icmp ugt i64 %0, 33554432
  br i1 %52, label %53, label %130

53:                                               ; preds = %51
  %54 = icmp ugt i64 %0, 67108864
  br i1 %54, label %55, label %130

55:                                               ; preds = %53
  %56 = icmp ugt i64 %0, 134217728
  br i1 %56, label %57, label %130

57:                                               ; preds = %55
  %58 = icmp ugt i64 %0, 268435456
  br i1 %58, label %59, label %130

59:                                               ; preds = %57
  %60 = icmp ugt i64 %0, 536870912
  br i1 %60, label %61, label %130

61:                                               ; preds = %59
  %62 = icmp ugt i64 %0, 1073741824
  br i1 %62, label %63, label %130

63:                                               ; preds = %61
  %64 = icmp ugt i64 %0, 2147483648
  br i1 %64, label %65, label %130

65:                                               ; preds = %63
  %66 = icmp ugt i64 %0, 4294967296
  br i1 %66, label %67, label %130

67:                                               ; preds = %65
  %68 = icmp ugt i64 %0, 8589934592
  br i1 %68, label %69, label %130

69:                                               ; preds = %67
  %70 = icmp ugt i64 %0, 17179869184
  br i1 %70, label %71, label %130

71:                                               ; preds = %69
  %72 = icmp ugt i64 %0, 34359738368
  br i1 %72, label %73, label %130

73:                                               ; preds = %71
  %74 = icmp ugt i64 %0, 68719476736
  br i1 %74, label %75, label %130

75:                                               ; preds = %73
  %76 = icmp ugt i64 %0, 137438953472
  br i1 %76, label %77, label %130

77:                                               ; preds = %75
  %78 = icmp ugt i64 %0, 274877906944
  br i1 %78, label %79, label %130

79:                                               ; preds = %77
  %80 = icmp ugt i64 %0, 549755813888
  br i1 %80, label %81, label %130

81:                                               ; preds = %79
  %82 = icmp ugt i64 %0, 1099511627776
  br i1 %82, label %83, label %130

83:                                               ; preds = %81
  %84 = icmp ugt i64 %0, 2199023255552
  br i1 %84, label %85, label %130

85:                                               ; preds = %83
  %86 = icmp ugt i64 %0, 4398046511104
  br i1 %86, label %87, label %130

87:                                               ; preds = %85
  %88 = icmp ugt i64 %0, 8796093022208
  br i1 %88, label %89, label %130

89:                                               ; preds = %87
  %90 = icmp ugt i64 %0, 17592186044416
  br i1 %90, label %91, label %130

91:                                               ; preds = %89
  %92 = icmp ugt i64 %0, 35184372088832
  br i1 %92, label %93, label %130

93:                                               ; preds = %91
  %94 = icmp ugt i64 %0, 70368744177664
  br i1 %94, label %95, label %130

95:                                               ; preds = %93
  %96 = icmp ugt i64 %0, 140737488355328
  br i1 %96, label %97, label %130

97:                                               ; preds = %95
  %98 = icmp ugt i64 %0, 281474976710656
  br i1 %98, label %99, label %130

99:                                               ; preds = %97
  %100 = icmp ugt i64 %0, 562949953421312
  br i1 %100, label %101, label %130

101:                                              ; preds = %99
  %102 = icmp ugt i64 %0, 1125899906842624
  br i1 %102, label %103, label %130

103:                                              ; preds = %101
  %104 = icmp ugt i64 %0, 2251799813685248
  br i1 %104, label %105, label %130

105:                                              ; preds = %103
  %106 = icmp ugt i64 %0, 4503599627370496
  br i1 %106, label %107, label %130

107:                                              ; preds = %105
  %108 = icmp ugt i64 %0, 9007199254740992
  br i1 %108, label %109, label %130

109:                                              ; preds = %107
  %110 = icmp ugt i64 %0, 18014398509481984
  br i1 %110, label %111, label %130

111:                                              ; preds = %109
  %112 = icmp ugt i64 %0, 36028797018963968
  br i1 %112, label %113, label %130

113:                                              ; preds = %111
  %114 = icmp ugt i64 %0, 72057594037927936
  br i1 %114, label %115, label %130

115:                                              ; preds = %113
  %116 = icmp ugt i64 %0, 144115188075855872
  br i1 %116, label %117, label %130

117:                                              ; preds = %115
  %118 = icmp ugt i64 %0, 288230376151711744
  br i1 %118, label %119, label %130

119:                                              ; preds = %117
  %120 = icmp ugt i64 %0, 576460752303423488
  br i1 %120, label %121, label %130

121:                                              ; preds = %119
  %122 = icmp ugt i64 %0, 1152921504606846976
  br i1 %122, label %123, label %130

123:                                              ; preds = %121
  %124 = icmp ugt i64 %0, 2305843009213693952
  br i1 %124, label %125, label %130

125:                                              ; preds = %123
  %126 = icmp ugt i64 %0, 4611686018427387904
  br i1 %126, label %127, label %130

127:                                              ; preds = %125
  %128 = icmp ugt i64 %0, -9223372036854775808
  %129 = select i1 %128, i64 64, i64 63
  br label %130

130:                                              ; preds = %127, %125, %123, %121, %119, %117, %115, %113, %111, %109, %107, %105, %103, %101, %99, %97, %95, %93, %91, %89, %87, %85, %83, %81, %79, %77, %75, %73, %71, %69, %67, %65, %63, %61, %59, %57, %55, %53, %51, %49, %47, %45, %43, %41, %39, %37, %35, %33, %31, %29, %27, %25, %23, %21, %19, %17, %15, %13, %11, %9, %7, %5, %3, %1
  %131 = phi i64 [ 0, %1 ], [ 1, %3 ], [ 2, %5 ], [ 3, %7 ], [ 4, %9 ], [ 5, %11 ], [ 6, %13 ], [ 7, %15 ], [ 8, %17 ], [ 9, %19 ], [ 10, %21 ], [ 11, %23 ], [ 12, %25 ], [ 13, %27 ], [ 14, %29 ], [ 15, %31 ], [ 16, %33 ], [ 17, %35 ], [ 18, %37 ], [ 19, %39 ], [ 20, %41 ], [ 21, %43 ], [ 22, %45 ], [ 23, %47 ], [ 24, %49 ], [ 25, %51 ], [ 26, %53 ], [ 27, %55 ], [ 28, %57 ], [ 29, %59 ], [ 30, %61 ], [ 31, %63 ], [ 32, %65 ], [ 33, %67 ], [ 34, %69 ], [ 35, %71 ], [ 36, %73 ], [ 37, %75 ], [ 38, %77 ], [ 39, %79 ], [ 40, %81 ], [ 41, %83 ], [ 42, %85 ], [ 43, %87 ], [ 44, %89 ], [ 45, %91 ], [ 46, %93 ], [ 47, %95 ], [ 48, %97 ], [ 49, %99 ], [ 50, %101 ], [ 51, %103 ], [ 52, %105 ], [ 53, %107 ], [ 54, %109 ], [ 55, %111 ], [ 56, %113 ], [ 57, %115 ], [ 58, %117 ], [ 59, %119 ], [ 60, %121 ], [ 61, %123 ], [ 62, %125 ], [ %129, %127 ]
  ret i64 %131
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #10

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN10polynomialISt7complexIdEED0Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #4 comdat {
  store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV10polynomialISt7complexIdEE, i64 16), ptr %0, align 8, !tbaa !11
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !tbaa !35
  %4 = icmp eq ptr %3, null
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @_ZdaPv(ptr noundef nonnull %3) #16
  br label %6

6:                                                ; preds = %1, %5
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #16
  ret void
}

; Function Attrs: nounwind
declare { double, double } @cexp([2 x double] noundef alignstack(8)) local_unnamed_addr #8

declare { double, double } @__divdc3(double, double, double, double) local_unnamed_addr

declare { double, double } @__muldc3(double, double, double, double) local_unnamed_addr

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #12

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { cold noreturn }
attributes #10 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #11 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #12 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #13 = { nounwind willreturn memory(read) }
attributes #14 = { nounwind }
attributes #15 = { builtin allocsize(0) }
attributes #16 = { builtin nounwind }
attributes #17 = { noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"vtable pointer", !10, i64 0}
!13 = !{!14, !16, i64 16}
!14 = !{!"_ZTS10polynomialIdE", !15, i64 8, !16, i64 16}
!15 = !{!"p1 double", !8, i64 0}
!16 = !{!"long", !9, i64 0}
!17 = !{!14, !15, i64 8}
!18 = !{!16, !16, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"double", !9, i64 0}
!21 = distinct !{!21, !22}
!22 = !{!"llvm.loop.mustprogress"}
!23 = distinct !{!23, !22, !24, !25}
!24 = !{!"llvm.loop.isvectorized", i32 1}
!25 = !{!"llvm.loop.unroll.runtime.disable"}
!26 = distinct !{!26, !22, !24}
!27 = distinct !{!27, !22, !24, !25}
!28 = distinct !{!28, !22, !24}
!29 = distinct !{!29, !22, !24, !25}
!30 = distinct !{!30, !22, !24}
!31 = distinct !{!31, !22, !24, !25}
!32 = distinct !{!32, !22, !24}
!33 = distinct !{!33, !22, !24, !25}
!34 = distinct !{!34, !22, !24}
!35 = !{!36, !37, i64 8}
!36 = !{!"_ZTS10polynomialISt7complexIdEE", !37, i64 8, !16, i64 16}
!37 = !{!"p1 _ZTSSt7complexIdE", !8, i64 0}
!38 = !{!"branch_weights", i32 1, i32 1048575}
!39 = distinct !{!39, !22}
!40 = !{!36, !16, i64 16}
!41 = !{i64 0, i64 16, !42}
!42 = !{!9, !9, i64 0}
!43 = distinct !{!43, !22}
!44 = !{!45}
!45 = distinct !{!45, !46}
!46 = distinct !{!46, !"LVerDomain"}
!47 = !{!48}
!48 = distinct !{!48, !46}
!49 = distinct !{!49, !22, !24, !25}
!50 = distinct !{!50, !22, !24}
!51 = distinct !{!51, !22, !24, !25}
!52 = distinct !{!52, !22, !24}
!53 = !{!54}
!54 = distinct !{!54, !55, !"_ZN10polynomialIdE11bit_reverseERKS0_: argument 0"}
!55 = distinct !{!55, !"_ZN10polynomialIdE11bit_reverseERKS0_"}
!56 = distinct !{!56, !22}
!57 = distinct !{!57, !22}
!58 = distinct !{!58, !22}
!59 = distinct !{!59, !22}
!60 = distinct !{!60, !22}
!61 = !{!62}
!62 = distinct !{!62, !63, !"_ZN10polynomialIdE11bit_reverseERKS_ISt7complexIdEE: argument 0"}
!63 = distinct !{!63, !"_ZN10polynomialIdE11bit_reverseERKS_ISt7complexIdEE"}
!64 = distinct !{!64, !22}
!65 = distinct !{!65, !22, !24, !25}
!66 = distinct !{!66, !22}
!67 = distinct !{!67, !22}
!68 = distinct !{!68, !22}
!69 = distinct !{!69, !22, !25, !24}
