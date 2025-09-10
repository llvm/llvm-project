; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/objinst.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/objinst.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }

$_ZN6ToggleD0Ev = comdat any

$_ZN6Toggle8activateEv = comdat any

$_ZN6ToggleD2Ev = comdat any

$_ZN9NthToggleD0Ev = comdat any

$_ZN9NthToggle8activateEv = comdat any

$_ZTV6Toggle = comdat any

$_ZTI6Toggle = comdat any

$_ZTS6Toggle = comdat any

$_ZTV9NthToggle = comdat any

$_ZTI9NthToggle = comdat any

$_ZTS9NthToggle = comdat any

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [5 x i8] c"true\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"false\00", align 1
@_ZTV6Toggle = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI6Toggle, ptr @_ZN6ToggleD2Ev, ptr @_ZN6ToggleD0Ev, ptr @_ZN6Toggle8activateEv] }, comdat, align 8
@_ZTI6Toggle = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS6Toggle }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS6Toggle = linkonce_odr dso_local constant [8 x i8] c"6Toggle\00", comdat, align 1
@_ZTV9NthToggle = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI9NthToggle, ptr @_ZN6ToggleD2Ev, ptr @_ZN9NthToggleD0Ev, ptr @_ZN9NthToggle8activateEv] }, comdat, align 8
@_ZTI9NthToggle = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS9NthToggle, ptr @_ZTI6Toggle }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS9NthToggle = linkonce_odr dso_local constant [11 x i8] c"9NthToggle\00", comdat, align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #8
  br label %8

8:                                                ; preds = %2, %4
  %9 = tail call noalias noundef nonnull dereferenceable(16) ptr @_Znwm(i64 noundef 16) #9
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV6Toggle, i64 16), ptr %9, align 8, !tbaa !11
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store i8 0, ptr %10, align 8, !tbaa !13
  %11 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.1, i64 noundef 5)
  %12 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %13 = getelementptr i8, ptr %12, i64 -24
  %14 = load i64, ptr %13, align 8
  %15 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %14
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 240
  %17 = load ptr, ptr %16, align 8, !tbaa !16
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %20

19:                                               ; preds = %131, %98, %65, %32, %8
  tail call void @_ZSt16__throw_bad_castv() #10
  unreachable

20:                                               ; preds = %8
  %21 = getelementptr inbounds nuw i8, ptr %17, i64 56
  %22 = load i8, ptr %21, align 8, !tbaa !33
  %23 = icmp eq i8 %22, 0
  br i1 %23, label %27, label %24

24:                                               ; preds = %20
  %25 = getelementptr inbounds nuw i8, ptr %17, i64 67
  %26 = load i8, ptr %25, align 1, !tbaa !39
  br label %32

27:                                               ; preds = %20
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %17)
  %28 = load ptr, ptr %17, align 8, !tbaa !11
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 48
  %30 = load ptr, ptr %29, align 8
  %31 = tail call noundef i8 %30(ptr noundef nonnull align 8 dereferenceable(570) %17, i8 noundef 10)
  br label %32

32:                                               ; preds = %24, %27
  %33 = phi i8 [ %26, %24 ], [ %31, %27 ]
  %34 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %33)
  %35 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %34)
  %36 = load ptr, ptr %9, align 8, !tbaa !11
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %38 = load ptr, ptr %37, align 8
  %39 = tail call noundef nonnull align 8 dereferenceable(9) ptr %38(ptr noundef nonnull align 8 dereferenceable(9) %9)
  %40 = getelementptr inbounds nuw i8, ptr %39, i64 8
  %41 = load i8, ptr %40, align 8, !tbaa !13, !range !40, !noundef !41
  %42 = trunc nuw i8 %41 to i1
  %43 = select i1 %42, ptr @.str, ptr @.str.1
  %44 = select i1 %42, i64 4, i64 5
  %45 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %43, i64 noundef %44)
  %46 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %47 = getelementptr i8, ptr %46, i64 -24
  %48 = load i64, ptr %47, align 8
  %49 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %48
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 240
  %51 = load ptr, ptr %50, align 8, !tbaa !16
  %52 = icmp eq ptr %51, null
  br i1 %52, label %19, label %53

53:                                               ; preds = %32
  %54 = getelementptr inbounds nuw i8, ptr %51, i64 56
  %55 = load i8, ptr %54, align 8, !tbaa !33
  %56 = icmp eq i8 %55, 0
  br i1 %56, label %60, label %57

57:                                               ; preds = %53
  %58 = getelementptr inbounds nuw i8, ptr %51, i64 67
  %59 = load i8, ptr %58, align 1, !tbaa !39
  br label %65

60:                                               ; preds = %53
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %51)
  %61 = load ptr, ptr %51, align 8, !tbaa !11
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 48
  %63 = load ptr, ptr %62, align 8
  %64 = tail call noundef i8 %63(ptr noundef nonnull align 8 dereferenceable(570) %51, i8 noundef 10)
  br label %65

65:                                               ; preds = %60, %57
  %66 = phi i8 [ %59, %57 ], [ %64, %60 ]
  %67 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %66)
  %68 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %67)
  %69 = load ptr, ptr %9, align 8, !tbaa !11
  %70 = getelementptr inbounds nuw i8, ptr %69, i64 16
  %71 = load ptr, ptr %70, align 8
  %72 = tail call noundef nonnull align 8 dereferenceable(9) ptr %71(ptr noundef nonnull align 8 dereferenceable(9) %9)
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 8
  %74 = load i8, ptr %73, align 8, !tbaa !13, !range !40, !noundef !41
  %75 = trunc nuw i8 %74 to i1
  %76 = select i1 %75, ptr @.str, ptr @.str.1
  %77 = select i1 %75, i64 4, i64 5
  %78 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %76, i64 noundef %77)
  %79 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %80 = getelementptr i8, ptr %79, i64 -24
  %81 = load i64, ptr %80, align 8
  %82 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %81
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 240
  %84 = load ptr, ptr %83, align 8, !tbaa !16
  %85 = icmp eq ptr %84, null
  br i1 %85, label %19, label %86

86:                                               ; preds = %65
  %87 = getelementptr inbounds nuw i8, ptr %84, i64 56
  %88 = load i8, ptr %87, align 8, !tbaa !33
  %89 = icmp eq i8 %88, 0
  br i1 %89, label %93, label %90

90:                                               ; preds = %86
  %91 = getelementptr inbounds nuw i8, ptr %84, i64 67
  %92 = load i8, ptr %91, align 1, !tbaa !39
  br label %98

93:                                               ; preds = %86
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %84)
  %94 = load ptr, ptr %84, align 8, !tbaa !11
  %95 = getelementptr inbounds nuw i8, ptr %94, i64 48
  %96 = load ptr, ptr %95, align 8
  %97 = tail call noundef i8 %96(ptr noundef nonnull align 8 dereferenceable(570) %84, i8 noundef 10)
  br label %98

98:                                               ; preds = %93, %90
  %99 = phi i8 [ %92, %90 ], [ %97, %93 ]
  %100 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %99)
  %101 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %100)
  %102 = load ptr, ptr %9, align 8, !tbaa !11
  %103 = getelementptr inbounds nuw i8, ptr %102, i64 16
  %104 = load ptr, ptr %103, align 8
  %105 = tail call noundef nonnull align 8 dereferenceable(9) ptr %104(ptr noundef nonnull align 8 dereferenceable(9) %9)
  %106 = getelementptr inbounds nuw i8, ptr %105, i64 8
  %107 = load i8, ptr %106, align 8, !tbaa !13, !range !40, !noundef !41
  %108 = trunc nuw i8 %107 to i1
  %109 = select i1 %108, ptr @.str, ptr @.str.1
  %110 = select i1 %108, i64 4, i64 5
  %111 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %109, i64 noundef %110)
  %112 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %113 = getelementptr i8, ptr %112, i64 -24
  %114 = load i64, ptr %113, align 8
  %115 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %114
  %116 = getelementptr inbounds nuw i8, ptr %115, i64 240
  %117 = load ptr, ptr %116, align 8, !tbaa !16
  %118 = icmp eq ptr %117, null
  br i1 %118, label %19, label %119

119:                                              ; preds = %98
  %120 = getelementptr inbounds nuw i8, ptr %117, i64 56
  %121 = load i8, ptr %120, align 8, !tbaa !33
  %122 = icmp eq i8 %121, 0
  br i1 %122, label %126, label %123

123:                                              ; preds = %119
  %124 = getelementptr inbounds nuw i8, ptr %117, i64 67
  %125 = load i8, ptr %124, align 1, !tbaa !39
  br label %131

126:                                              ; preds = %119
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %117)
  %127 = load ptr, ptr %117, align 8, !tbaa !11
  %128 = getelementptr inbounds nuw i8, ptr %127, i64 48
  %129 = load ptr, ptr %128, align 8
  %130 = tail call noundef i8 %129(ptr noundef nonnull align 8 dereferenceable(570) %117, i8 noundef 10)
  br label %131

131:                                              ; preds = %126, %123
  %132 = phi i8 [ %125, %123 ], [ %130, %126 ]
  %133 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %132)
  %134 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %133)
  %135 = load ptr, ptr %9, align 8, !tbaa !11
  %136 = getelementptr inbounds nuw i8, ptr %135, i64 16
  %137 = load ptr, ptr %136, align 8
  %138 = tail call noundef nonnull align 8 dereferenceable(9) ptr %137(ptr noundef nonnull align 8 dereferenceable(9) %9)
  %139 = getelementptr inbounds nuw i8, ptr %138, i64 8
  %140 = load i8, ptr %139, align 8, !tbaa !13, !range !40, !noundef !41
  %141 = trunc nuw i8 %140 to i1
  %142 = select i1 %141, ptr @.str, ptr @.str.1
  %143 = select i1 %141, i64 4, i64 5
  %144 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %142, i64 noundef %143)
  %145 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %146 = getelementptr i8, ptr %145, i64 -24
  %147 = load i64, ptr %146, align 8
  %148 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %147
  %149 = getelementptr inbounds nuw i8, ptr %148, i64 240
  %150 = load ptr, ptr %149, align 8, !tbaa !16
  %151 = icmp eq ptr %150, null
  br i1 %151, label %19, label %152

152:                                              ; preds = %131
  %153 = getelementptr inbounds nuw i8, ptr %150, i64 56
  %154 = load i8, ptr %153, align 8, !tbaa !33
  %155 = icmp eq i8 %154, 0
  br i1 %155, label %159, label %156

156:                                              ; preds = %152
  %157 = getelementptr inbounds nuw i8, ptr %150, i64 67
  %158 = load i8, ptr %157, align 1, !tbaa !39
  br label %164

159:                                              ; preds = %152
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %150)
  %160 = load ptr, ptr %150, align 8, !tbaa !11
  %161 = getelementptr inbounds nuw i8, ptr %160, i64 48
  %162 = load ptr, ptr %161, align 8
  %163 = tail call noundef i8 %162(ptr noundef nonnull align 8 dereferenceable(570) %150, i8 noundef 10)
  br label %164

164:                                              ; preds = %159, %156
  %165 = phi i8 [ %158, %156 ], [ %163, %159 ]
  %166 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %165)
  %167 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %166)
  %168 = load ptr, ptr %9, align 8, !tbaa !11
  %169 = getelementptr inbounds nuw i8, ptr %168, i64 8
  %170 = load ptr, ptr %169, align 8
  tail call void %170(ptr noundef nonnull align 8 dereferenceable(9) %9) #8
  %171 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %172 = getelementptr i8, ptr %171, i64 -24
  %173 = load i64, ptr %172, align 8
  %174 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %173
  %175 = getelementptr inbounds nuw i8, ptr %174, i64 240
  %176 = load ptr, ptr %175, align 8, !tbaa !16
  %177 = icmp eq ptr %176, null
  br i1 %177, label %178, label %179

178:                                              ; preds = %164
  tail call void @_ZSt16__throw_bad_castv() #10
  unreachable

179:                                              ; preds = %164
  %180 = getelementptr inbounds nuw i8, ptr %176, i64 56
  %181 = load i8, ptr %180, align 8, !tbaa !33
  %182 = icmp eq i8 %181, 0
  br i1 %182, label %186, label %183

183:                                              ; preds = %179
  %184 = getelementptr inbounds nuw i8, ptr %176, i64 67
  %185 = load i8, ptr %184, align 1, !tbaa !39
  br label %191

186:                                              ; preds = %179
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %176)
  %187 = load ptr, ptr %176, align 8, !tbaa !11
  %188 = getelementptr inbounds nuw i8, ptr %187, i64 48
  %189 = load ptr, ptr %188, align 8
  %190 = tail call noundef i8 %189(ptr noundef nonnull align 8 dereferenceable(570) %176, i8 noundef 10)
  br label %191

191:                                              ; preds = %186, %183
  %192 = phi i8 [ %185, %183 ], [ %190, %186 ]
  %193 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %192)
  %194 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %193)
  %195 = tail call noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #9
  %196 = getelementptr inbounds nuw i8, ptr %195, i64 8
  store i8 1, ptr %196, align 8, !tbaa !13
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV9NthToggle, i64 16), ptr %195, align 8, !tbaa !11
  %197 = getelementptr inbounds nuw i8, ptr %195, i64 12
  store <2 x i32> <i32 3, i32 1>, ptr %197, align 4, !tbaa !42
  %198 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 4)
  %199 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %200 = getelementptr i8, ptr %199, i64 -24
  %201 = load i64, ptr %200, align 8
  %202 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %201
  %203 = getelementptr inbounds nuw i8, ptr %202, i64 240
  %204 = load ptr, ptr %203, align 8, !tbaa !16
  %205 = icmp eq ptr %204, null
  br i1 %205, label %206, label %207

206:                                              ; preds = %417, %384, %351, %318, %285, %252, %219, %191
  tail call void @_ZSt16__throw_bad_castv() #10
  unreachable

207:                                              ; preds = %191
  %208 = getelementptr inbounds nuw i8, ptr %204, i64 56
  %209 = load i8, ptr %208, align 8, !tbaa !33
  %210 = icmp eq i8 %209, 0
  br i1 %210, label %214, label %211

211:                                              ; preds = %207
  %212 = getelementptr inbounds nuw i8, ptr %204, i64 67
  %213 = load i8, ptr %212, align 1, !tbaa !39
  br label %219

214:                                              ; preds = %207
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %204)
  %215 = load ptr, ptr %204, align 8, !tbaa !11
  %216 = getelementptr inbounds nuw i8, ptr %215, i64 48
  %217 = load ptr, ptr %216, align 8
  %218 = tail call noundef i8 %217(ptr noundef nonnull align 8 dereferenceable(570) %204, i8 noundef 10)
  br label %219

219:                                              ; preds = %211, %214
  %220 = phi i8 [ %213, %211 ], [ %218, %214 ]
  %221 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %220)
  %222 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %221)
  %223 = load ptr, ptr %195, align 8, !tbaa !11
  %224 = getelementptr inbounds nuw i8, ptr %223, i64 16
  %225 = load ptr, ptr %224, align 8
  %226 = tail call noundef nonnull align 8 dereferenceable(9) ptr %225(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %227 = getelementptr inbounds nuw i8, ptr %226, i64 8
  %228 = load i8, ptr %227, align 8, !tbaa !13, !range !40, !noundef !41
  %229 = trunc nuw i8 %228 to i1
  %230 = select i1 %229, ptr @.str, ptr @.str.1
  %231 = select i1 %229, i64 4, i64 5
  %232 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %230, i64 noundef %231)
  %233 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %234 = getelementptr i8, ptr %233, i64 -24
  %235 = load i64, ptr %234, align 8
  %236 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %235
  %237 = getelementptr inbounds nuw i8, ptr %236, i64 240
  %238 = load ptr, ptr %237, align 8, !tbaa !16
  %239 = icmp eq ptr %238, null
  br i1 %239, label %206, label %240

240:                                              ; preds = %219
  %241 = getelementptr inbounds nuw i8, ptr %238, i64 56
  %242 = load i8, ptr %241, align 8, !tbaa !33
  %243 = icmp eq i8 %242, 0
  br i1 %243, label %247, label %244

244:                                              ; preds = %240
  %245 = getelementptr inbounds nuw i8, ptr %238, i64 67
  %246 = load i8, ptr %245, align 1, !tbaa !39
  br label %252

247:                                              ; preds = %240
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %238)
  %248 = load ptr, ptr %238, align 8, !tbaa !11
  %249 = getelementptr inbounds nuw i8, ptr %248, i64 48
  %250 = load ptr, ptr %249, align 8
  %251 = tail call noundef i8 %250(ptr noundef nonnull align 8 dereferenceable(570) %238, i8 noundef 10)
  br label %252

252:                                              ; preds = %247, %244
  %253 = phi i8 [ %246, %244 ], [ %251, %247 ]
  %254 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %253)
  %255 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %254)
  %256 = load ptr, ptr %195, align 8, !tbaa !11
  %257 = getelementptr inbounds nuw i8, ptr %256, i64 16
  %258 = load ptr, ptr %257, align 8
  %259 = tail call noundef nonnull align 8 dereferenceable(9) ptr %258(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %260 = getelementptr inbounds nuw i8, ptr %259, i64 8
  %261 = load i8, ptr %260, align 8, !tbaa !13, !range !40, !noundef !41
  %262 = trunc nuw i8 %261 to i1
  %263 = select i1 %262, ptr @.str, ptr @.str.1
  %264 = select i1 %262, i64 4, i64 5
  %265 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %263, i64 noundef %264)
  %266 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %267 = getelementptr i8, ptr %266, i64 -24
  %268 = load i64, ptr %267, align 8
  %269 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %268
  %270 = getelementptr inbounds nuw i8, ptr %269, i64 240
  %271 = load ptr, ptr %270, align 8, !tbaa !16
  %272 = icmp eq ptr %271, null
  br i1 %272, label %206, label %273

273:                                              ; preds = %252
  %274 = getelementptr inbounds nuw i8, ptr %271, i64 56
  %275 = load i8, ptr %274, align 8, !tbaa !33
  %276 = icmp eq i8 %275, 0
  br i1 %276, label %280, label %277

277:                                              ; preds = %273
  %278 = getelementptr inbounds nuw i8, ptr %271, i64 67
  %279 = load i8, ptr %278, align 1, !tbaa !39
  br label %285

280:                                              ; preds = %273
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %271)
  %281 = load ptr, ptr %271, align 8, !tbaa !11
  %282 = getelementptr inbounds nuw i8, ptr %281, i64 48
  %283 = load ptr, ptr %282, align 8
  %284 = tail call noundef i8 %283(ptr noundef nonnull align 8 dereferenceable(570) %271, i8 noundef 10)
  br label %285

285:                                              ; preds = %280, %277
  %286 = phi i8 [ %279, %277 ], [ %284, %280 ]
  %287 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %286)
  %288 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %287)
  %289 = load ptr, ptr %195, align 8, !tbaa !11
  %290 = getelementptr inbounds nuw i8, ptr %289, i64 16
  %291 = load ptr, ptr %290, align 8
  %292 = tail call noundef nonnull align 8 dereferenceable(9) ptr %291(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %293 = getelementptr inbounds nuw i8, ptr %292, i64 8
  %294 = load i8, ptr %293, align 8, !tbaa !13, !range !40, !noundef !41
  %295 = trunc nuw i8 %294 to i1
  %296 = select i1 %295, ptr @.str, ptr @.str.1
  %297 = select i1 %295, i64 4, i64 5
  %298 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %296, i64 noundef %297)
  %299 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %300 = getelementptr i8, ptr %299, i64 -24
  %301 = load i64, ptr %300, align 8
  %302 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %301
  %303 = getelementptr inbounds nuw i8, ptr %302, i64 240
  %304 = load ptr, ptr %303, align 8, !tbaa !16
  %305 = icmp eq ptr %304, null
  br i1 %305, label %206, label %306

306:                                              ; preds = %285
  %307 = getelementptr inbounds nuw i8, ptr %304, i64 56
  %308 = load i8, ptr %307, align 8, !tbaa !33
  %309 = icmp eq i8 %308, 0
  br i1 %309, label %313, label %310

310:                                              ; preds = %306
  %311 = getelementptr inbounds nuw i8, ptr %304, i64 67
  %312 = load i8, ptr %311, align 1, !tbaa !39
  br label %318

313:                                              ; preds = %306
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %304)
  %314 = load ptr, ptr %304, align 8, !tbaa !11
  %315 = getelementptr inbounds nuw i8, ptr %314, i64 48
  %316 = load ptr, ptr %315, align 8
  %317 = tail call noundef i8 %316(ptr noundef nonnull align 8 dereferenceable(570) %304, i8 noundef 10)
  br label %318

318:                                              ; preds = %313, %310
  %319 = phi i8 [ %312, %310 ], [ %317, %313 ]
  %320 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %319)
  %321 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %320)
  %322 = load ptr, ptr %195, align 8, !tbaa !11
  %323 = getelementptr inbounds nuw i8, ptr %322, i64 16
  %324 = load ptr, ptr %323, align 8
  %325 = tail call noundef nonnull align 8 dereferenceable(9) ptr %324(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %326 = getelementptr inbounds nuw i8, ptr %325, i64 8
  %327 = load i8, ptr %326, align 8, !tbaa !13, !range !40, !noundef !41
  %328 = trunc nuw i8 %327 to i1
  %329 = select i1 %328, ptr @.str, ptr @.str.1
  %330 = select i1 %328, i64 4, i64 5
  %331 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %329, i64 noundef %330)
  %332 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %333 = getelementptr i8, ptr %332, i64 -24
  %334 = load i64, ptr %333, align 8
  %335 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %334
  %336 = getelementptr inbounds nuw i8, ptr %335, i64 240
  %337 = load ptr, ptr %336, align 8, !tbaa !16
  %338 = icmp eq ptr %337, null
  br i1 %338, label %206, label %339

339:                                              ; preds = %318
  %340 = getelementptr inbounds nuw i8, ptr %337, i64 56
  %341 = load i8, ptr %340, align 8, !tbaa !33
  %342 = icmp eq i8 %341, 0
  br i1 %342, label %346, label %343

343:                                              ; preds = %339
  %344 = getelementptr inbounds nuw i8, ptr %337, i64 67
  %345 = load i8, ptr %344, align 1, !tbaa !39
  br label %351

346:                                              ; preds = %339
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %337)
  %347 = load ptr, ptr %337, align 8, !tbaa !11
  %348 = getelementptr inbounds nuw i8, ptr %347, i64 48
  %349 = load ptr, ptr %348, align 8
  %350 = tail call noundef i8 %349(ptr noundef nonnull align 8 dereferenceable(570) %337, i8 noundef 10)
  br label %351

351:                                              ; preds = %346, %343
  %352 = phi i8 [ %345, %343 ], [ %350, %346 ]
  %353 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %352)
  %354 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %353)
  %355 = load ptr, ptr %195, align 8, !tbaa !11
  %356 = getelementptr inbounds nuw i8, ptr %355, i64 16
  %357 = load ptr, ptr %356, align 8
  %358 = tail call noundef nonnull align 8 dereferenceable(9) ptr %357(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %359 = getelementptr inbounds nuw i8, ptr %358, i64 8
  %360 = load i8, ptr %359, align 8, !tbaa !13, !range !40, !noundef !41
  %361 = trunc nuw i8 %360 to i1
  %362 = select i1 %361, ptr @.str, ptr @.str.1
  %363 = select i1 %361, i64 4, i64 5
  %364 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %362, i64 noundef %363)
  %365 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %366 = getelementptr i8, ptr %365, i64 -24
  %367 = load i64, ptr %366, align 8
  %368 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %367
  %369 = getelementptr inbounds nuw i8, ptr %368, i64 240
  %370 = load ptr, ptr %369, align 8, !tbaa !16
  %371 = icmp eq ptr %370, null
  br i1 %371, label %206, label %372

372:                                              ; preds = %351
  %373 = getelementptr inbounds nuw i8, ptr %370, i64 56
  %374 = load i8, ptr %373, align 8, !tbaa !33
  %375 = icmp eq i8 %374, 0
  br i1 %375, label %379, label %376

376:                                              ; preds = %372
  %377 = getelementptr inbounds nuw i8, ptr %370, i64 67
  %378 = load i8, ptr %377, align 1, !tbaa !39
  br label %384

379:                                              ; preds = %372
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %370)
  %380 = load ptr, ptr %370, align 8, !tbaa !11
  %381 = getelementptr inbounds nuw i8, ptr %380, i64 48
  %382 = load ptr, ptr %381, align 8
  %383 = tail call noundef i8 %382(ptr noundef nonnull align 8 dereferenceable(570) %370, i8 noundef 10)
  br label %384

384:                                              ; preds = %379, %376
  %385 = phi i8 [ %378, %376 ], [ %383, %379 ]
  %386 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %385)
  %387 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %386)
  %388 = load ptr, ptr %195, align 8, !tbaa !11
  %389 = getelementptr inbounds nuw i8, ptr %388, i64 16
  %390 = load ptr, ptr %389, align 8
  %391 = tail call noundef nonnull align 8 dereferenceable(9) ptr %390(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %392 = getelementptr inbounds nuw i8, ptr %391, i64 8
  %393 = load i8, ptr %392, align 8, !tbaa !13, !range !40, !noundef !41
  %394 = trunc nuw i8 %393 to i1
  %395 = select i1 %394, ptr @.str, ptr @.str.1
  %396 = select i1 %394, i64 4, i64 5
  %397 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %395, i64 noundef %396)
  %398 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %399 = getelementptr i8, ptr %398, i64 -24
  %400 = load i64, ptr %399, align 8
  %401 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %400
  %402 = getelementptr inbounds nuw i8, ptr %401, i64 240
  %403 = load ptr, ptr %402, align 8, !tbaa !16
  %404 = icmp eq ptr %403, null
  br i1 %404, label %206, label %405

405:                                              ; preds = %384
  %406 = getelementptr inbounds nuw i8, ptr %403, i64 56
  %407 = load i8, ptr %406, align 8, !tbaa !33
  %408 = icmp eq i8 %407, 0
  br i1 %408, label %412, label %409

409:                                              ; preds = %405
  %410 = getelementptr inbounds nuw i8, ptr %403, i64 67
  %411 = load i8, ptr %410, align 1, !tbaa !39
  br label %417

412:                                              ; preds = %405
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %403)
  %413 = load ptr, ptr %403, align 8, !tbaa !11
  %414 = getelementptr inbounds nuw i8, ptr %413, i64 48
  %415 = load ptr, ptr %414, align 8
  %416 = tail call noundef i8 %415(ptr noundef nonnull align 8 dereferenceable(570) %403, i8 noundef 10)
  br label %417

417:                                              ; preds = %412, %409
  %418 = phi i8 [ %411, %409 ], [ %416, %412 ]
  %419 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %418)
  %420 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %419)
  %421 = load ptr, ptr %195, align 8, !tbaa !11
  %422 = getelementptr inbounds nuw i8, ptr %421, i64 16
  %423 = load ptr, ptr %422, align 8
  %424 = tail call noundef nonnull align 8 dereferenceable(9) ptr %423(ptr noundef nonnull align 8 dereferenceable(20) %195)
  %425 = getelementptr inbounds nuw i8, ptr %424, i64 8
  %426 = load i8, ptr %425, align 8, !tbaa !13, !range !40, !noundef !41
  %427 = trunc nuw i8 %426 to i1
  %428 = select i1 %427, ptr @.str, ptr @.str.1
  %429 = select i1 %427, i64 4, i64 5
  %430 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %428, i64 noundef %429)
  %431 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %432 = getelementptr i8, ptr %431, i64 -24
  %433 = load i64, ptr %432, align 8
  %434 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %433
  %435 = getelementptr inbounds nuw i8, ptr %434, i64 240
  %436 = load ptr, ptr %435, align 8, !tbaa !16
  %437 = icmp eq ptr %436, null
  br i1 %437, label %206, label %438

438:                                              ; preds = %417
  %439 = getelementptr inbounds nuw i8, ptr %436, i64 56
  %440 = load i8, ptr %439, align 8, !tbaa !33
  %441 = icmp eq i8 %440, 0
  br i1 %441, label %445, label %442

442:                                              ; preds = %438
  %443 = getelementptr inbounds nuw i8, ptr %436, i64 67
  %444 = load i8, ptr %443, align 1, !tbaa !39
  br label %450

445:                                              ; preds = %438
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %436)
  %446 = load ptr, ptr %436, align 8, !tbaa !11
  %447 = getelementptr inbounds nuw i8, ptr %446, i64 48
  %448 = load ptr, ptr %447, align 8
  %449 = tail call noundef i8 %448(ptr noundef nonnull align 8 dereferenceable(570) %436, i8 noundef 10)
  br label %450

450:                                              ; preds = %445, %442
  %451 = phi i8 [ %444, %442 ], [ %449, %445 ]
  %452 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %451)
  %453 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %452)
  %454 = load ptr, ptr %195, align 8, !tbaa !11
  %455 = getelementptr inbounds nuw i8, ptr %454, i64 8
  %456 = load ptr, ptr %455, align 8
  tail call void %456(ptr noundef nonnull align 8 dereferenceable(20) %195) #8
  ret i32 0
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN6ToggleD0Ev(ptr noundef nonnull align 8 dereferenceable(9) %0) unnamed_addr #4 comdat {
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 16) #11
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(9) ptr @_ZN6Toggle8activateEv(ptr noundef nonnull align 8 dereferenceable(9) %0) unnamed_addr #4 comdat {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load i8, ptr %2, align 8, !tbaa !13, !range !40, !noundef !41
  %4 = trunc nuw i8 %3 to i1
  %5 = xor i1 %4, true
  %6 = zext i1 %5 to i8
  store i8 %6, ptr %2, align 8, !tbaa !13
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN6ToggleD2Ev(ptr noundef nonnull align 8 dereferenceable(9) %0) unnamed_addr #4 comdat {
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN9NthToggleD0Ev(ptr noundef nonnull align 8 dereferenceable(20) %0) unnamed_addr #5 comdat {
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #11
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(9) ptr @_ZN9NthToggle8activateEv(ptr noundef nonnull align 8 dereferenceable(20) %0) unnamed_addr #4 comdat {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load i32, ptr %2, align 8, !tbaa !43
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 8, !tbaa !43
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %6 = load i32, ptr %5, align 4, !tbaa !45
  %7 = icmp slt i32 %4, %6
  br i1 %7, label %14, label %8

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load i8, ptr %9, align 8, !tbaa !13, !range !40, !noundef !41
  %11 = trunc nuw i8 %10 to i1
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i8
  store i8 %13, ptr %9, align 8, !tbaa !13
  store i32 0, ptr %2, align 8, !tbaa !43
  br label %14

14:                                               ; preds = %8, %1
  ret ptr %0
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #6

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #7

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #6

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { inlinehint mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind }
attributes #9 = { builtin allocsize(0) }
attributes #10 = { cold noreturn }
attributes #11 = { builtin nounwind }

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
!13 = !{!14, !15, i64 8}
!14 = !{!"_ZTS6Toggle", !15, i64 8}
!15 = !{!"bool", !9, i64 0}
!16 = !{!17, !30, i64 240}
!17 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !18, i64 0, !28, i64 216, !9, i64 224, !15, i64 225, !29, i64 232, !30, i64 240, !31, i64 248, !32, i64 256}
!18 = !{!"_ZTSSt8ios_base", !19, i64 8, !19, i64 16, !20, i64 24, !21, i64 28, !21, i64 32, !22, i64 40, !23, i64 48, !9, i64 64, !24, i64 192, !25, i64 200, !26, i64 208}
!19 = !{!"long", !9, i64 0}
!20 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!21 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!22 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!23 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !19, i64 8}
!24 = !{!"int", !9, i64 0}
!25 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!26 = !{!"_ZTSSt6locale", !27, i64 0}
!27 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!28 = !{!"p1 _ZTSSo", !8, i64 0}
!29 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!30 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!31 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!32 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!33 = !{!34, !9, i64 56}
!34 = !{!"_ZTSSt5ctypeIcE", !35, i64 0, !36, i64 16, !15, i64 24, !37, i64 32, !37, i64 40, !38, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!35 = !{!"_ZTSNSt6locale5facetE", !24, i64 8}
!36 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!37 = !{!"p1 int", !8, i64 0}
!38 = !{!"p1 short", !8, i64 0}
!39 = !{!9, !9, i64 0}
!40 = !{i8 0, i8 2}
!41 = !{}
!42 = !{!24, !24, i64 0}
!43 = !{!44, !24, i64 16}
!44 = !{!"_ZTS9NthToggle", !14, i64 0, !24, i64 12, !24, i64 16}
!45 = !{!44, !24, i64 12}
