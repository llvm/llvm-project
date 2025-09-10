; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/lists1.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/lists1.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::__cxx11::list" = type { %"class.std::__cxx11::_List_base" }
%"class.std::__cxx11::_List_base" = type { %"struct.std::__cxx11::_List_base<int, std::allocator<int>>::_List_impl" }
%"struct.std::__cxx11::_List_base<int, std::allocator<int>>::_List_impl" = type { %"struct.std::__detail::_List_node_header" }
%"struct.std::__detail::_List_node_header" = type { %"struct.std::__detail::_List_node_base", i64 }
%"struct.std::__detail::_List_node_base" = type { ptr, ptr }

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"false\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"true\00", align 1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z12list_print_nNSt7__cxx114listIiSaIiEEEi(ptr noundef readonly captures(address) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %1, -1
  %4 = load ptr, ptr %0, align 8, !tbaa !6
  %5 = icmp ne ptr %4, %0
  %6 = icmp sgt i32 %1, 0
  %7 = and i1 %5, %6
  br i1 %7, label %8, label %23

8:                                                ; preds = %2, %17
  %9 = phi ptr [ %19, %17 ], [ %4, %2 ]
  %10 = phi i32 [ %18, %17 ], [ 0, %2 ]
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %12 = load i32, ptr %11, align 4, !tbaa !12
  %13 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %12)
  %14 = icmp slt i32 %10, %3
  br i1 %14, label %15, label %17

15:                                               ; preds = %8
  %16 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 1)
  br label %17

17:                                               ; preds = %8, %15
  %18 = add nuw nsw i32 %10, 1
  %19 = load ptr, ptr %9, align 8, !tbaa !6
  %20 = icmp ne ptr %19, %0
  %21 = icmp slt i32 %18, %1
  %22 = select i1 %20, i1 %21, i1 false
  br i1 %22, label %8, label %23, !llvm.loop !14

23:                                               ; preds = %17, %2
  %24 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !16
  %25 = getelementptr i8, ptr %24, i64 -24
  %26 = load i64, ptr %25, align 8
  %27 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 240
  %29 = load ptr, ptr %28, align 8, !tbaa !18
  %30 = icmp eq ptr %29, null
  br i1 %30, label %31, label %32

31:                                               ; preds = %23
  tail call void @_ZSt16__throw_bad_castv() #9
  unreachable

32:                                               ; preds = %23
  %33 = getelementptr inbounds nuw i8, ptr %29, i64 56
  %34 = load i8, ptr %33, align 8, !tbaa !35
  %35 = icmp eq i8 %34, 0
  br i1 %35, label %39, label %36

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %29, i64 67
  %38 = load i8, ptr %37, align 1, !tbaa !41
  br label %44

39:                                               ; preds = %32
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %29)
  %40 = load ptr, ptr %29, align 8, !tbaa !16
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 48
  %42 = load ptr, ptr %41, align 8
  %43 = tail call noundef i8 %42(ptr noundef nonnull align 8 dereferenceable(570) %29, i8 noundef 10)
  br label %44

44:                                               ; preds = %36, %39
  %45 = phi i8 [ %38, %36 ], [ %43, %39 ]
  %46 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %45)
  %47 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %46)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
  %3 = alloca %"class.std::__cxx11::list", align 8
  %4 = alloca %"class.std::__cxx11::list", align 8
  %5 = alloca %"class.std::__cxx11::list", align 8
  %6 = alloca %"class.std::__cxx11::list", align 8
  %7 = icmp eq i32 %0, 2
  br i1 %7, label %8, label %14

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !42
  %11 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %10, ptr noundef null, i32 noundef 10) #10
  %12 = trunc i64 %11 to i32
  %13 = tail call i32 @llvm.smax.i32(i32 %12, i32 1)
  br label %14

14:                                               ; preds = %8, %2
  %15 = phi i32 [ 1000000, %2 ], [ %13, %8 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #10
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %3, ptr %16, align 8, !tbaa !44
  store ptr %3, ptr %3, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i64 0, ptr %17, align 8, !tbaa !45
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #10
  %18 = zext nneg i32 %15 to i64
  %19 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %4, ptr %19, align 8, !tbaa !44
  store ptr %4, ptr %4, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 0, ptr %20, align 8, !tbaa !45
  br label %21

21:                                               ; preds = %14, %24
  %22 = phi i64 [ %28, %24 ], [ %18, %14 ]
  %23 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #11
          to label %24 unwind label %30

24:                                               ; preds = %21
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store i32 0, ptr %25, align 4, !tbaa !12
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %23, ptr noundef nonnull align 8 dereferenceable(24) %4) #10
  %26 = load i64, ptr %20, align 8, !tbaa !47
  %27 = add i64 %26, 1
  store i64 %27, ptr %20, align 8, !tbaa !47
  %28 = add nsw i64 %22, -1
  %29 = icmp eq i64 %28, 0
  br i1 %29, label %38, label %21, !llvm.loop !50

30:                                               ; preds = %21
  %31 = landingpad { ptr, i32 }
          cleanup
  %32 = load ptr, ptr %4, align 8, !tbaa !6
  %33 = icmp eq ptr %32, %4
  br i1 %33, label %380, label %34

34:                                               ; preds = %30, %34
  %35 = phi ptr [ %36, %34 ], [ %32, %30 ]
  %36 = load ptr, ptr %35, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %35, i64 noundef 24) #12
  %37 = icmp eq ptr %36, %4
  br i1 %37, label %380, label %34, !llvm.loop !51

38:                                               ; preds = %24
  %39 = load ptr, ptr %4, align 8, !tbaa !6
  %40 = icmp eq ptr %39, %4
  br i1 %40, label %65, label %41

41:                                               ; preds = %38, %41
  %42 = phi i32 [ %44, %41 ], [ 1, %38 ]
  %43 = phi ptr [ %45, %41 ], [ %39, %38 ]
  %44 = add nuw nsw i32 %42, 1
  %45 = load ptr, ptr %43, align 8, !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %43, i64 16
  store i32 %42, ptr %46, align 8, !tbaa !12
  %47 = icmp eq ptr %45, %4
  br i1 %47, label %48, label %41, !llvm.loop !52

48:                                               ; preds = %41, %52
  %49 = phi ptr [ %61, %52 ], [ %39, %41 ]
  %50 = load ptr, ptr %3, align 8, !tbaa !6
  %51 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #11
          to label %52 unwind label %63

52:                                               ; preds = %48
  %53 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %54 = getelementptr inbounds nuw i8, ptr %51, i64 16
  %55 = load i32, ptr %53, align 4, !tbaa !12
  store i32 %55, ptr %54, align 4, !tbaa !12
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %51, ptr noundef %50) #10
  %56 = load i64, ptr %17, align 8, !tbaa !47
  %57 = add i64 %56, 1
  store i64 %57, ptr %17, align 8, !tbaa !47
  %58 = load ptr, ptr %4, align 8, !tbaa !6
  %59 = load i64, ptr %20, align 8, !tbaa !47
  %60 = add i64 %59, -1
  store i64 %60, ptr %20, align 8, !tbaa !47
  call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16) %58) #10
  call void @_ZdlPvm(ptr noundef nonnull %58, i64 noundef 24) #12
  %61 = load ptr, ptr %4, align 8, !tbaa !6
  %62 = icmp eq ptr %61, %4
  br i1 %62, label %65, label %48, !llvm.loop !53

63:                                               ; preds = %48
  %64 = landingpad { ptr, i32 }
          cleanup
  br label %372

65:                                               ; preds = %52, %38
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %5, ptr %66, align 8, !tbaa !44
  store ptr %5, ptr %5, align 8, !tbaa !6
  %67 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i64 0, ptr %67, align 8, !tbaa !45
  %68 = load ptr, ptr %3, align 8, !tbaa !6
  %69 = icmp eq ptr %68, %3
  br i1 %69, label %89, label %70

70:                                               ; preds = %65, %73
  %71 = phi ptr [ %79, %73 ], [ %68, %65 ]
  %72 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #11
          to label %73 unwind label %81

73:                                               ; preds = %70
  %74 = getelementptr inbounds nuw i8, ptr %71, i64 16
  %75 = getelementptr inbounds nuw i8, ptr %72, i64 16
  %76 = load i32, ptr %74, align 4, !tbaa !12
  store i32 %76, ptr %75, align 4, !tbaa !12
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %72, ptr noundef nonnull align 8 dereferenceable(24) %5) #10
  %77 = load i64, ptr %67, align 8, !tbaa !47
  %78 = add i64 %77, 1
  store i64 %78, ptr %67, align 8, !tbaa !47
  %79 = load ptr, ptr %71, align 8, !tbaa !6
  %80 = icmp eq ptr %79, %3
  br i1 %80, label %89, label %70, !llvm.loop !54

81:                                               ; preds = %70
  %82 = landingpad { ptr, i32 }
          cleanup
  %83 = load ptr, ptr %5, align 8, !tbaa !6
  %84 = icmp eq ptr %83, %5
  br i1 %84, label %372, label %85

85:                                               ; preds = %81, %85
  %86 = phi ptr [ %87, %85 ], [ %83, %81 ]
  %87 = load ptr, ptr %86, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %86, i64 noundef 24) #12
  %88 = icmp eq ptr %87, %5
  br i1 %88, label %372, label %85, !llvm.loop !51

89:                                               ; preds = %73, %65
  invoke void @_Z12list_print_nNSt7__cxx114listIiSaIiEEEi(ptr noundef nonnull %5, i32 noundef 2)
          to label %90 unwind label %209

90:                                               ; preds = %89
  %91 = load ptr, ptr %5, align 8, !tbaa !6
  %92 = icmp eq ptr %91, %5
  br i1 %92, label %97, label %93

93:                                               ; preds = %90, %93
  %94 = phi ptr [ %95, %93 ], [ %91, %90 ]
  %95 = load ptr, ptr %94, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %94, i64 noundef 24) #12
  %96 = icmp eq ptr %95, %5
  br i1 %96, label %97, label %93, !llvm.loop !51

97:                                               ; preds = %93, %90
  call void @_ZNSt8__detail15_List_node_base10_M_reverseEv(ptr noundef nonnull align 8 dereferenceable(24) %3) #10
  %98 = load ptr, ptr %3, align 8, !tbaa !6
  %99 = icmp eq ptr %98, %3
  br i1 %99, label %113, label %100

100:                                              ; preds = %97, %106
  %101 = phi ptr [ %107, %106 ], [ %98, %97 ]
  %102 = freeze ptr %101
  %103 = getelementptr inbounds nuw i8, ptr %102, i64 16
  %104 = load i32, ptr %103, align 4, !tbaa !12
  %105 = icmp eq i32 %104, 0
  br i1 %105, label %109, label %106

106:                                              ; preds = %100
  %107 = load ptr, ptr %102, align 8, !tbaa !6
  %108 = icmp eq ptr %107, %3
  br i1 %108, label %113, label %100, !llvm.loop !55

109:                                              ; preds = %100
  %110 = icmp eq ptr %102, %3
  %111 = select i1 %110, ptr @.str.1, ptr @.str.2
  %112 = select i1 %110, i64 5, i64 4
  br label %113

113:                                              ; preds = %106, %109, %97
  %114 = phi ptr [ @.str.1, %97 ], [ %111, %109 ], [ @.str.1, %106 ]
  %115 = phi i64 [ 5, %97 ], [ %112, %109 ], [ 5, %106 ]
  %116 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %114, i64 noundef %115)
          to label %117 unwind label %217

117:                                              ; preds = %113
  %118 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !16
  %119 = getelementptr i8, ptr %118, i64 -24
  %120 = load i64, ptr %119, align 8
  %121 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %120
  %122 = getelementptr inbounds nuw i8, ptr %121, i64 240
  %123 = load ptr, ptr %122, align 8, !tbaa !18
  %124 = icmp eq ptr %123, null
  br i1 %124, label %125, label %127

125:                                              ; preds = %117
  invoke void @_ZSt16__throw_bad_castv() #9
          to label %126 unwind label %217

126:                                              ; preds = %125
  unreachable

127:                                              ; preds = %117
  %128 = getelementptr inbounds nuw i8, ptr %123, i64 56
  %129 = load i8, ptr %128, align 8, !tbaa !35
  %130 = icmp eq i8 %129, 0
  br i1 %130, label %134, label %131

131:                                              ; preds = %127
  %132 = getelementptr inbounds nuw i8, ptr %123, i64 67
  %133 = load i8, ptr %132, align 1, !tbaa !41
  br label %140

134:                                              ; preds = %127
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %123)
          to label %135 unwind label %217

135:                                              ; preds = %134
  %136 = load ptr, ptr %123, align 8, !tbaa !16
  %137 = getelementptr inbounds nuw i8, ptr %136, i64 48
  %138 = load ptr, ptr %137, align 8
  %139 = invoke noundef i8 %138(ptr noundef nonnull align 8 dereferenceable(570) %123, i8 noundef 10)
          to label %140 unwind label %217

140:                                              ; preds = %135, %131
  %141 = phi i8 [ %133, %131 ], [ %139, %135 ]
  %142 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %141)
          to label %143 unwind label %217

143:                                              ; preds = %140
  %144 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %142)
          to label %145 unwind label %217

145:                                              ; preds = %143
  %146 = load ptr, ptr %3, align 8, !tbaa !6
  %147 = icmp eq ptr %146, %3
  br i1 %147, label %161, label %148

148:                                              ; preds = %145, %154
  %149 = phi ptr [ %155, %154 ], [ %146, %145 ]
  %150 = freeze ptr %149
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 16
  %152 = load i32, ptr %151, align 4, !tbaa !12
  %153 = icmp eq i32 %152, %15
  br i1 %153, label %157, label %154

154:                                              ; preds = %148
  %155 = load ptr, ptr %150, align 8, !tbaa !6
  %156 = icmp eq ptr %155, %3
  br i1 %156, label %161, label %148, !llvm.loop !55

157:                                              ; preds = %148
  %158 = icmp eq ptr %150, %3
  %159 = select i1 %158, ptr @.str.1, ptr @.str.2
  %160 = select i1 %158, i64 5, i64 4
  br label %161

161:                                              ; preds = %154, %157, %145
  %162 = phi ptr [ @.str.1, %145 ], [ %159, %157 ], [ @.str.1, %154 ]
  %163 = phi i64 [ 5, %145 ], [ %160, %157 ], [ 5, %154 ]
  %164 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %162, i64 noundef %163)
          to label %165 unwind label %219

165:                                              ; preds = %161
  %166 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !16
  %167 = getelementptr i8, ptr %166, i64 -24
  %168 = load i64, ptr %167, align 8
  %169 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %168
  %170 = getelementptr inbounds nuw i8, ptr %169, i64 240
  %171 = load ptr, ptr %170, align 8, !tbaa !18
  %172 = icmp eq ptr %171, null
  br i1 %172, label %173, label %175

173:                                              ; preds = %165
  invoke void @_ZSt16__throw_bad_castv() #9
          to label %174 unwind label %219

174:                                              ; preds = %173
  unreachable

175:                                              ; preds = %165
  %176 = getelementptr inbounds nuw i8, ptr %171, i64 56
  %177 = load i8, ptr %176, align 8, !tbaa !35
  %178 = icmp eq i8 %177, 0
  br i1 %178, label %182, label %179

179:                                              ; preds = %175
  %180 = getelementptr inbounds nuw i8, ptr %171, i64 67
  %181 = load i8, ptr %180, align 1, !tbaa !41
  br label %188

182:                                              ; preds = %175
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %171)
          to label %183 unwind label %219

183:                                              ; preds = %182
  %184 = load ptr, ptr %171, align 8, !tbaa !16
  %185 = getelementptr inbounds nuw i8, ptr %184, i64 48
  %186 = load ptr, ptr %185, align 8
  %187 = invoke noundef i8 %186(ptr noundef nonnull align 8 dereferenceable(570) %171, i8 noundef 10)
          to label %188 unwind label %219

188:                                              ; preds = %183, %179
  %189 = phi i8 [ %181, %179 ], [ %187, %183 ]
  %190 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %189)
          to label %191 unwind label %219

191:                                              ; preds = %188
  %192 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %190)
          to label %193 unwind label %219

193:                                              ; preds = %191
  %194 = lshr i32 %15, 1
  %195 = load ptr, ptr %3, align 8, !tbaa !6
  %196 = icmp eq ptr %195, %3
  br i1 %196, label %226, label %197

197:                                              ; preds = %193, %223
  %198 = phi ptr [ %224, %223 ], [ %195, %193 ]
  %199 = getelementptr inbounds nuw i8, ptr %198, i64 16
  %200 = load i32, ptr %199, align 4, !tbaa !12
  %201 = icmp slt i32 %200, %194
  br i1 %201, label %202, label %223

202:                                              ; preds = %197
  %203 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #11
          to label %204 unwind label %221

204:                                              ; preds = %202
  %205 = getelementptr inbounds nuw i8, ptr %203, i64 16
  %206 = load i32, ptr %199, align 4, !tbaa !12
  store i32 %206, ptr %205, align 4, !tbaa !12
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %203, ptr noundef nonnull align 8 dereferenceable(24) %4) #10
  %207 = load i64, ptr %20, align 8, !tbaa !47
  %208 = add i64 %207, 1
  store i64 %208, ptr %20, align 8, !tbaa !47
  br label %223

209:                                              ; preds = %89
  %210 = landingpad { ptr, i32 }
          cleanup
  %211 = load ptr, ptr %5, align 8, !tbaa !6
  %212 = icmp eq ptr %211, %5
  br i1 %212, label %372, label %213

213:                                              ; preds = %209, %213
  %214 = phi ptr [ %215, %213 ], [ %211, %209 ]
  %215 = load ptr, ptr %214, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %214, i64 noundef 24) #12
  %216 = icmp eq ptr %215, %5
  br i1 %216, label %372, label %213, !llvm.loop !51

217:                                              ; preds = %143, %140, %135, %134, %125, %113
  %218 = landingpad { ptr, i32 }
          cleanup
  br label %372

219:                                              ; preds = %191, %188, %183, %182, %173, %161
  %220 = landingpad { ptr, i32 }
          cleanup
  br label %372

221:                                              ; preds = %202
  %222 = landingpad { ptr, i32 }
          cleanup
  br label %372

223:                                              ; preds = %204, %197
  %224 = load ptr, ptr %198, align 8, !tbaa !6
  %225 = icmp eq ptr %224, %3
  br i1 %225, label %226, label %197, !llvm.loop !56

226:                                              ; preds = %223, %193
  %227 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %6, ptr %227, align 8, !tbaa !44
  store ptr %6, ptr %6, align 8, !tbaa !6
  %228 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i64 0, ptr %228, align 8, !tbaa !45
  %229 = load ptr, ptr %4, align 8, !tbaa !6
  %230 = icmp eq ptr %229, %4
  br i1 %230, label %250, label %231

231:                                              ; preds = %226, %234
  %232 = phi ptr [ %240, %234 ], [ %229, %226 ]
  %233 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #11
          to label %234 unwind label %242

234:                                              ; preds = %231
  %235 = getelementptr inbounds nuw i8, ptr %232, i64 16
  %236 = getelementptr inbounds nuw i8, ptr %233, i64 16
  %237 = load i32, ptr %235, align 4, !tbaa !12
  store i32 %237, ptr %236, align 4, !tbaa !12
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %233, ptr noundef nonnull align 8 dereferenceable(24) %6) #10
  %238 = load i64, ptr %228, align 8, !tbaa !47
  %239 = add i64 %238, 1
  store i64 %239, ptr %228, align 8, !tbaa !47
  %240 = load ptr, ptr %232, align 8, !tbaa !6
  %241 = icmp eq ptr %240, %4
  br i1 %241, label %250, label %231, !llvm.loop !54

242:                                              ; preds = %231
  %243 = landingpad { ptr, i32 }
          cleanup
  %244 = load ptr, ptr %6, align 8, !tbaa !6
  %245 = icmp eq ptr %244, %6
  br i1 %245, label %372, label %246

246:                                              ; preds = %242, %246
  %247 = phi ptr [ %248, %246 ], [ %244, %242 ]
  %248 = load ptr, ptr %247, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %247, i64 noundef 24) #12
  %249 = icmp eq ptr %248, %6
  br i1 %249, label %372, label %246, !llvm.loop !51

250:                                              ; preds = %234, %226
  invoke void @_Z12list_print_nNSt7__cxx114listIiSaIiEEEi(ptr noundef nonnull %6, i32 noundef 10)
          to label %251 unwind label %271

251:                                              ; preds = %250
  %252 = load ptr, ptr %6, align 8, !tbaa !6
  %253 = icmp eq ptr %252, %6
  br i1 %253, label %258, label %254

254:                                              ; preds = %251, %254
  %255 = phi ptr [ %256, %254 ], [ %252, %251 ]
  %256 = load ptr, ptr %255, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %255, i64 noundef 24) #12
  %257 = icmp eq ptr %256, %6
  br i1 %257, label %258, label %254, !llvm.loop !51

258:                                              ; preds = %254, %251
  %259 = load ptr, ptr %4, align 8, !tbaa !6
  %260 = icmp eq ptr %259, %4
  br i1 %260, label %279, label %261

261:                                              ; preds = %258, %261
  %262 = phi ptr [ %269, %261 ], [ %259, %258 ]
  %263 = phi i32 [ %268, %261 ], [ 0, %258 ]
  %264 = getelementptr inbounds nuw i8, ptr %262, i64 16
  %265 = load i32, ptr %264, align 4, !tbaa !12
  %266 = icmp slt i32 %265, 1000
  %267 = select i1 %266, i32 %265, i32 0
  %268 = add nsw i32 %267, %263
  %269 = load ptr, ptr %262, align 8, !tbaa !6
  %270 = icmp eq ptr %269, %4
  br i1 %270, label %279, label %261, !llvm.loop !57

271:                                              ; preds = %250
  %272 = landingpad { ptr, i32 }
          cleanup
  %273 = load ptr, ptr %6, align 8, !tbaa !6
  %274 = icmp eq ptr %273, %6
  br i1 %274, label %372, label %275

275:                                              ; preds = %271, %275
  %276 = phi ptr [ %277, %275 ], [ %273, %271 ]
  %277 = load ptr, ptr %276, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %276, i64 noundef 24) #12
  %278 = icmp eq ptr %277, %6
  br i1 %278, label %372, label %275, !llvm.loop !51

279:                                              ; preds = %261, %258
  %280 = phi i32 [ 0, %258 ], [ %268, %261 ]
  %281 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %280)
          to label %282 unwind label %370

282:                                              ; preds = %279
  %283 = load ptr, ptr %281, align 8, !tbaa !16
  %284 = getelementptr i8, ptr %283, i64 -24
  %285 = load i64, ptr %284, align 8
  %286 = getelementptr inbounds i8, ptr %281, i64 %285
  %287 = getelementptr inbounds nuw i8, ptr %286, i64 240
  %288 = load ptr, ptr %287, align 8, !tbaa !18
  %289 = icmp eq ptr %288, null
  br i1 %289, label %335, label %290

290:                                              ; preds = %282
  %291 = getelementptr inbounds nuw i8, ptr %288, i64 56
  %292 = load i8, ptr %291, align 8, !tbaa !35
  %293 = icmp eq i8 %292, 0
  br i1 %293, label %297, label %294

294:                                              ; preds = %290
  %295 = getelementptr inbounds nuw i8, ptr %288, i64 67
  %296 = load i8, ptr %295, align 1, !tbaa !41
  br label %303

297:                                              ; preds = %290
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %288)
          to label %298 unwind label %370

298:                                              ; preds = %297
  %299 = load ptr, ptr %288, align 8, !tbaa !16
  %300 = getelementptr inbounds nuw i8, ptr %299, i64 48
  %301 = load ptr, ptr %300, align 8
  %302 = invoke noundef i8 %301(ptr noundef nonnull align 8 dereferenceable(570) %288, i8 noundef 10)
          to label %303 unwind label %370

303:                                              ; preds = %298, %294
  %304 = phi i8 [ %296, %294 ], [ %302, %298 ]
  %305 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %281, i8 noundef %304)
          to label %306 unwind label %370

306:                                              ; preds = %303
  %307 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %305)
          to label %308 unwind label %370

308:                                              ; preds = %306
  %309 = load ptr, ptr %3, align 8, !tbaa !6
  %310 = icmp eq ptr %309, %3
  br i1 %310, label %311, label %313

311:                                              ; preds = %308
  %312 = load i64, ptr %20, align 8, !tbaa !47
  br label %317

313:                                              ; preds = %308
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %4, ptr noundef %309, ptr noundef nonnull align 8 dereferenceable(24) %3) #10
  %314 = load i64, ptr %17, align 8, !tbaa !47
  %315 = load i64, ptr %20, align 8, !tbaa !47
  %316 = add i64 %315, %314
  store i64 %316, ptr %20, align 8, !tbaa !47
  store i64 0, ptr %17, align 8, !tbaa !47
  br label %317

317:                                              ; preds = %311, %313
  %318 = phi i64 [ %312, %311 ], [ %316, %313 ]
  %319 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %318)
          to label %320 unwind label %370

320:                                              ; preds = %317
  %321 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %319, ptr noundef nonnull @.str, i64 noundef 1)
          to label %322 unwind label %370

322:                                              ; preds = %320
  %323 = load ptr, ptr %19, align 8, !tbaa !44
  %324 = getelementptr inbounds nuw i8, ptr %323, i64 16
  %325 = load i32, ptr %324, align 4, !tbaa !12
  %326 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %319, i32 noundef %325)
          to label %327 unwind label %370

327:                                              ; preds = %322
  %328 = load ptr, ptr %326, align 8, !tbaa !16
  %329 = getelementptr i8, ptr %328, i64 -24
  %330 = load i64, ptr %329, align 8
  %331 = getelementptr inbounds i8, ptr %326, i64 %330
  %332 = getelementptr inbounds nuw i8, ptr %331, i64 240
  %333 = load ptr, ptr %332, align 8, !tbaa !18
  %334 = icmp eq ptr %333, null
  br i1 %334, label %335, label %337

335:                                              ; preds = %327, %282
  invoke void @_ZSt16__throw_bad_castv() #9
          to label %336 unwind label %370

336:                                              ; preds = %335
  unreachable

337:                                              ; preds = %327
  %338 = getelementptr inbounds nuw i8, ptr %333, i64 56
  %339 = load i8, ptr %338, align 8, !tbaa !35
  %340 = icmp eq i8 %339, 0
  br i1 %340, label %344, label %341

341:                                              ; preds = %337
  %342 = getelementptr inbounds nuw i8, ptr %333, i64 67
  %343 = load i8, ptr %342, align 1, !tbaa !41
  br label %350

344:                                              ; preds = %337
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %333)
          to label %345 unwind label %370

345:                                              ; preds = %344
  %346 = load ptr, ptr %333, align 8, !tbaa !16
  %347 = getelementptr inbounds nuw i8, ptr %346, i64 48
  %348 = load ptr, ptr %347, align 8
  %349 = invoke noundef i8 %348(ptr noundef nonnull align 8 dereferenceable(570) %333, i8 noundef 10)
          to label %350 unwind label %370

350:                                              ; preds = %345, %341
  %351 = phi i8 [ %343, %341 ], [ %349, %345 ]
  %352 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %326, i8 noundef %351)
          to label %353 unwind label %370

353:                                              ; preds = %350
  %354 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %352)
          to label %355 unwind label %370

355:                                              ; preds = %353
  %356 = load ptr, ptr %4, align 8, !tbaa !6
  %357 = icmp eq ptr %356, %4
  br i1 %357, label %362, label %358

358:                                              ; preds = %355, %358
  %359 = phi ptr [ %360, %358 ], [ %356, %355 ]
  %360 = load ptr, ptr %359, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %359, i64 noundef 24) #12
  %361 = icmp eq ptr %360, %4
  br i1 %361, label %362, label %358, !llvm.loop !51

362:                                              ; preds = %358, %355
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #10
  %363 = load ptr, ptr %3, align 8, !tbaa !6
  %364 = icmp eq ptr %363, %3
  br i1 %364, label %369, label %365

365:                                              ; preds = %362, %365
  %366 = phi ptr [ %367, %365 ], [ %363, %362 ]
  %367 = load ptr, ptr %366, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %366, i64 noundef 24) #12
  %368 = icmp eq ptr %367, %3
  br i1 %368, label %369, label %365, !llvm.loop !51

369:                                              ; preds = %365, %362
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #10
  ret i32 0

370:                                              ; preds = %335, %353, %350, %345, %344, %306, %303, %298, %297, %320, %317, %322, %279
  %371 = landingpad { ptr, i32 }
          cleanup
  br label %372

372:                                              ; preds = %85, %213, %246, %275, %271, %242, %221, %219, %217, %209, %81, %63, %370
  %373 = phi { ptr, i32 } [ %371, %370 ], [ %64, %63 ], [ %82, %81 ], [ %210, %209 ], [ %218, %217 ], [ %220, %219 ], [ %222, %221 ], [ %243, %242 ], [ %272, %271 ], [ %272, %275 ], [ %243, %246 ], [ %210, %213 ], [ %82, %85 ]
  %374 = load ptr, ptr %4, align 8, !tbaa !6
  %375 = icmp eq ptr %374, %4
  br i1 %375, label %380, label %376

376:                                              ; preds = %372, %376
  %377 = phi ptr [ %378, %376 ], [ %374, %372 ]
  %378 = load ptr, ptr %377, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %377, i64 noundef 24) #12
  %379 = icmp eq ptr %378, %4
  br i1 %379, label %380, label %376, !llvm.loop !51

380:                                              ; preds = %34, %376, %372, %30
  %381 = phi { ptr, i32 } [ %31, %30 ], [ %373, %372 ], [ %373, %376 ], [ %31, %34 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #10
  %382 = load ptr, ptr %3, align 8, !tbaa !6
  %383 = icmp eq ptr %382, %3
  br i1 %383, label %388, label %384

384:                                              ; preds = %380, %384
  %385 = phi ptr [ %386, %384 ], [ %382, %380 ]
  %386 = load ptr, ptr %385, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %385, i64 noundef 24) #12
  %387 = icmp eq ptr %386, %3
  br i1 %387, label %388, label %384, !llvm.loop !51

388:                                              ; preds = %384, %380
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #10
  resume { ptr, i32 } %381
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #6

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #2

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #7

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16)) local_unnamed_addr #4

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base10_M_reverseEv(ptr noundef nonnull align 8 dereferenceable(16)) local_unnamed_addr #4

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef, ptr noundef) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #8

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { cold noreturn }
attributes #10 = { nounwind }
attributes #11 = { builtin allocsize(0) }
attributes #12 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTSNSt8__detail15_List_node_baseE", !8, i64 0, !8, i64 8}
!8 = !{!"p1 _ZTSNSt8__detail15_List_node_baseE", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !10, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
!16 = !{!17, !17, i64 0}
!17 = !{!"vtable pointer", !11, i64 0}
!18 = !{!19, !32, i64 240}
!19 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !20, i64 0, !29, i64 216, !10, i64 224, !30, i64 225, !31, i64 232, !32, i64 240, !33, i64 248, !34, i64 256}
!20 = !{!"_ZTSSt8ios_base", !21, i64 8, !21, i64 16, !22, i64 24, !23, i64 28, !23, i64 32, !24, i64 40, !25, i64 48, !10, i64 64, !13, i64 192, !26, i64 200, !27, i64 208}
!21 = !{!"long", !10, i64 0}
!22 = !{!"_ZTSSt13_Ios_Fmtflags", !10, i64 0}
!23 = !{!"_ZTSSt12_Ios_Iostate", !10, i64 0}
!24 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !9, i64 0}
!25 = !{!"_ZTSNSt8ios_base6_WordsE", !9, i64 0, !21, i64 8}
!26 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !9, i64 0}
!27 = !{!"_ZTSSt6locale", !28, i64 0}
!28 = !{!"p1 _ZTSNSt6locale5_ImplE", !9, i64 0}
!29 = !{!"p1 _ZTSSo", !9, i64 0}
!30 = !{!"bool", !10, i64 0}
!31 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !9, i64 0}
!32 = !{!"p1 _ZTSSt5ctypeIcE", !9, i64 0}
!33 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!34 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!35 = !{!36, !10, i64 56}
!36 = !{!"_ZTSSt5ctypeIcE", !37, i64 0, !38, i64 16, !30, i64 24, !39, i64 32, !39, i64 40, !40, i64 48, !10, i64 56, !10, i64 57, !10, i64 313, !10, i64 569}
!37 = !{!"_ZTSNSt6locale5facetE", !13, i64 8}
!38 = !{!"p1 _ZTS15__locale_struct", !9, i64 0}
!39 = !{!"p1 int", !9, i64 0}
!40 = !{!"p1 short", !9, i64 0}
!41 = !{!10, !10, i64 0}
!42 = !{!43, !43, i64 0}
!43 = !{!"p1 omnipotent char", !9, i64 0}
!44 = !{!7, !8, i64 8}
!45 = !{!46, !21, i64 16}
!46 = !{!"_ZTSNSt8__detail17_List_node_headerE", !7, i64 0, !21, i64 16}
!47 = !{!48, !21, i64 16}
!48 = !{!"_ZTSNSt7__cxx1110_List_baseIiSaIiEEE", !49, i64 0}
!49 = !{!"_ZTSNSt7__cxx1110_List_baseIiSaIiEE10_List_implE", !46, i64 0}
!50 = distinct !{!50, !15}
!51 = distinct !{!51, !15}
!52 = distinct !{!52, !15}
!53 = distinct !{!53, !15}
!54 = distinct !{!54, !15}
!55 = distinct !{!55, !15}
!56 = distinct !{!56, !15}
!57 = distinct !{!57, !15}
