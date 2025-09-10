; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/lists.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/lists.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::__cxx11::list" = type { %"class.std::__cxx11::_List_base" }
%"class.std::__cxx11::_List_base" = type { %"struct.std::__cxx11::_List_base<unsigned long, std::allocator<unsigned long>>::_List_impl" }
%"struct.std::__cxx11::_List_base<unsigned long, std::allocator<unsigned long>>::_List_impl" = type { %"struct.std::__detail::_List_node_header" }
%"struct.std::__detail::_List_node_header" = type { %"struct.std::__detail::_List_node_base", i64 }
%"struct.std::__detail::_List_node_base" = type { ptr, ptr }

@_ZSt4cout = external global %"class.std::basic_ostream", align 8

; Function Attrs: mustprogress uwtable
define dso_local noundef i64 @_Z10test_listsv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::__cxx11::list", align 8
  %2 = alloca %"class.std::__cxx11::list", align 8
  %3 = alloca %"class.std::__cxx11::list", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #8
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %1, ptr %4, align 8, !tbaa !6
  store ptr %1, ptr %1, align 8, !tbaa !12
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 0, ptr %5, align 8, !tbaa !13
  br label %6

6:                                                ; preds = %9, %0
  %7 = phi i64 [ %13, %9 ], [ 10000, %0 ]
  %8 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #9
          to label %9 unwind label %15

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store i64 0, ptr %10, align 8, !tbaa !16
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(24) %1) #8
  %11 = load i64, ptr %5, align 8, !tbaa !17
  %12 = add i64 %11, 1
  store i64 %12, ptr %5, align 8, !tbaa !17
  %13 = add nsw i64 %7, -1
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %23, label %6, !llvm.loop !20

15:                                               ; preds = %6
  %16 = landingpad { ptr, i32 }
          cleanup
  %17 = load ptr, ptr %1, align 8, !tbaa !12
  %18 = icmp eq ptr %17, %1
  br i1 %18, label %176, label %19

19:                                               ; preds = %15, %19
  %20 = phi ptr [ %21, %19 ], [ %17, %15 ]
  %21 = load ptr, ptr %20, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %20, i64 noundef 24) #10
  %22 = icmp eq ptr %21, %1
  br i1 %22, label %176, label %19, !llvm.loop !22

23:                                               ; preds = %9
  %24 = load ptr, ptr %1, align 8, !tbaa !12
  %25 = icmp eq ptr %24, %1
  br i1 %25, label %26, label %31

26:                                               ; preds = %23
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %27 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %2, ptr %27, align 8, !tbaa !6
  store ptr %2, ptr %2, align 8, !tbaa !12
  %28 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i64 0, ptr %28, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %3, ptr %29, align 8, !tbaa !6
  store ptr %3, ptr %3, align 8, !tbaa !12
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i64 0, ptr %30, align 8, !tbaa !13
  br label %116

31:                                               ; preds = %23, %31
  %32 = phi i64 [ %34, %31 ], [ 1, %23 ]
  %33 = phi ptr [ %35, %31 ], [ %24, %23 ]
  %34 = add nuw nsw i64 %32, 1
  %35 = load ptr, ptr %33, align 8, !tbaa !12
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 16
  store i64 %32, ptr %36, align 8, !tbaa !16
  %37 = icmp eq ptr %35, %1
  br i1 %37, label %38, label %31, !llvm.loop !23

38:                                               ; preds = %31
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %39 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %2, ptr %39, align 8, !tbaa !6
  store ptr %2, ptr %2, align 8, !tbaa !12
  %40 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i64 0, ptr %40, align 8, !tbaa !13
  br label %41

41:                                               ; preds = %38, %44
  %42 = phi ptr [ %50, %44 ], [ %24, %38 ]
  %43 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #9
          to label %44 unwind label %52

44:                                               ; preds = %41
  %45 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %46 = getelementptr inbounds nuw i8, ptr %43, i64 16
  %47 = load i64, ptr %45, align 8, !tbaa !16
  store i64 %47, ptr %46, align 8, !tbaa !16
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %43, ptr noundef nonnull align 8 dereferenceable(24) %2) #8
  %48 = load i64, ptr %40, align 8, !tbaa !17
  %49 = add i64 %48, 1
  store i64 %49, ptr %40, align 8, !tbaa !17
  %50 = load ptr, ptr %42, align 8, !tbaa !12
  %51 = icmp eq ptr %50, %1
  br i1 %51, label %60, label %41, !llvm.loop !24

52:                                               ; preds = %41
  %53 = landingpad { ptr, i32 }
          cleanup
  %54 = load ptr, ptr %2, align 8, !tbaa !12
  %55 = icmp eq ptr %54, %2
  br i1 %55, label %168, label %56

56:                                               ; preds = %52, %56
  %57 = phi ptr [ %58, %56 ], [ %54, %52 ]
  %58 = load ptr, ptr %57, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %57, i64 noundef 24) #10
  %59 = icmp eq ptr %58, %2
  br i1 %59, label %168, label %56, !llvm.loop !22

60:                                               ; preds = %44
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  %61 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %3, ptr %61, align 8, !tbaa !6
  store ptr %3, ptr %3, align 8, !tbaa !12
  %62 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i64 0, ptr %62, align 8, !tbaa !13
  %63 = icmp eq i64 %49, 0
  br i1 %63, label %116, label %64

64:                                               ; preds = %60, %68
  %65 = phi i64 [ %69, %68 ], [ %49, %60 ]
  %66 = load ptr, ptr %2, align 8, !tbaa !12
  %67 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #9
          to label %68 unwind label %81

68:                                               ; preds = %64
  %69 = add i64 %65, -1
  %70 = getelementptr inbounds nuw i8, ptr %66, i64 16
  %71 = getelementptr inbounds nuw i8, ptr %67, i64 16
  %72 = load i64, ptr %70, align 8, !tbaa !16
  store i64 %72, ptr %71, align 8, !tbaa !16
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %67, ptr noundef nonnull align 8 dereferenceable(24) %3) #8
  %73 = load i64, ptr %62, align 8, !tbaa !17
  %74 = add i64 %73, 1
  store i64 %74, ptr %62, align 8, !tbaa !17
  %75 = load ptr, ptr %2, align 8, !tbaa !12
  %76 = load i64, ptr %40, align 8, !tbaa !17
  %77 = add i64 %76, -1
  store i64 %77, ptr %40, align 8, !tbaa !17
  call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16) %75) #8
  call void @_ZdlPvm(ptr noundef nonnull %75, i64 noundef 24) #10
  %78 = icmp eq i64 %69, 0
  br i1 %78, label %98, label %64, !llvm.loop !25

79:                                               ; preds = %101
  %80 = landingpad { ptr, i32 }
          cleanup
  br label %83

81:                                               ; preds = %64
  %82 = landingpad { ptr, i32 }
          cleanup
  br label %83

83:                                               ; preds = %81, %79
  %84 = phi { ptr, i32 } [ %80, %79 ], [ %82, %81 ]
  %85 = load ptr, ptr %3, align 8, !tbaa !12
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %91, label %87

87:                                               ; preds = %83, %87
  %88 = phi ptr [ %89, %87 ], [ %85, %83 ]
  %89 = load ptr, ptr %88, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %88, i64 noundef 24) #10
  %90 = icmp eq ptr %89, %3
  br i1 %90, label %91, label %87, !llvm.loop !22

91:                                               ; preds = %87, %83
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  %92 = load ptr, ptr %2, align 8, !tbaa !12
  %93 = icmp eq ptr %92, %2
  br i1 %93, label %168, label %94

94:                                               ; preds = %91, %94
  %95 = phi ptr [ %96, %94 ], [ %92, %91 ]
  %96 = load ptr, ptr %95, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %95, i64 noundef 24) #10
  %97 = icmp eq ptr %96, %2
  br i1 %97, label %168, label %94, !llvm.loop !22

98:                                               ; preds = %68
  %99 = load i64, ptr %62, align 8, !tbaa !17
  %100 = icmp eq i64 %99, 0
  br i1 %100, label %116, label %101

101:                                              ; preds = %98, %105
  %102 = phi i64 [ %106, %105 ], [ %99, %98 ]
  %103 = load ptr, ptr %61, align 8, !tbaa !6
  %104 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #9
          to label %105 unwind label %79

105:                                              ; preds = %101
  %106 = add i64 %102, -1
  %107 = getelementptr inbounds nuw i8, ptr %103, i64 16
  %108 = getelementptr inbounds nuw i8, ptr %104, i64 16
  %109 = load i64, ptr %107, align 8, !tbaa !16
  store i64 %109, ptr %108, align 8, !tbaa !16
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %104, ptr noundef nonnull align 8 dereferenceable(24) %2) #8
  %110 = load i64, ptr %40, align 8, !tbaa !17
  %111 = add i64 %110, 1
  store i64 %111, ptr %40, align 8, !tbaa !17
  %112 = load ptr, ptr %61, align 8, !tbaa !6
  %113 = load i64, ptr %62, align 8, !tbaa !17
  %114 = add i64 %113, -1
  store i64 %114, ptr %62, align 8, !tbaa !17
  call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16) %112) #8
  call void @_ZdlPvm(ptr noundef nonnull %112, i64 noundef 24) #10
  %115 = icmp eq i64 %106, 0
  br i1 %115, label %116, label %101, !llvm.loop !26

116:                                              ; preds = %105, %26, %60, %98
  %117 = phi ptr [ %40, %98 ], [ %28, %26 ], [ %40, %60 ], [ %40, %105 ]
  call void @_ZNSt8__detail15_List_node_base10_M_reverseEv(ptr noundef nonnull align 8 dereferenceable(24) %1) #8
  %118 = load ptr, ptr %1, align 8, !tbaa !12
  %119 = getelementptr inbounds nuw i8, ptr %118, i64 16
  %120 = load i64, ptr %119, align 8, !tbaa !16
  %121 = icmp eq i64 %120, 10000
  br i1 %121, label %122, label %145

122:                                              ; preds = %116
  %123 = load i64, ptr %5, align 8, !tbaa !17
  %124 = load i64, ptr %117, align 8, !tbaa !17
  %125 = icmp eq i64 %123, %124
  br i1 %125, label %126, label %145

126:                                              ; preds = %122, %134
  %127 = phi ptr [ %130, %134 ], [ %1, %122 ]
  %128 = phi ptr [ %129, %134 ], [ %2, %122 ]
  %129 = load ptr, ptr %128, align 8, !tbaa !12
  %130 = load ptr, ptr %127, align 8, !tbaa !12
  %131 = icmp ne ptr %130, %1
  %132 = icmp ne ptr %129, %2
  %133 = select i1 %131, i1 %132, i1 false
  br i1 %133, label %134, label %140

134:                                              ; preds = %126
  %135 = getelementptr inbounds nuw i8, ptr %130, i64 16
  %136 = load i64, ptr %135, align 8, !tbaa !16
  %137 = getelementptr inbounds nuw i8, ptr %129, i64 16
  %138 = load i64, ptr %137, align 8, !tbaa !16
  %139 = icmp eq i64 %136, %138
  br i1 %139, label %126, label %140, !llvm.loop !27

140:                                              ; preds = %126, %134
  %141 = icmp eq ptr %130, %1
  %142 = icmp eq ptr %129, %2
  %143 = select i1 %141, i1 %142, i1 false
  %144 = select i1 %143, i64 %123, i64 0
  br label %145

145:                                              ; preds = %140, %122, %116
  %146 = phi i64 [ 0, %116 ], [ 0, %122 ], [ %144, %140 ]
  %147 = load ptr, ptr %3, align 8, !tbaa !12
  %148 = icmp eq ptr %147, %3
  br i1 %148, label %153, label %149

149:                                              ; preds = %145, %149
  %150 = phi ptr [ %151, %149 ], [ %147, %145 ]
  %151 = load ptr, ptr %150, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %150, i64 noundef 24) #10
  %152 = icmp eq ptr %151, %3
  br i1 %152, label %153, label %149, !llvm.loop !22

153:                                              ; preds = %149, %145
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  %154 = load ptr, ptr %2, align 8, !tbaa !12
  %155 = icmp eq ptr %154, %2
  br i1 %155, label %160, label %156

156:                                              ; preds = %153, %156
  %157 = phi ptr [ %158, %156 ], [ %154, %153 ]
  %158 = load ptr, ptr %157, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %157, i64 noundef 24) #10
  %159 = icmp eq ptr %158, %2
  br i1 %159, label %160, label %156, !llvm.loop !22

160:                                              ; preds = %156, %153
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  %161 = load ptr, ptr %1, align 8, !tbaa !12
  %162 = icmp eq ptr %161, %1
  br i1 %162, label %167, label %163

163:                                              ; preds = %160, %163
  %164 = phi ptr [ %165, %163 ], [ %161, %160 ]
  %165 = load ptr, ptr %164, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %164, i64 noundef 24) #10
  %166 = icmp eq ptr %165, %1
  br i1 %166, label %167, label %163, !llvm.loop !22

167:                                              ; preds = %163, %160
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #8
  ret i64 %146

168:                                              ; preds = %56, %94, %91, %52
  %169 = phi { ptr, i32 } [ %53, %52 ], [ %84, %91 ], [ %84, %94 ], [ %53, %56 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  %170 = load ptr, ptr %1, align 8, !tbaa !12
  %171 = icmp eq ptr %170, %1
  br i1 %171, label %176, label %172

172:                                              ; preds = %168, %172
  %173 = phi ptr [ %174, %172 ], [ %170, %168 ]
  %174 = load ptr, ptr %173, align 8, !tbaa !12
  call void @_ZdlPvm(ptr noundef nonnull %173, i64 noundef 24) #10
  %175 = icmp eq ptr %174, %1
  br i1 %175, label %176, label %172, !llvm.loop !22

176:                                              ; preds = %19, %172, %168, %15
  %177 = phi { ptr, i32 } [ %16, %15 ], [ %169, %168 ], [ %169, %172 ], [ %16, %19 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #8
  resume { ptr, i32 } %177
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #2 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %13

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !28
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #8
  %8 = trunc i64 %7 to i32
  %9 = icmp slt i32 %8, 1
  br i1 %9, label %13, label %10

10:                                               ; preds = %4
  %11 = and i64 %7, 2147483647
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %20, label %13

13:                                               ; preds = %4, %2, %10
  %14 = phi i64 [ 1, %4 ], [ 3000, %2 ], [ %11, %10 ]
  br label %15

15:                                               ; preds = %13, %15
  %16 = phi i64 [ %18, %15 ], [ %14, %13 ]
  %17 = tail call noundef i64 @_Z10test_listsv()
  %18 = add nsw i64 %16, -1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %20, label %15, !llvm.loop !30

20:                                               ; preds = %15, %10
  %21 = phi i64 [ 0, %10 ], [ %17, %15 ]
  %22 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %21)
  %23 = load ptr, ptr %22, align 8, !tbaa !31
  %24 = getelementptr i8, ptr %23, i64 -24
  %25 = load i64, ptr %24, align 8
  %26 = getelementptr inbounds i8, ptr %22, i64 %25
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 240
  %28 = load ptr, ptr %27, align 8, !tbaa !33
  %29 = icmp eq ptr %28, null
  br i1 %29, label %30, label %31

30:                                               ; preds = %20
  tail call void @_ZSt16__throw_bad_castv() #11
  unreachable

31:                                               ; preds = %20
  %32 = getelementptr inbounds nuw i8, ptr %28, i64 56
  %33 = load i8, ptr %32, align 8, !tbaa !50
  %34 = icmp eq i8 %33, 0
  br i1 %34, label %38, label %35

35:                                               ; preds = %31
  %36 = getelementptr inbounds nuw i8, ptr %28, i64 67
  %37 = load i8, ptr %36, align 1, !tbaa !56
  br label %43

38:                                               ; preds = %31
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %28)
  %39 = load ptr, ptr %28, align 8, !tbaa !31
  %40 = getelementptr inbounds nuw i8, ptr %39, i64 48
  %41 = load ptr, ptr %40, align 8
  %42 = tail call noundef i8 %41(ptr noundef nonnull align 8 dereferenceable(570) %28, i8 noundef 10)
  br label %43

43:                                               ; preds = %35, %38
  %44 = phi i8 [ %37, %35 ], [ %42, %38 ]
  %45 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %22, i8 noundef %44)
  %46 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %45)
  ret i32 0
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #3

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #4

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #5

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16)) local_unnamed_addr #4

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base10_M_reverseEv(ptr noundef nonnull align 8 dereferenceable(16)) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #6

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #7

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #6

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind }
attributes #9 = { builtin allocsize(0) }
attributes #10 = { builtin nounwind }
attributes #11 = { cold noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 8}
!7 = !{!"_ZTSNSt8__detail15_List_node_baseE", !8, i64 0, !8, i64 8}
!8 = !{!"p1 _ZTSNSt8__detail15_List_node_baseE", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!7, !8, i64 0}
!13 = !{!14, !15, i64 16}
!14 = !{!"_ZTSNSt8__detail17_List_node_headerE", !7, i64 0, !15, i64 16}
!15 = !{!"long", !10, i64 0}
!16 = !{!15, !15, i64 0}
!17 = !{!18, !15, i64 16}
!18 = !{!"_ZTSNSt7__cxx1110_List_baseImSaImEEE", !19, i64 0}
!19 = !{!"_ZTSNSt7__cxx1110_List_baseImSaImEE10_List_implE", !14, i64 0}
!20 = distinct !{!20, !21}
!21 = !{!"llvm.loop.mustprogress"}
!22 = distinct !{!22, !21}
!23 = distinct !{!23, !21}
!24 = distinct !{!24, !21}
!25 = distinct !{!25, !21}
!26 = distinct !{!26, !21}
!27 = distinct !{!27, !21}
!28 = !{!29, !29, i64 0}
!29 = !{!"p1 omnipotent char", !9, i64 0}
!30 = distinct !{!30, !21}
!31 = !{!32, !32, i64 0}
!32 = !{!"vtable pointer", !11, i64 0}
!33 = !{!34, !47, i64 240}
!34 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !35, i64 0, !44, i64 216, !10, i64 224, !45, i64 225, !46, i64 232, !47, i64 240, !48, i64 248, !49, i64 256}
!35 = !{!"_ZTSSt8ios_base", !15, i64 8, !15, i64 16, !36, i64 24, !37, i64 28, !37, i64 32, !38, i64 40, !39, i64 48, !10, i64 64, !40, i64 192, !41, i64 200, !42, i64 208}
!36 = !{!"_ZTSSt13_Ios_Fmtflags", !10, i64 0}
!37 = !{!"_ZTSSt12_Ios_Iostate", !10, i64 0}
!38 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !9, i64 0}
!39 = !{!"_ZTSNSt8ios_base6_WordsE", !9, i64 0, !15, i64 8}
!40 = !{!"int", !10, i64 0}
!41 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !9, i64 0}
!42 = !{!"_ZTSSt6locale", !43, i64 0}
!43 = !{!"p1 _ZTSNSt6locale5_ImplE", !9, i64 0}
!44 = !{!"p1 _ZTSSo", !9, i64 0}
!45 = !{!"bool", !10, i64 0}
!46 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !9, i64 0}
!47 = !{!"p1 _ZTSSt5ctypeIcE", !9, i64 0}
!48 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!49 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!50 = !{!51, !10, i64 56}
!51 = !{!"_ZTSSt5ctypeIcE", !52, i64 0, !53, i64 16, !45, i64 24, !54, i64 32, !54, i64 40, !55, i64 48, !10, i64 56, !10, i64 57, !10, i64 313, !10, i64 569}
!52 = !{!"_ZTSNSt6locale5facetE", !40, i64 8}
!53 = !{!"p1 _ZTS15__locale_struct", !9, i64 0}
!54 = !{!"p1 int", !9, i64 0}
!55 = !{!"p1 short", !9, i64 0}
!56 = !{!10, !10, i64 0}
