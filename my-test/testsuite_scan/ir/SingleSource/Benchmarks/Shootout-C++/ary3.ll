; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/ary3.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/ary3.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.1 = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %15

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #7
  %8 = shl i64 %7, 32
  %9 = ashr exact i64 %8, 32
  %10 = icmp ugt i64 %9, 2305843009213693951
  br i1 %10, label %11, label %12

11:                                               ; preds = %4
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.1) #8
  unreachable

12:                                               ; preds = %4
  %13 = trunc i64 %7 to i32
  %14 = icmp eq i64 %8, 0
  br i1 %14, label %36, label %15

15:                                               ; preds = %2, %12
  %16 = phi i32 [ %13, %12 ], [ 1500000, %2 ]
  %17 = phi i64 [ %9, %12 ], [ 1500000, %2 ]
  %18 = shl nuw nsw i64 %17, 2
  %19 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %18) #9
  %20 = getelementptr inbounds nuw i32, ptr %19, i64 %17
  store i32 0, ptr %19, align 4, !tbaa !11
  %21 = icmp eq i64 %17, 1
  br i1 %21, label %25, label %22

22:                                               ; preds = %15
  %23 = getelementptr i8, ptr %19, i64 4
  %24 = add nsw i64 %18, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %23, i8 0, i64 %24, i1 false), !tbaa !11
  br label %25

25:                                               ; preds = %15, %22
  %26 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %18) #9
          to label %27 unwind label %112

27:                                               ; preds = %25
  %28 = getelementptr inbounds nuw i32, ptr %26, i64 %17
  store i32 0, ptr %26, align 4, !tbaa !11
  %29 = getelementptr i8, ptr %26, i64 4
  %30 = add nsw i64 %17, -1
  %31 = icmp eq i64 %30, 0
  br i1 %31, label %36, label %32

32:                                               ; preds = %27
  %33 = add nsw i64 %18, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %29, i8 0, i64 %33, i1 false), !tbaa !11
  %34 = shl nuw nsw i64 %30, 2
  %35 = getelementptr inbounds nuw i8, ptr %29, i64 %34
  br label %36

36:                                               ; preds = %32, %27, %12
  %37 = phi ptr [ %19, %27 ], [ %19, %32 ], [ null, %12 ]
  %38 = phi ptr [ %20, %27 ], [ %20, %32 ], [ null, %12 ]
  %39 = phi i32 [ %16, %27 ], [ %16, %32 ], [ %13, %12 ]
  %40 = phi ptr [ %26, %27 ], [ %26, %32 ], [ null, %12 ]
  %41 = phi ptr [ %28, %27 ], [ %28, %32 ], [ null, %12 ]
  %42 = phi ptr [ %29, %27 ], [ %35, %32 ], [ null, %12 ]
  %43 = icmp sgt i32 %39, 0
  br i1 %43, label %44, label %114

44:                                               ; preds = %36
  %45 = zext nneg i32 %39 to i64
  %46 = icmp ult i32 %39, 8
  br i1 %46, label %63, label %47

47:                                               ; preds = %44
  %48 = and i64 %45, 2147483640
  br label %49

49:                                               ; preds = %49, %47
  %50 = phi i64 [ 0, %47 ], [ %58, %49 ]
  %51 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %47 ], [ %59, %49 ]
  %52 = getelementptr inbounds nuw i32, ptr %37, i64 %50
  %53 = trunc <4 x i64> %51 to <4 x i32>
  %54 = add <4 x i32> %53, splat (i32 1)
  %55 = trunc <4 x i64> %51 to <4 x i32>
  %56 = add <4 x i32> %55, splat (i32 5)
  %57 = getelementptr inbounds nuw i8, ptr %52, i64 16
  store <4 x i32> %54, ptr %52, align 4, !tbaa !11
  store <4 x i32> %56, ptr %57, align 4, !tbaa !11
  %58 = add nuw i64 %50, 8
  %59 = add <4 x i64> %51, splat (i64 8)
  %60 = icmp eq i64 %58, %48
  br i1 %60, label %61, label %49, !llvm.loop !13

61:                                               ; preds = %49
  %62 = icmp eq i64 %48, %45
  br i1 %62, label %65, label %63

63:                                               ; preds = %44, %61
  %64 = phi i64 [ 0, %44 ], [ %48, %61 ]
  br label %106

65:                                               ; preds = %106, %61
  %66 = zext nneg i32 %39 to i64
  %67 = icmp ult i32 %39, 8
  %68 = and i64 %45, 2147483640
  %69 = sub nsw i64 %66, %68
  %70 = icmp eq i64 %68, %45
  br label %71

71:                                               ; preds = %65, %103
  %72 = phi i32 [ %104, %103 ], [ 0, %65 ]
  br i1 %67, label %92, label %73

73:                                               ; preds = %71, %73
  %74 = phi i64 [ %89, %73 ], [ 0, %71 ]
  %75 = xor i64 %74, -1
  %76 = add i64 %75, %66
  %77 = getelementptr inbounds nuw i32, ptr %37, i64 %76
  %78 = getelementptr inbounds i8, ptr %77, i64 -12
  %79 = getelementptr inbounds i8, ptr %77, i64 -28
  %80 = load <4 x i32>, ptr %78, align 4, !tbaa !11
  %81 = load <4 x i32>, ptr %79, align 4, !tbaa !11
  %82 = getelementptr inbounds nuw i32, ptr %40, i64 %76
  %83 = getelementptr inbounds i8, ptr %82, i64 -12
  %84 = getelementptr inbounds i8, ptr %82, i64 -28
  %85 = load <4 x i32>, ptr %83, align 4, !tbaa !11
  %86 = load <4 x i32>, ptr %84, align 4, !tbaa !11
  %87 = add nsw <4 x i32> %85, %80
  %88 = add nsw <4 x i32> %86, %81
  store <4 x i32> %87, ptr %83, align 4, !tbaa !11
  store <4 x i32> %88, ptr %84, align 4, !tbaa !11
  %89 = add nuw i64 %74, 8
  %90 = icmp eq i64 %89, %68
  br i1 %90, label %91, label %73, !llvm.loop !17

91:                                               ; preds = %73
  br i1 %70, label %103, label %92

92:                                               ; preds = %71, %91
  %93 = phi i64 [ %66, %71 ], [ %69, %91 ]
  br label %94

94:                                               ; preds = %92, %94
  %95 = phi i64 [ %96, %94 ], [ %93, %92 ]
  %96 = add nsw i64 %95, -1
  %97 = getelementptr inbounds nuw i32, ptr %37, i64 %96
  %98 = load i32, ptr %97, align 4, !tbaa !11
  %99 = getelementptr inbounds nuw i32, ptr %40, i64 %96
  %100 = load i32, ptr %99, align 4, !tbaa !11
  %101 = add nsw i32 %100, %98
  store i32 %101, ptr %99, align 4, !tbaa !11
  %102 = icmp sgt i64 %95, 1
  br i1 %102, label %94, label %103, !llvm.loop !18

103:                                              ; preds = %94, %91
  %104 = add nuw nsw i32 %72, 1
  %105 = icmp eq i32 %104, 1000
  br i1 %105, label %114, label %71, !llvm.loop !19

106:                                              ; preds = %63, %106
  %107 = phi i64 [ %108, %106 ], [ %64, %63 ]
  %108 = add nuw nsw i64 %107, 1
  %109 = getelementptr inbounds nuw i32, ptr %37, i64 %107
  %110 = trunc nuw nsw i64 %108 to i32
  store i32 %110, ptr %109, align 4, !tbaa !11
  %111 = icmp eq i64 %108, %45
  br i1 %111, label %65, label %106, !llvm.loop !20

112:                                              ; preds = %25
  %113 = landingpad { ptr, i32 }
          cleanup
  br label %167

114:                                              ; preds = %103, %36
  %115 = load i32, ptr %40, align 4, !tbaa !11
  %116 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %115)
          to label %117 unwind label %161

117:                                              ; preds = %114
  %118 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %116, ptr noundef nonnull @.str, i64 noundef 1)
          to label %119 unwind label %161

119:                                              ; preds = %117
  %120 = getelementptr inbounds i8, ptr %42, i64 -4
  %121 = load i32, ptr %120, align 4, !tbaa !11
  %122 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %116, i32 noundef %121)
          to label %123 unwind label %161

123:                                              ; preds = %119
  %124 = load ptr, ptr %122, align 8, !tbaa !21
  %125 = getelementptr i8, ptr %124, i64 -24
  %126 = load i64, ptr %125, align 8
  %127 = getelementptr inbounds i8, ptr %122, i64 %126
  %128 = getelementptr inbounds nuw i8, ptr %127, i64 240
  %129 = load ptr, ptr %128, align 8, !tbaa !23
  %130 = icmp eq ptr %129, null
  br i1 %130, label %131, label %133

131:                                              ; preds = %123
  invoke void @_ZSt16__throw_bad_castv() #8
          to label %132 unwind label %161

132:                                              ; preds = %131
  unreachable

133:                                              ; preds = %123
  %134 = getelementptr inbounds nuw i8, ptr %129, i64 56
  %135 = load i8, ptr %134, align 8, !tbaa !40
  %136 = icmp eq i8 %135, 0
  br i1 %136, label %140, label %137

137:                                              ; preds = %133
  %138 = getelementptr inbounds nuw i8, ptr %129, i64 67
  %139 = load i8, ptr %138, align 1, !tbaa !46
  br label %146

140:                                              ; preds = %133
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %129)
          to label %141 unwind label %161

141:                                              ; preds = %140
  %142 = load ptr, ptr %129, align 8, !tbaa !21
  %143 = getelementptr inbounds nuw i8, ptr %142, i64 48
  %144 = load ptr, ptr %143, align 8
  %145 = invoke noundef i8 %144(ptr noundef nonnull align 8 dereferenceable(570) %129, i8 noundef 10)
          to label %146 unwind label %161

146:                                              ; preds = %141, %137
  %147 = phi i8 [ %139, %137 ], [ %145, %141 ]
  %148 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %122, i8 noundef %147)
          to label %149 unwind label %161

149:                                              ; preds = %146
  %150 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %148)
          to label %151 unwind label %161

151:                                              ; preds = %149
  %152 = ptrtoint ptr %41 to i64
  %153 = ptrtoint ptr %40 to i64
  %154 = sub i64 %152, %153
  tail call void @_ZdlPvm(ptr noundef nonnull %40, i64 noundef %154) #10
  %155 = icmp eq ptr %37, null
  br i1 %155, label %160, label %156

156:                                              ; preds = %151
  %157 = ptrtoint ptr %38 to i64
  %158 = ptrtoint ptr %37 to i64
  %159 = sub i64 %157, %158
  tail call void @_ZdlPvm(ptr noundef nonnull %37, i64 noundef %159) #10
  br label %160

160:                                              ; preds = %151, %156
  ret i32 0

161:                                              ; preds = %114, %119, %117, %131, %140, %141, %146, %149
  %162 = landingpad { ptr, i32 }
          cleanup
  %163 = ptrtoint ptr %41 to i64
  %164 = ptrtoint ptr %40 to i64
  %165 = sub i64 %163, %164
  tail call void @_ZdlPvm(ptr noundef nonnull %40, i64 noundef %165) #10
  %166 = icmp eq ptr %37, null
  br i1 %166, label %174, label %167

167:                                              ; preds = %112, %161
  %168 = phi { ptr, i32 } [ %113, %112 ], [ %162, %161 ]
  %169 = phi ptr [ %20, %112 ], [ %38, %161 ]
  %170 = phi ptr [ %19, %112 ], [ %37, %161 ]
  %171 = ptrtoint ptr %169 to i64
  %172 = ptrtoint ptr %170 to i64
  %173 = sub i64 %171, %172
  tail call void @_ZdlPvm(ptr noundef nonnull %170, i64 noundef %173) #10
  br label %174

174:                                              ; preds = %167, %161
  %175 = phi { ptr, i32 } [ %162, %161 ], [ %168, %167 ]
  resume { ptr, i32 } %175
}

declare i32 @__gxx_personality_v0(...)

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #1

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #3

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #1

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #3

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { nounwind }
attributes #8 = { cold noreturn }
attributes #9 = { builtin allocsize(0) }
attributes #10 = { builtin nounwind }

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
!12 = !{!"int", !9, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !15, !16}
!18 = distinct !{!18, !14, !16, !15}
!19 = distinct !{!19, !14}
!20 = distinct !{!20, !14, !16, !15}
!21 = !{!22, !22, i64 0}
!22 = !{!"vtable pointer", !10, i64 0}
!23 = !{!24, !37, i64 240}
!24 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !25, i64 0, !34, i64 216, !9, i64 224, !35, i64 225, !36, i64 232, !37, i64 240, !38, i64 248, !39, i64 256}
!25 = !{!"_ZTSSt8ios_base", !26, i64 8, !26, i64 16, !27, i64 24, !28, i64 28, !28, i64 32, !29, i64 40, !30, i64 48, !9, i64 64, !12, i64 192, !31, i64 200, !32, i64 208}
!26 = !{!"long", !9, i64 0}
!27 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!28 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!29 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!30 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !26, i64 8}
!31 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!32 = !{!"_ZTSSt6locale", !33, i64 0}
!33 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!34 = !{!"p1 _ZTSSo", !8, i64 0}
!35 = !{!"bool", !9, i64 0}
!36 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!37 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!38 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!39 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!40 = !{!41, !9, i64 56}
!41 = !{!"_ZTSSt5ctypeIcE", !42, i64 0, !43, i64 16, !35, i64 24, !44, i64 32, !44, i64 40, !45, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!42 = !{!"_ZTSNSt6locale5facetE", !12, i64 8}
!43 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!44 = !{!"p1 int", !8, i64 0}
!45 = !{!"p1 short", !8, i64 0}
!46 = !{!9, !9, i64 0}
