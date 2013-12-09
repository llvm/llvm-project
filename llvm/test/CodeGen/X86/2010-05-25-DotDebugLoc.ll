; RUN: llc -mtriple=x86_64-pc-linux -O2 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux -O2 -regalloc=basic < %s | FileCheck %s
; Test to check .debug_loc support. This test case emits many debug_loc entries.

; CHECK: Loc expr size
; CHECK-NEXT: .short
; CHECK-NEXT: .Ltmp
; CHECK-NEXT: DW_OP_reg

%0 = type { double }

define hidden %0 @__divsc3(float %a, float %b, float %c, float %d) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata !{float %a}, i64 0, metadata !0)
  tail call void @llvm.dbg.value(metadata !{float %b}, i64 0, metadata !11)
  tail call void @llvm.dbg.value(metadata !{float %c}, i64 0, metadata !12)
  tail call void @llvm.dbg.value(metadata !{float %d}, i64 0, metadata !13)
  %0 = tail call float @fabsf(float %c) nounwind readnone, !dbg !19 ; <float> [#uses=1]
  %1 = tail call float @fabsf(float %d) nounwind readnone, !dbg !19 ; <float> [#uses=1]
  %2 = fcmp olt float %0, %1, !dbg !19            ; <i1> [#uses=1]
  br i1 %2, label %bb, label %bb1, !dbg !19

bb:                                               ; preds = %entry
  %3 = fdiv float %c, %d, !dbg !20                ; <float> [#uses=3]
  tail call void @llvm.dbg.value(metadata !{float %3}, i64 0, metadata !16), !dbg !20
  %4 = fmul float %3, %c, !dbg !21                ; <float> [#uses=1]
  %5 = fadd float %4, %d, !dbg !21                ; <float> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{float %5}, i64 0, metadata !14), !dbg !21
  %6 = fmul float %3, %a, !dbg !22                ; <float> [#uses=1]
  %7 = fadd float %6, %b, !dbg !22                ; <float> [#uses=1]
  %8 = fdiv float %7, %5, !dbg !22                ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %8}, i64 0, metadata !17), !dbg !22
  %9 = fmul float %3, %b, !dbg !23                ; <float> [#uses=1]
  %10 = fsub float %9, %a, !dbg !23               ; <float> [#uses=1]
  %11 = fdiv float %10, %5, !dbg !23              ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %11}, i64 0, metadata !18), !dbg !23
  br label %bb2, !dbg !23

bb1:                                              ; preds = %entry
  %12 = fdiv float %d, %c, !dbg !24               ; <float> [#uses=3]
  tail call void @llvm.dbg.value(metadata !{float %12}, i64 0, metadata !16), !dbg !24
  %13 = fmul float %12, %d, !dbg !25              ; <float> [#uses=1]
  %14 = fadd float %13, %c, !dbg !25              ; <float> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{float %14}, i64 0, metadata !14), !dbg !25
  %15 = fmul float %12, %b, !dbg !26              ; <float> [#uses=1]
  %16 = fadd float %15, %a, !dbg !26              ; <float> [#uses=1]
  %17 = fdiv float %16, %14, !dbg !26             ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %17}, i64 0, metadata !17), !dbg !26
  %18 = fmul float %12, %a, !dbg !27              ; <float> [#uses=1]
  %19 = fsub float %b, %18, !dbg !27              ; <float> [#uses=1]
  %20 = fdiv float %19, %14, !dbg !27             ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %20}, i64 0, metadata !18), !dbg !27
  br label %bb2, !dbg !27

bb2:                                              ; preds = %bb1, %bb
  %y.0 = phi float [ %11, %bb ], [ %20, %bb1 ]    ; <float> [#uses=5]
  %x.0 = phi float [ %8, %bb ], [ %17, %bb1 ]     ; <float> [#uses=5]
  %21 = fcmp uno float %x.0, 0.000000e+00, !dbg !28 ; <i1> [#uses=1]
  %22 = fcmp uno float %y.0, 0.000000e+00, !dbg !28 ; <i1> [#uses=1]
  %or.cond = and i1 %21, %22                      ; <i1> [#uses=1]
  br i1 %or.cond, label %bb4, label %bb46, !dbg !28

bb4:                                              ; preds = %bb2
  %23 = fcmp une float %c, 0.000000e+00, !dbg !29 ; <i1> [#uses=1]
  %24 = fcmp une float %d, 0.000000e+00, !dbg !29 ; <i1> [#uses=1]
  %or.cond93 = or i1 %23, %24                     ; <i1> [#uses=1]
  br i1 %or.cond93, label %bb9, label %bb6, !dbg !29

bb6:                                              ; preds = %bb4
  %25 = fcmp uno float %a, 0.000000e+00, !dbg !29 ; <i1> [#uses=1]
  %26 = fcmp uno float %b, 0.000000e+00, !dbg !29 ; <i1> [#uses=1]
  %or.cond94 = and i1 %25, %26                    ; <i1> [#uses=1]
  br i1 %or.cond94, label %bb9, label %bb8, !dbg !29

bb8:                                              ; preds = %bb6
  %27 = tail call float @copysignf(float 0x7FF0000000000000, float %c) nounwind readnone, !dbg !30 ; <float> [#uses=2]
  %28 = fmul float %27, %a, !dbg !30              ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %28}, i64 0, metadata !17), !dbg !30
  %29 = fmul float %27, %b, !dbg !31              ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %29}, i64 0, metadata !18), !dbg !31
  br label %bb46, !dbg !31

bb9:                                              ; preds = %bb6, %bb4
  %30 = fcmp ord float %a, 0.000000e+00           ; <i1> [#uses=1]
  %31 = fsub float %a, %a, !dbg !32               ; <float> [#uses=3]
  %32 = fcmp uno float %31, 0.000000e+00          ; <i1> [#uses=1]
  %33 = and i1 %30, %32, !dbg !32                 ; <i1> [#uses=2]
  br i1 %33, label %bb14, label %bb11, !dbg !32

bb11:                                             ; preds = %bb9
  %34 = fcmp ord float %b, 0.000000e+00           ; <i1> [#uses=1]
  %35 = fsub float %b, %b, !dbg !32               ; <float> [#uses=1]
  %36 = fcmp uno float %35, 0.000000e+00          ; <i1> [#uses=1]
  %37 = and i1 %34, %36, !dbg !32                 ; <i1> [#uses=1]
  br i1 %37, label %bb14, label %bb27, !dbg !32

bb14:                                             ; preds = %bb11, %bb9
  %38 = fsub float %c, %c, !dbg !32               ; <float> [#uses=1]
  %39 = fcmp ord float %38, 0.000000e+00          ; <i1> [#uses=1]
  br i1 %39, label %bb15, label %bb27, !dbg !32

bb15:                                             ; preds = %bb14
  %40 = fsub float %d, %d, !dbg !32               ; <float> [#uses=1]
  %41 = fcmp ord float %40, 0.000000e+00          ; <i1> [#uses=1]
  br i1 %41, label %bb16, label %bb27, !dbg !32

bb16:                                             ; preds = %bb15
  %iftmp.0.0 = select i1 %33, float 1.000000e+00, float 0.000000e+00 ; <float> [#uses=1]
  %42 = tail call float @copysignf(float %iftmp.0.0, float %a) nounwind readnone, !dbg !33 ; <float> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{float %42}, i64 0, metadata !0), !dbg !33
  %43 = fcmp ord float %b, 0.000000e+00           ; <i1> [#uses=1]
  %44 = fsub float %b, %b, !dbg !34               ; <float> [#uses=1]
  %45 = fcmp uno float %44, 0.000000e+00          ; <i1> [#uses=1]
  %46 = and i1 %43, %45, !dbg !34                 ; <i1> [#uses=1]
  %iftmp.1.0 = select i1 %46, float 1.000000e+00, float 0.000000e+00 ; <float> [#uses=1]
  %47 = tail call float @copysignf(float %iftmp.1.0, float %b) nounwind readnone, !dbg !34 ; <float> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{float %47}, i64 0, metadata !11), !dbg !34
  %48 = fmul float %42, %c, !dbg !35              ; <float> [#uses=1]
  %49 = fmul float %47, %d, !dbg !35              ; <float> [#uses=1]
  %50 = fadd float %48, %49, !dbg !35             ; <float> [#uses=1]
  %51 = fmul float %50, 0x7FF0000000000000, !dbg !35 ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %51}, i64 0, metadata !17), !dbg !35
  %52 = fmul float %47, %c, !dbg !36              ; <float> [#uses=1]
  %53 = fmul float %42, %d, !dbg !36              ; <float> [#uses=1]
  %54 = fsub float %52, %53, !dbg !36             ; <float> [#uses=1]
  %55 = fmul float %54, 0x7FF0000000000000, !dbg !36 ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %55}, i64 0, metadata !18), !dbg !36
  br label %bb46, !dbg !36

bb27:                                             ; preds = %bb15, %bb14, %bb11
  %56 = fcmp ord float %c, 0.000000e+00           ; <i1> [#uses=1]
  %57 = fsub float %c, %c, !dbg !37               ; <float> [#uses=1]
  %58 = fcmp uno float %57, 0.000000e+00          ; <i1> [#uses=1]
  %59 = and i1 %56, %58, !dbg !37                 ; <i1> [#uses=2]
  br i1 %59, label %bb33, label %bb30, !dbg !37

bb30:                                             ; preds = %bb27
  %60 = fcmp ord float %d, 0.000000e+00           ; <i1> [#uses=1]
  %61 = fsub float %d, %d, !dbg !37               ; <float> [#uses=1]
  %62 = fcmp uno float %61, 0.000000e+00          ; <i1> [#uses=1]
  %63 = and i1 %60, %62, !dbg !37                 ; <i1> [#uses=1]
  %64 = fcmp ord float %31, 0.000000e+00          ; <i1> [#uses=1]
  %or.cond95 = and i1 %63, %64                    ; <i1> [#uses=1]
  br i1 %or.cond95, label %bb34, label %bb46, !dbg !37

bb33:                                             ; preds = %bb27
  %.old = fcmp ord float %31, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %.old, label %bb34, label %bb46, !dbg !37

bb34:                                             ; preds = %bb33, %bb30
  %65 = fsub float %b, %b, !dbg !37               ; <float> [#uses=1]
  %66 = fcmp ord float %65, 0.000000e+00          ; <i1> [#uses=1]
  br i1 %66, label %bb35, label %bb46, !dbg !37

bb35:                                             ; preds = %bb34
  %iftmp.2.0 = select i1 %59, float 1.000000e+00, float 0.000000e+00 ; <float> [#uses=1]
  %67 = tail call float @copysignf(float %iftmp.2.0, float %c) nounwind readnone, !dbg !38 ; <float> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{float %67}, i64 0, metadata !12), !dbg !38
  %68 = fcmp ord float %d, 0.000000e+00           ; <i1> [#uses=1]
  %69 = fsub float %d, %d, !dbg !39               ; <float> [#uses=1]
  %70 = fcmp uno float %69, 0.000000e+00          ; <i1> [#uses=1]
  %71 = and i1 %68, %70, !dbg !39                 ; <i1> [#uses=1]
  %iftmp.3.0 = select i1 %71, float 1.000000e+00, float 0.000000e+00 ; <float> [#uses=1]
  %72 = tail call float @copysignf(float %iftmp.3.0, float %d) nounwind readnone, !dbg !39 ; <float> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{float %72}, i64 0, metadata !13), !dbg !39
  %73 = fmul float %67, %a, !dbg !40              ; <float> [#uses=1]
  %74 = fmul float %72, %b, !dbg !40              ; <float> [#uses=1]
  %75 = fadd float %73, %74, !dbg !40             ; <float> [#uses=1]
  %76 = fmul float %75, 0.000000e+00, !dbg !40    ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %76}, i64 0, metadata !17), !dbg !40
  %77 = fmul float %67, %b, !dbg !41              ; <float> [#uses=1]
  %78 = fmul float %72, %a, !dbg !41              ; <float> [#uses=1]
  %79 = fsub float %77, %78, !dbg !41             ; <float> [#uses=1]
  %80 = fmul float %79, 0.000000e+00, !dbg !41    ; <float> [#uses=1]
  tail call void @llvm.dbg.value(metadata !{float %80}, i64 0, metadata !18), !dbg !41
  br label %bb46, !dbg !41

bb46:                                             ; preds = %bb35, %bb34, %bb33, %bb30, %bb16, %bb8, %bb2
  %y.1 = phi float [ %80, %bb35 ], [ %y.0, %bb34 ], [ %y.0, %bb33 ], [ %y.0, %bb30 ], [ %55, %bb16 ], [ %29, %bb8 ], [ %y.0, %bb2 ] ; <float> [#uses=2]
  %x.1 = phi float [ %76, %bb35 ], [ %x.0, %bb34 ], [ %x.0, %bb33 ], [ %x.0, %bb30 ], [ %51, %bb16 ], [ %28, %bb8 ], [ %x.0, %bb2 ] ; <float> [#uses=1]
  %81 = fmul float %y.1, 0.000000e+00, !dbg !42   ; <float> [#uses=1]
  %82 = fadd float %y.1, 0.000000e+00, !dbg !42   ; <float> [#uses=1]
  %tmpr = fadd float %x.1, %81, !dbg !42          ; <float> [#uses=1]
  %tmp89 = bitcast float %tmpr to i32             ; <i32> [#uses=1]
  %tmp90 = zext i32 %tmp89 to i64                 ; <i64> [#uses=1]
  %tmp85 = bitcast float %82 to i32               ; <i32> [#uses=1]
  %tmp86 = zext i32 %tmp85 to i64                 ; <i64> [#uses=1]
  %tmp87 = shl i64 %tmp86, 32                     ; <i64> [#uses=1]
  %ins = or i64 %tmp90, %tmp87                    ; <i64> [#uses=1]
  %tmp84 = bitcast i64 %ins to double             ; <double> [#uses=1]
  %mrv75 = insertvalue %0 undef, double %tmp84, 0, !dbg !42 ; <%0> [#uses=1]
  ret %0 %mrv75, !dbg !42
}

declare float @fabsf(float)

declare float @copysignf(float, float) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!48}

!0 = metadata !{i32 786689, metadata !1, metadata !"a", metadata !2, i32 1921, metadata !9, i32 0, null} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 786478, metadata !45, metadata !2, metadata !"__divsc3", metadata !"__divsc3", metadata !"__divsc3", i32 1922, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, %0 (float, float, float, float)* @__divsc3, null, null, metadata !43, i32 1922} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 786473, metadata !45} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786449, metadata !45, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, metadata !"", i32 0, metadata !47, metadata !47, metadata !44, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 786453, metadata !45, metadata !2, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !9, metadata !9, metadata !9, metadata !9}
!6 = metadata !{i32 786454, metadata !46, metadata !7, metadata !"SCtype", i32 170, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_typedef ]
!7 = metadata !{i32 786473, metadata !46} ; [ DW_TAG_file_type ]
!8 = metadata !{i32 786468, metadata !45, metadata !2, metadata !"complex float", i32 0, i64 64, i64 32, i64 0, i32 0, i32 3} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 786454, metadata !46, metadata !7, metadata !"SFtype", i32 167, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ]
!10 = metadata !{i32 786468, metadata !45, metadata !2, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 786689, metadata !1, metadata !"b", metadata !2, i32 1921, metadata !9, i32 0, null} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 786689, metadata !1, metadata !"c", metadata !2, i32 1921, metadata !9, i32 0, null} ; [ DW_TAG_arg_variable ]
!13 = metadata !{i32 786689, metadata !1, metadata !"d", metadata !2, i32 1921, metadata !9, i32 0, null} ; [ DW_TAG_arg_variable ]
!14 = metadata !{i32 786688, metadata !15, metadata !"denom", metadata !2, i32 1923, metadata !9, i32 0, null} ; [ DW_TAG_auto_variable ]
!15 = metadata !{i32 786443, metadata !45, metadata !1, i32 1922, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 786688, metadata !15, metadata !"ratio", metadata !2, i32 1923, metadata !9, i32 0, null} ; [ DW_TAG_auto_variable ]
!17 = metadata !{i32 786688, metadata !15, metadata !"x", metadata !2, i32 1923, metadata !9, i32 0, null} ; [ DW_TAG_auto_variable ]
!18 = metadata !{i32 786688, metadata !15, metadata !"y", metadata !2, i32 1923, metadata !9, i32 0, null} ; [ DW_TAG_auto_variable ]
!19 = metadata !{i32 1929, i32 0, metadata !15, null}
!20 = metadata !{i32 1931, i32 0, metadata !15, null}
!21 = metadata !{i32 1932, i32 0, metadata !15, null}
!22 = metadata !{i32 1933, i32 0, metadata !15, null}
!23 = metadata !{i32 1934, i32 0, metadata !15, null}
!24 = metadata !{i32 1938, i32 0, metadata !15, null}
!25 = metadata !{i32 1939, i32 0, metadata !15, null}
!26 = metadata !{i32 1940, i32 0, metadata !15, null}
!27 = metadata !{i32 1941, i32 0, metadata !15, null}
!28 = metadata !{i32 1946, i32 0, metadata !15, null}
!29 = metadata !{i32 1948, i32 0, metadata !15, null}
!30 = metadata !{i32 1950, i32 0, metadata !15, null}
!31 = metadata !{i32 1951, i32 0, metadata !15, null}
!32 = metadata !{i32 1953, i32 0, metadata !15, null}
!33 = metadata !{i32 1955, i32 0, metadata !15, null}
!34 = metadata !{i32 1956, i32 0, metadata !15, null}
!35 = metadata !{i32 1957, i32 0, metadata !15, null}
!36 = metadata !{i32 1958, i32 0, metadata !15, null}
!37 = metadata !{i32 1960, i32 0, metadata !15, null}
!38 = metadata !{i32 1962, i32 0, metadata !15, null}
!39 = metadata !{i32 1963, i32 0, metadata !15, null}
!40 = metadata !{i32 1964, i32 0, metadata !15, null}
!41 = metadata !{i32 1965, i32 0, metadata !15, null}
!42 = metadata !{i32 1969, i32 0, metadata !15, null}
!43 = metadata !{metadata !0, metadata !11, metadata !12, metadata !13, metadata !14, metadata !16, metadata !17, metadata !18}
!44 = metadata !{metadata !1}
!45 = metadata !{metadata !"libgcc2.c", metadata !"/Users/yash/clean/LG.D/gcc/../../llvmgcc/gcc"}
!46 = metadata !{metadata !"libgcc2.h", metadata !"/Users/yash/clean/LG.D/gcc/../../llvmgcc/gcc"}
!47 = metadata !{i32 0}
!48 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
