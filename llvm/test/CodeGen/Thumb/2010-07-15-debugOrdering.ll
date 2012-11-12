; RUN: llc -mtriple=thumbv6-apple-darwin10 < %s | FileCheck %s
; RUN: opt -strip-debug < %s | llc -mtriple=thumbv6-apple-darwin10 | FileCheck %s
; Stripping out debug info formerly caused the last two multiplies to be emitted in
; the other order.  7797940 (part of it dated 6/29/2010..7/15/2010).

%0 = type { [3 x double] }

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%0*, i32, i32)* @_Z19getClosestDiagonal3ii to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define void @_Z19getClosestDiagonal3ii(%0* noalias sret, i32, i32) nounwind {
; CHECK: blx ___muldf3
; CHECK: blx ___muldf3
; CHECK: beq LBB0
; CHECK: blx ___muldf3
; <label>:3
  switch i32 %1, label %4 [
    i32 0, label %5
    i32 3, label %5
  ]

; <label>:4                                       ; preds = %3
  br label %5, !dbg !0

; <label>:5                                       ; preds = %4, %3, %3
  %storemerge = phi double [ -1.000000e+00, %4 ], [ 1.000000e+00, %3 ], [ 1.000000e+00, %3 ] ; <double> [#uses=1]
  %v_6 = icmp slt i32 %1, 2                         ; <i1> [#uses=1]
  %storemerge1 = select i1 %v_6, double 1.000000e+00, double -1.000000e+00 ; <double> [#uses=3]
  call void @llvm.dbg.value(metadata !{double %storemerge}, i64 0, metadata !91), !dbg !0
  %v_7 = icmp eq i32 %2, 1, !dbg !92                ; <i1> [#uses=1]
  %storemerge2 = select i1 %v_7, double 1.000000e+00, double -1.000000e+00 ; <double> [#uses=3]
  %v_8 = getelementptr inbounds %0* %0, i32 0, i32 0, i32 0 ; <double*> [#uses=1]
  %v_10 = getelementptr inbounds %0* %0, i32 0, i32 0, i32 2 ; <double*> [#uses=1]
  %v_11 = fmul double %storemerge1, %storemerge1, !dbg !93 ; <double> [#uses=1]
  %v_15 = tail call double @sqrt(double %v_11) nounwind readonly, !dbg !93 ; <double> [#uses=1]
  %v_16 = fdiv double 1.000000e+00, %v_15, !dbg !93   ; <double> [#uses=3]
  %v_17 = fmul double %storemerge, %v_16, !dbg !97    ; <double> [#uses=1]
  store double %v_17, double* %v_8, align 4, !dbg !97
  %v_19 = fmul double %storemerge2, %v_16, !dbg !97   ; <double> [#uses=1]
  store double %v_19, double* %v_10, align 4, !dbg !97
  ret void, !dbg !98
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare double @sqrt(double) nounwind readonly

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!0 = metadata !{i32 46, i32 0, metadata !1, null}
!1 = metadata !{i32 524299, metadata !2, i32 44, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 524299, metadata !3, i32 44, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 524334, i32 0, metadata !4, metadata !"getClosestDiagonal3", metadata !"getClosestDiagonal3", metadata !"_Z19getClosestDiagonal3ii", metadata !4, i32 44, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 524329, metadata !"ggEdgeDiscrepancy.cc", metadata !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src", metadata !5} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 524305, i32 0, i32 4, metadata !"ggEdgeDiscrepancy.cc", metadata !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build 00)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!6 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{metadata !8, metadata !22, metadata !22}
!8 = metadata !{i32 524307, metadata !4, metadata !"ggVector3", metadata !9, i32 66, i64 192, i64 32, i64 0, i32 0, null, metadata !10, i32 0, null} ; [ DW_TAG_structure_type ]
!9 = metadata !{i32 524329, metadata !"ggVector3.h", metadata !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src", metadata !5} ; [ DW_TAG_file_type ]
!10 = metadata !{metadata !11, metadata !16, metadata !23, metadata !26, metadata !29, metadata !30, metadata !35, metadata !36, metadata !37, metadata !41, metadata !42, metadata !43, metadata !46, metadata !47, metadata !48, metadata !52, metadata !53, metadata !54, metadata !57, metadata !60, metadata !63, metadata !66, metadata !70, metadata !71, metadata !74, metadata !75, metadata !76, metadata !77, metadata !78, metadata !81, metadata !82, metadata !83, metadata !84, metadata !85, metadata !88, metadata !89, metadata !90}
!11 = metadata !{i32 524301, metadata !8, metadata !"e", metadata !9, i32 160, i64 192, i64 32, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ]
!12 = metadata !{i32 524289, metadata !4, metadata !"", metadata !4, i32 0, i64 192, i64 32, i64 0, i32 0, metadata !13, metadata !14, i32 0, null} ; [ DW_TAG_array_type ]
!13 = metadata !{i32 524324, metadata !4, metadata !"double", metadata !4, i32 0, i64 64, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 524321, i64 0, i64 2}        ; [ DW_TAG_subrange_type ]
!16 = metadata !{i32 524334, i32 0, metadata !8, metadata !"ggVector3", metadata !"ggVector3", metadata !"", metadata !9, i32 72, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null} ; [ DW_TAG_subroutine_type ]
!18 = metadata !{null, metadata !19, metadata !20}
!19 = metadata !{i32 524303, metadata !4, metadata !"", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 64, metadata !8} ; [ DW_TAG_pointer_type ]
!20 = metadata !{i32 524310, metadata !21, metadata !"ggBoolean", metadata !21, i32 478, i64 0, i64 0, i64 0, i32 0, metadata !22} ; [ DW_TAG_typedef ]
!21 = metadata !{i32 524329, metadata !"math.h", metadata !"/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm", metadata !5} ; [ DW_TAG_file_type ]
!22 = metadata !{i32 524324, metadata !4, metadata !"int", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!23 = metadata !{i32 524334, i32 0, metadata !8, metadata !"ggVector3", metadata !"ggVector3", metadata !"", metadata !9, i32 73, metadata !24, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!24 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, null} ; [ DW_TAG_subroutine_type ]
!25 = metadata !{null, metadata !19}
!26 = metadata !{i32 524334, i32 0, metadata !8, metadata !"ggVector3", metadata !"ggVector3", metadata !"", metadata !9, i32 74, metadata !27, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!27 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !28, i32 0, null} ; [ DW_TAG_subroutine_type ]
!28 = metadata !{null, metadata !19, metadata !13, metadata !13, metadata !13}
!29 = metadata !{i32 524334, i32 0, metadata !8, metadata !"Set", metadata !"Set", metadata !"_ZN9ggVector33SetEddd", metadata !9, i32 81, metadata !27, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!30 = metadata !{i32 524334, i32 0, metadata !8, metadata !"x", metadata !"x", metadata !"_ZNK9ggVector31xEv", metadata !9, i32 82, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!31 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !32, i32 0, null} ; [ DW_TAG_subroutine_type ]
!32 = metadata !{metadata !13, metadata !33}
!33 = metadata !{i32 524303, metadata !4, metadata !"", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 64, metadata !34} ; [ DW_TAG_pointer_type ]
!34 = metadata !{i32 524326, metadata !4, metadata !"", metadata !4, i32 0, i64 192, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ]
!35 = metadata !{i32 524334, i32 0, metadata !8, metadata !"y", metadata !"y", metadata !"_ZNK9ggVector31yEv", metadata !9, i32 83, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!36 = metadata !{i32 524334, i32 0, metadata !8, metadata !"z", metadata !"z", metadata !"_ZNK9ggVector31zEv", metadata !9, i32 84, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!37 = metadata !{i32 524334, i32 0, metadata !8, metadata !"x", metadata !"x", metadata !"_ZN9ggVector31xEv", metadata !9, i32 85, metadata !38, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!38 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !39, i32 0, null} ; [ DW_TAG_subroutine_type ]
!39 = metadata !{metadata !40, metadata !19}
!40 = metadata !{i32 524304, metadata !4, metadata !"double", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !13} ; [ DW_TAG_reference_type ]
!41 = metadata !{i32 524334, i32 0, metadata !8, metadata !"y", metadata !"y", metadata !"_ZN9ggVector31yEv", metadata !9, i32 86, metadata !38, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!42 = metadata !{i32 524334, i32 0, metadata !8, metadata !"z", metadata !"z", metadata !"_ZN9ggVector31zEv", metadata !9, i32 87, metadata !38, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!43 = metadata !{i32 524334, i32 0, metadata !8, metadata !"SetX", metadata !"SetX", metadata !"_ZN9ggVector34SetXEd", metadata !9, i32 88, metadata !44, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!44 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !45, i32 0, null} ; [ DW_TAG_subroutine_type ]
!45 = metadata !{null, metadata !19, metadata !13}
!46 = metadata !{i32 524334, i32 0, metadata !8, metadata !"SetY", metadata !"SetY", metadata !"_ZN9ggVector34SetYEd", metadata !9, i32 89, metadata !44, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!47 = metadata !{i32 524334, i32 0, metadata !8, metadata !"SetZ", metadata !"SetZ", metadata !"_ZN9ggVector34SetZEd", metadata !9, i32 90, metadata !44, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!48 = metadata !{i32 524334, i32 0, metadata !8, metadata !"ggVector3", metadata !"ggVector3", metadata !"", metadata !9, i32 92, metadata !49, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!49 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !50, i32 0, null} ; [ DW_TAG_subroutine_type ]
!50 = metadata !{null, metadata !19, metadata !51}
!51 = metadata !{i32 524304, metadata !4, metadata !"", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !34} ; [ DW_TAG_reference_type ]
!52 = metadata !{i32 524334, i32 0, metadata !8, metadata !"tolerance", metadata !"tolerance", metadata !"_ZNK9ggVector39toleranceEv", metadata !9, i32 100, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!53 = metadata !{i32 524334, i32 0, metadata !8, metadata !"tolerance", metadata !"tolerance", metadata !"_ZN9ggVector39toleranceEv", metadata !9, i32 101, metadata !38, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!54 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator+", metadata !"operator+", metadata !"_ZNK9ggVector3psEv", metadata !9, i32 107, metadata !55, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!55 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !56, i32 0, null} ; [ DW_TAG_subroutine_type ]
!56 = metadata !{metadata !51, metadata !33}
!57 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator-", metadata !"operator-", metadata !"_ZNK9ggVector3ngEv", metadata !9, i32 108, metadata !58, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!58 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !59, i32 0, null} ; [ DW_TAG_subroutine_type ]
!59 = metadata !{metadata !8, metadata !33}
!60 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator[]", metadata !"operator[]", metadata !"_ZNK9ggVector3ixEi", metadata !9, i32 290, metadata !61, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!61 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !62, i32 0, null} ; [ DW_TAG_subroutine_type ]
!62 = metadata !{metadata !13, metadata !33, metadata !22}
!63 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator[]", metadata !"operator[]", metadata !"_ZN9ggVector3ixEi", metadata !9, i32 278, metadata !64, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!64 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !65, i32 0, null} ; [ DW_TAG_subroutine_type ]
!65 = metadata !{metadata !40, metadata !19, metadata !22}
!66 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator+=", metadata !"operator+=", metadata !"_ZN9ggVector3pLERKS_", metadata !9, i32 303, metadata !67, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!67 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !68, i32 0, null} ; [ DW_TAG_subroutine_type ]
!68 = metadata !{metadata !69, metadata !19, metadata !51}
!69 = metadata !{i32 524304, metadata !4, metadata !"ggVector3", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_reference_type ]
!70 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator-=", metadata !"operator-=", metadata !"_ZN9ggVector3mIERKS_", metadata !9, i32 310, metadata !67, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!71 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator*=", metadata !"operator*=", metadata !"_ZN9ggVector3mLEd", metadata !9, i32 317, metadata !72, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!72 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !73, i32 0, null} ; [ DW_TAG_subroutine_type ]
!73 = metadata !{metadata !69, metadata !19, metadata !13}
!74 = metadata !{i32 524334, i32 0, metadata !8, metadata !"operator/=", metadata !"operator/=", metadata !"_ZN9ggVector3dVEd", metadata !9, i32 324, metadata !72, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!75 = metadata !{i32 524334, i32 0, metadata !8, metadata !"length", metadata !"length", metadata !"_ZNK9ggVector36lengthEv", metadata !9, i32 121, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!76 = metadata !{i32 524334, i32 0, metadata !8, metadata !"squaredLength", metadata !"squaredLength", metadata !"_ZNK9ggVector313squaredLengthEv", metadata !9, i32 122, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!77 = metadata !{i32 524334, i32 0, metadata !8, metadata !"MakeUnitVector", metadata !"MakeUnitVector", metadata !"_ZN9ggVector314MakeUnitVectorEv", metadata !9, i32 217, metadata !24, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!78 = metadata !{i32 524334, i32 0, metadata !8, metadata !"Perturb", metadata !"Perturb", metadata !"_ZNK9ggVector37PerturbEdd", metadata !9, i32 126, metadata !79, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!79 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !80, i32 0, null} ; [ DW_TAG_subroutine_type ]
!80 = metadata !{metadata !8, metadata !33, metadata !13, metadata !13}
!81 = metadata !{i32 524334, i32 0, metadata !8, metadata !"maxComponent", metadata !"maxComponent", metadata !"_ZNK9ggVector312maxComponentEv", metadata !9, i32 128, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!82 = metadata !{i32 524334, i32 0, metadata !8, metadata !"minComponent", metadata !"minComponent", metadata !"_ZNK9ggVector312minComponentEv", metadata !9, i32 129, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!83 = metadata !{i32 524334, i32 0, metadata !8, metadata !"maxAbsComponent", metadata !"maxAbsComponent", metadata !"_ZNK9ggVector315maxAbsComponentEv", metadata !9, i32 131, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!84 = metadata !{i32 524334, i32 0, metadata !8, metadata !"minAbsComponent", metadata !"minAbsComponent", metadata !"_ZNK9ggVector315minAbsComponentEv", metadata !9, i32 132, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!85 = metadata !{i32 524334, i32 0, metadata !8, metadata !"indexOfMinComponent", metadata !"indexOfMinComponent", metadata !"_ZNK9ggVector319indexOfMinComponentEv", metadata !9, i32 133, metadata !86, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!86 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !87, i32 0, null} ; [ DW_TAG_subroutine_type ]
!87 = metadata !{metadata !22, metadata !33}
!88 = metadata !{i32 524334, i32 0, metadata !8, metadata !"indexOfMinAbsComponent", metadata !"indexOfMinAbsComponent", metadata !"_ZNK9ggVector322indexOfMinAbsComponentEv", metadata !9, i32 137, metadata !86, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!89 = metadata !{i32 524334, i32 0, metadata !8, metadata !"indexOfMaxComponent", metadata !"indexOfMaxComponent", metadata !"_ZNK9ggVector319indexOfMaxComponentEv", metadata !9, i32 146, metadata !86, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!90 = metadata !{i32 524334, i32 0, metadata !8, metadata !"indexOfMaxAbsComponent", metadata !"indexOfMaxAbsComponent", metadata !"_ZNK9ggVector322indexOfMaxAbsComponentEv", metadata !9, i32 150, metadata !86, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!91 = metadata !{i32 524544, metadata !1, metadata !"vx", metadata !4, i32 46, metadata !13} ; [ DW_TAG_auto_variable ]
!92 = metadata !{i32 48, i32 0, metadata !1, null}
!93 = metadata !{i32 218, i32 0, metadata !94, metadata !96}
!94 = metadata !{i32 524299, metadata !95, i32 217, i32 0} ; [ DW_TAG_lexical_block ]
!95 = metadata !{i32 524299, metadata !77, i32 217, i32 0} ; [ DW_TAG_lexical_block ]
!96 = metadata !{i32 51, i32 0, metadata !1, null}
!97 = metadata !{i32 227, i32 0, metadata !94, metadata !96}
!98 = metadata !{i32 52, i32 0, metadata !1, null}
