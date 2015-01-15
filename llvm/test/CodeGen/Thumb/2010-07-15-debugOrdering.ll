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
  call void @llvm.dbg.value(metadata double %storemerge, i64 0, metadata !91, metadata !{!"0x102"}), !dbg !0
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare double @sqrt(double) nounwind readonly

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!104}
!0 = !MDLocation(line: 46, scope: !1)
!1 = !{!"0xb\0044\000\000", !101, !2} ; [ DW_TAG_lexical_block ]
!2 = !{!"0xb\0044\000\000", !101, !3} ; [ DW_TAG_lexical_block ]
!3 = !{!"0x2e\00getClosestDiagonal3\00getClosestDiagonal3\00_Z19getClosestDiagonal3ii\0044\000\001\000\006\000\000\000", !101, null, !6, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!4 = !{!"0x29", !101} ; [ DW_TAG_file_type ]
!5 = !{!"0x11\004\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 00)\001\00\000\00\000", !101, !102, !102, !103, null, null} ; [ DW_TAG_compile_unit ]
!6 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !22, !22}
!8 = !{!"0x13\00ggVector3\0066\00192\0032\000\000\000", !99, null, null, !10, null, null, null} ; [ DW_TAG_structure_type ] [ggVector3] [line 66, size 192, align 32, offset 0] [def] [from ]
!9 = !{!"0x29", !"ggVector3.h", !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src", !5} ; [ DW_TAG_file_type ]
!99 = !{!"ggVector3.h", !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src"}
!10 = !{!11, !16, !23, !26, !29, !30, !35, !36, !37, !41, !42, !43, !46, !47, !48, !52, !53, !54, !57, !60, !63, !66, !70, !71, !74, !75, !76, !77, !78, !81, !82, !83, !84, !85, !88, !89, !90}
!11 = !{!"0xd\00e\00160\00192\0032\000\000", !99, !8, !12} ; [ DW_TAG_member ]
!12 = !{!"0x1\00\000\00192\0032\000\000", !101, !4, !13, !14, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 192, align 32, offset 0] [from double]
!13 = !{!"0x24\00double\000\0064\0032\000\000\004", !101, !4} ; [ DW_TAG_base_type ]
!14 = !{!15}
!15 = !{!"0x21\000\003"}        ; [ DW_TAG_subrange_type ]
!16 = !{!"0x2e\00ggVector3\00ggVector3\00\0072\000\000\000\006\000\000\000", !9, !8, !17, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!17 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !19, !20}
!19 = !{!"0xf\00\000\0032\0032\000\0064", !101, !4, !8} ; [ DW_TAG_pointer_type ]
!20 = !{!"0x16\00ggBoolean\00478\000\000\000\000", !100, null, !22} ; [ DW_TAG_typedef ]
!21 = !{!"0x29", !"math.h", !"/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm", !5} ; [ DW_TAG_file_type ]
!100 = !{!"math.h", !"/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm"}
!22 = !{!"0x24\00int\000\0032\0032\000\000\005", !101, !4} ; [ DW_TAG_base_type ]
!23 = !{!"0x2e\00ggVector3\00ggVector3\00\0073\000\000\000\006\000\000\000", !9, !8, !24, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!24 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = !{null, !19}
!26 = !{!"0x2e\00ggVector3\00ggVector3\00\0074\000\000\000\006\000\000\000", !9, !8, !27, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!27 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !28, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = !{null, !19, !13, !13, !13}
!29 = !{!"0x2e\00Set\00Set\00_ZN9ggVector33SetEddd\0081\000\000\000\006\000\000\000", !9, !8, !27, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!30 = !{!"0x2e\00x\00x\00_ZNK9ggVector31xEv\0082\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!31 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !32, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!32 = !{!13, !33}
!33 = !{!"0xf\00\000\0032\0032\000\0064", !101, !4, !34} ; [ DW_TAG_pointer_type ]
!34 = !{!"0x26\00\000\00192\0032\000\000", !101, !4, !8} ; [ DW_TAG_const_type ]
!35 = !{!"0x2e\00y\00y\00_ZNK9ggVector31yEv\0083\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!36 = !{!"0x2e\00z\00z\00_ZNK9ggVector31zEv\0084\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!37 = !{!"0x2e\00x\00x\00_ZN9ggVector31xEv\0085\000\001\000\006\000\000\000", !9, !8, !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!38 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !39, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!39 = !{!40, !19}
!40 = !{!"0x10\00double\000\0032\0032\000\000", !101, !4, !13} ; [ DW_TAG_reference_type ]
!41 = !{!"0x2e\00y\00y\00_ZN9ggVector31yEv\0086\000\001\000\006\000\000\000", !9, !8, !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!42 = !{!"0x2e\00z\00z\00_ZN9ggVector31zEv\0087\000\001\000\006\000\000\000", !9, !8, !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!43 = !{!"0x2e\00SetX\00SetX\00_ZN9ggVector34SetXEd\0088\000\000\000\006\000\000\000", !9, !8, !44, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!44 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !45, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = !{null, !19, !13}
!46 = !{!"0x2e\00SetY\00SetY\00_ZN9ggVector34SetYEd\0089\000\000\000\006\000\000\000", !9, !8, !44, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!47 = !{!"0x2e\00SetZ\00SetZ\00_ZN9ggVector34SetZEd\0090\000\000\000\006\000\000\000", !9, !8, !44, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!48 = !{!"0x2e\00ggVector3\00ggVector3\00\0092\000\000\000\006\000\000\000", !9, !8, !49, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!49 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !50, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!50 = !{null, !19, !51}
!51 = !{!"0x10\00\000\0032\0032\000\000", !101, !4, !34} ; [ DW_TAG_reference_type ]
!52 = !{!"0x2e\00tolerance\00tolerance\00_ZNK9ggVector39toleranceEv\00100\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!53 = !{!"0x2e\00tolerance\00tolerance\00_ZN9ggVector39toleranceEv\00101\000\000\000\006\000\000\000", !9, !8, !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!54 = !{!"0x2e\00operator+\00operator+\00_ZNK9ggVector3psEv\00107\000\000\000\006\000\000\000", !9, !8, !55, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!55 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !56, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!56 = !{!51, !33}
!57 = !{!"0x2e\00operator-\00operator-\00_ZNK9ggVector3ngEv\00108\000\000\000\006\000\000\000", !9, !8, !58, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!58 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !59, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!59 = !{!8, !33}
!60 = !{!"0x2e\00operator[]\00operator[]\00_ZNK9ggVector3ixEi\00290\000\000\000\006\000\000\000", !9, !8, !61, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!61 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !62, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!62 = !{!13, !33, !22}
!63 = !{!"0x2e\00operator[]\00operator[]\00_ZN9ggVector3ixEi\00278\000\000\000\006\000\000\000", !9, !8, !64, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!64 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !65, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = !{!40, !19, !22}
!66 = !{!"0x2e\00operator+=\00operator+=\00_ZN9ggVector3pLERKS_\00303\000\000\000\006\000\000\000", !9, !8, !67, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!67 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !68, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!68 = !{!69, !19, !51}
!69 = !{!"0x10\00ggVector3\000\0032\0032\000\000", !101, !4, !8} ; [ DW_TAG_reference_type ]
!70 = !{!"0x2e\00operator-=\00operator-=\00_ZN9ggVector3mIERKS_\00310\000\000\000\006\000\000\000", !9, !8, !67, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!71 = !{!"0x2e\00operator*=\00operator*=\00_ZN9ggVector3mLEd\00317\000\000\000\006\000\000\000", !9, !8, !72, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!72 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !73, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!73 = !{!69, !19, !13}
!74 = !{!"0x2e\00operator/=\00operator/=\00_ZN9ggVector3dVEd\00324\000\000\000\006\000\000\000", !9, !8, !72, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!75 = !{!"0x2e\00length\00length\00_ZNK9ggVector36lengthEv\00121\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!76 = !{!"0x2e\00squaredLength\00squaredLength\00_ZNK9ggVector313squaredLengthEv\00122\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!77 = !{!"0x2e\00MakeUnitVector\00MakeUnitVector\00_ZN9ggVector314MakeUnitVectorEv\00217\000\001\000\006\000\000\000", !9, !8, !24, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!78 = !{!"0x2e\00Perturb\00Perturb\00_ZNK9ggVector37PerturbEdd\00126\000\000\000\006\000\000\000", !9, !8, !79, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!79 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !80, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!80 = !{!8, !33, !13, !13}
!81 = !{!"0x2e\00maxComponent\00maxComponent\00_ZNK9ggVector312maxComponentEv\00128\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!82 = !{!"0x2e\00minComponent\00minComponent\00_ZNK9ggVector312minComponentEv\00129\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!83 = !{!"0x2e\00maxAbsComponent\00maxAbsComponent\00_ZNK9ggVector315maxAbsComponentEv\00131\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!84 = !{!"0x2e\00minAbsComponent\00minAbsComponent\00_ZNK9ggVector315minAbsComponentEv\00132\000\000\000\006\000\000\000", !9, !8, !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!85 = !{!"0x2e\00indexOfMinComponent\00indexOfMinComponent\00_ZNK9ggVector319indexOfMinComponentEv\00133\000\000\000\006\000\000\000", !9, !8, !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!86 = !{!"0x15\00\000\000\000\000\000\000", !101, !4, null, !87, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!87 = !{!22, !33}
!88 = !{!"0x2e\00indexOfMinAbsComponent\00indexOfMinAbsComponent\00_ZNK9ggVector322indexOfMinAbsComponentEv\00137\000\000\000\006\000\000\000", !9, !8, !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!89 = !{!"0x2e\00indexOfMaxComponent\00indexOfMaxComponent\00_ZNK9ggVector319indexOfMaxComponentEv\00146\000\000\000\006\000\000\000", !9, !8, !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!90 = !{!"0x2e\00indexOfMaxAbsComponent\00indexOfMaxAbsComponent\00_ZNK9ggVector322indexOfMaxAbsComponentEv\00150\000\000\000\006\000\000\000", !9, !8, !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!91 = !{!"0x100\00vx\0046\000", !1, !4, !13} ; [ DW_TAG_auto_variable ]
!92 = !MDLocation(line: 48, scope: !1)
!93 = !MDLocation(line: 218, scope: !94, inlinedAt: !96)
!94 = !{!"0xb\00217\000\000", !101, !95} ; [ DW_TAG_lexical_block ]
!95 = !{!"0xb\00217\000\000", !101, !77} ; [ DW_TAG_lexical_block ]
!96 = !MDLocation(line: 51, scope: !1)
!97 = !MDLocation(line: 227, scope: !94, inlinedAt: !96)
!98 = !MDLocation(line: 52, scope: !1)
!101 = !{!"ggEdgeDiscrepancy.cc", !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src"}
!102 = !{i32 0}
!103 = !{!3, !77}
!104 = !{i32 1, !"Debug Info Version", i32 2}
