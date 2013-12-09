; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | FileCheck --check-prefix=CHECK-DIS %s

; CHECK: 0x0000000b: DW_TAG_compile_unit
; CHECK:               DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000035] = "foo.cpp")
; CHECK: 0x{{[0-9a-f]+}}:   DW_TAG_class_type
; CHECK:                 DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]+}}] = "D")
; CHECK: 0x{{[0-9a-f]+}}:     DW_TAG_member
; CHECK:                   DW_AT_name [DW_FORM_strp]     ( .debug_str[0x{{[0-9a-f]+}}] = "c1")
; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x{{[0-9a-f]+}}] = "D")
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_artificial [DW_FORM_flag_present]       (true)

; CHECK-DIS: [artificial]

%class.D = type { i32, i32, i32, i32 }

@_ZN1DC1Ev = alias void (%class.D*)* @_ZN1DC2Ev
@_ZN1DC1ERKS_ = alias void (%class.D*, %class.D*)* @_ZN1DC2ERKS_

define void @_ZN1DC2Ev(%class.D* nocapture %this) unnamed_addr nounwind uwtable align 2 {
entry:
  tail call void @llvm.dbg.value(metadata !{%class.D* %this}, i64 0, metadata !29), !dbg !36
  %c1 = getelementptr inbounds %class.D* %this, i64 0, i32 0, !dbg !37
  store i32 1, i32* %c1, align 4, !dbg !37
  %c2 = getelementptr inbounds %class.D* %this, i64 0, i32 1, !dbg !42
  store i32 2, i32* %c2, align 4, !dbg !42
  %c3 = getelementptr inbounds %class.D* %this, i64 0, i32 2, !dbg !43
  store i32 3, i32* %c3, align 4, !dbg !43
  %c4 = getelementptr inbounds %class.D* %this, i64 0, i32 3, !dbg !44
  store i32 4, i32* %c4, align 4, !dbg !44
  ret void, !dbg !45
}

define void @_ZN1DC2ERKS_(%class.D* nocapture %this, %class.D* nocapture %d) unnamed_addr nounwind uwtable align 2 {
entry:
  tail call void @llvm.dbg.value(metadata !{%class.D* %this}, i64 0, metadata !34), !dbg !46
  tail call void @llvm.dbg.value(metadata !{%class.D* %d}, i64 0, metadata !35), !dbg !46
  %c1 = getelementptr inbounds %class.D* %d, i64 0, i32 0, !dbg !47
  %0 = load i32* %c1, align 4, !dbg !47
  %c12 = getelementptr inbounds %class.D* %this, i64 0, i32 0, !dbg !47
  store i32 %0, i32* %c12, align 4, !dbg !47
  %c2 = getelementptr inbounds %class.D* %d, i64 0, i32 1, !dbg !49
  %1 = load i32* %c2, align 4, !dbg !49
  %c23 = getelementptr inbounds %class.D* %this, i64 0, i32 1, !dbg !49
  store i32 %1, i32* %c23, align 4, !dbg !49
  %c3 = getelementptr inbounds %class.D* %d, i64 0, i32 2, !dbg !50
  %2 = load i32* %c3, align 4, !dbg !50
  %c34 = getelementptr inbounds %class.D* %this, i64 0, i32 2, !dbg !50
  store i32 %2, i32* %c34, align 4, !dbg !50
  %c4 = getelementptr inbounds %class.D* %d, i64 0, i32 3, !dbg !51
  %3 = load i32* %c4, align 4, !dbg !51
  %c45 = getelementptr inbounds %class.D* %this, i64 0, i32 3, !dbg !51
  store i32 %3, i32* %c45, align 4, !dbg !51
  ret void, !dbg !52
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!54}

!0 = metadata !{i32 786449, metadata !53, i32 4, metadata !"clang version 3.2 (trunk 167506) (llvm/trunk 167505)", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !31}
!5 = metadata !{i32 786478, metadata !6, null, metadata !"D", metadata !"D", metadata !"_ZN1DC2Ev", i32 12, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (%class.D*)* @_ZN1DC2Ev, null, metadata !17, metadata !27, i32 12} ; [ DW_TAG_subprogram ] [line 12] [def] [D]
!6 = metadata !{i32 786473, metadata !53} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from D]
!10 = metadata !{i32 786434, metadata !53, null, metadata !"D", i32 1, i64 128, i64 32, i32 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_class_type ] [D] [line 1, size 128, align 32, offset 0] [def] [from ]
!11 = metadata !{metadata !12, metadata !14, metadata !15, metadata !16, metadata !17, metadata !20}
!12 = metadata !{i32 786445, metadata !53, metadata !10, metadata !"c1", i32 6, i64 32, i64 32, i64 0, i32 1, metadata !13} ; [ DW_TAG_member ] [c1] [line 6, size 32, align 32, offset 0] [private] [from int]
!13 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{i32 786445, metadata !53, metadata !10, metadata !"c2", i32 7, i64 32, i64 32, i64 32, i32 1, metadata !13} ; [ DW_TAG_member ] [c2] [line 7, size 32, align 32, offset 32] [private] [from int]
!15 = metadata !{i32 786445, metadata !53, metadata !10, metadata !"c3", i32 8, i64 32, i64 32, i64 64, i32 1, metadata !13} ; [ DW_TAG_member ] [c3] [line 8, size 32, align 32, offset 64] [private] [from int]
!16 = metadata !{i32 786445, metadata !53, metadata !10, metadata !"c4", i32 9, i64 32, i64 32, i64 96, i32 1, metadata !13} ; [ DW_TAG_member ] [c4] [line 9, size 32, align 32, offset 96] [private] [from int]
!17 = metadata !{i32 786478, metadata !6, metadata !10, metadata !"D", metadata !"D", metadata !"", i32 3, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18, i32 3} ; [ DW_TAG_subprogram ] [line 3] [D]
!18 = metadata !{metadata !19}
!19 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!20 = metadata !{i32 786478, metadata !6, metadata !10, metadata !"D", metadata !"D", metadata !"", i32 4, metadata !21, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !25, i32 4} ; [ DW_TAG_subprogram ] [line 4] [D]
!21 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !22, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!22 = metadata !{null, metadata !9, metadata !23}
!23 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !24} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from D]
!25 = metadata !{metadata !26}
!26 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!27 = metadata !{metadata !28}
!28 = metadata !{metadata !29}
!29 = metadata !{i32 786689, metadata !5, metadata !"this", metadata !6, i32 16777228, metadata !30, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 12]
!30 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from D]
!31 = metadata !{i32 786478, metadata !6, null, metadata !"D", metadata !"D", metadata !"_ZN1DC2ERKS_", i32 19, metadata !21, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (%class.D*, %class.D*)* @_ZN1DC2ERKS_, null, metadata !20, metadata !32, i32 19} ; [ DW_TAG_subprogram ] [line 19] [def] [D]
!32 = metadata !{metadata !33}
!33 = metadata !{metadata !34, metadata !35}
!34 = metadata !{i32 786689, metadata !31, metadata !"this", metadata !6, i32 16777235, metadata !30, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 19]
!35 = metadata !{i32 786689, metadata !31, metadata !"d", metadata !6, i32 33554451, metadata !23, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [d] [line 19]
!36 = metadata !{i32 12, i32 0, metadata !5, null}
!37 = metadata !{i32 13, i32 0, metadata !38, null}
!38 = metadata !{i32 786443, metadata !5, i32 12, i32 0, metadata !6, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/foo.cpp]
!42 = metadata !{i32 14, i32 0, metadata !38, null}
!43 = metadata !{i32 15, i32 0, metadata !38, null}
!44 = metadata !{i32 16, i32 0, metadata !38, null}
!45 = metadata !{i32 17, i32 0, metadata !38, null}
!46 = metadata !{i32 19, i32 0, metadata !31, null}
!47 = metadata !{i32 20, i32 0, metadata !48, null}
!48 = metadata !{i32 786443, metadata !31, i32 19, i32 0, metadata !6, i32 1} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/foo.cpp]
!49 = metadata !{i32 21, i32 0, metadata !48, null}
!50 = metadata !{i32 22, i32 0, metadata !48, null}
!51 = metadata !{i32 23, i32 0, metadata !48, null}
!52 = metadata !{i32 24, i32 0, metadata !48, null}
!53 = metadata !{metadata !"foo.cpp", metadata !"/usr/local/google/home/echristo"}
!54 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
