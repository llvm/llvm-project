; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; rdar://13071590
; Check we are not emitting mutliple AT_const_value for a single member.
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_member
; CHECK: badbit
; CHECK: DW_AT_const_value [DW_FORM_sdata]      (1)
; CHECK-NOT: DW_AT_const_value
; CHECK: NULL

%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_os" }
%"class.std::basic_os" = type { %"class.std::os_base", %"class.std::basic_ostream"*, i8, i8 }
%"class.std::os_base" = type { i32 (...)**, i64, i64, i32, i32, i32 }

@_ZSt4cout = external global %"class.std::basic_ostream"
@.str = private unnamed_addr constant [6 x i8] c"c is \00", align 1

define i32 @main() {
entry:
  %call1.i = tail call %"class.std::basic_ostream"* @test(%"class.std::basic_ostream"* @_ZSt4cout, i8* getelementptr inbounds ([6 x i8]* @.str, i64 0, i64 0), i64 5)
  ret i32 0
}

declare %"class.std::basic_ostream"* @test(%"class.std::basic_ostream"*, i8*, i64)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1803}

!0 = metadata !{i32 786449, metadata !1802, i32 4, metadata !"clang version 3.3 (trunk 174207)", i1 true, metadata !"", i32 0, metadata !1, metadata !955, metadata !956, metadata !1786,  metadata !1786, metadata !""} ; [ DW_TAG_compile_unit ] [/privite/tmp/student2.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !26}
!4 = metadata !{i32 786489, null, metadata !"std", metadata !5, i32 48} ; [ DW_TAG_namespace ]
!5 = metadata !{i32 786473, metadata !1801} ; [ DW_TAG_file_type ]
!25 = metadata !{i32 786472, metadata !"_S_os_fmtflags_end", i64 65536} ; [ DW_TAG_enumerator ]
!26 = metadata !{i32 786436, metadata !1801, metadata !4, metadata !"_Ios_Iostate", i32 146, i64 32, i64 32, i32 0, i32 0, null, metadata !27, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [_Ios_Iostate] [line 146, size 32, align 32, offset 0] [def] [from ]
!27 = metadata !{metadata !28, metadata !29, metadata !30, metadata !31, metadata !32}
!28 = metadata !{i32 786472, metadata !"_S_goodbit", i64 0} ; [ DW_TAG_enumerator ] [_S_goodbit :: 0]
!29 = metadata !{i32 786472, metadata !"_S_badbit", i64 1} ; [ DW_TAG_enumerator ] [_S_badbit :: 1]
!30 = metadata !{i32 786472, metadata !"_S_eofbit", i64 2} ; [ DW_TAG_enumerator ] [_S_eofbit :: 2]
!31 = metadata !{i32 786472, metadata !"_S_failbit", i64 4} ; [ DW_TAG_enumerator ] [_S_failbit :: 4]
!32 = metadata !{i32 786472, metadata !"_S_os_ostate_end", i64 65536} ; [ DW_TAG_enumerator ] [_S_os_ostate_end :: 65536]
!49 = metadata !{i32 786434, metadata !1801, metadata !4, metadata !"os_base", i32 200, i64 1728, i64 64, i32 0, i32 0, null, metadata !50, i32 0, metadata !49, null, null} ; [ DW_TAG_class_type ] [os_base] [line 200, size 1728, align 64, offset 0] [def] [from ]
!50 = metadata !{metadata !77}
!54 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !55, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!55 = metadata !{metadata !56}
!56 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!77 = metadata !{i32 786445, metadata !1801, metadata !49, metadata !"badbit", i32 331, i64 0, i64 0, i64 0, i32 4096, metadata !78, i32 1} ; [ DW_TAG_member ]
!78 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !79} ; [ DW_TAG_const_type ]
!79 = metadata !{i32 786454, metadata !1801, metadata !49, metadata !"ostate", i32 327, i64 0, i64 0, i64 0, i32 0, metadata !26} ; [ DW_TAG_typedef ]
!955 = metadata !{i32 0}
!956 = metadata !{metadata !960}
!960 = metadata !{i32 786478, metadata !1802, null, metadata !"main", metadata !"main", metadata !"", i32 73, metadata !54, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* @main, null, null, metadata !955, i32 73} ; [ DW_TAG_subprogram ]
!961 = metadata !{i32 786473, metadata !1802} ; [ DW_TAG_file_type ]
!1786 = metadata !{metadata !1800}
!1800 = metadata !{i32 786484, i32 0, metadata !5, metadata !"badbit", metadata !"badbit", metadata !"badbit", metadata !5, i32 331, metadata !78, i32 1, i32 1, i32 1, metadata !77} ; [ DW_TAG_variable ]
!1801 = metadata !{metadata !"os_base.h", metadata !"/privite/tmp"}
!1802 = metadata !{metadata !"student2.cpp", metadata !"/privite/tmp"}
!1803 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
