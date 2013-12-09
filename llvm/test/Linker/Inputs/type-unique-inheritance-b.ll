; ModuleID = 'bar.cpp'

%class.B = type { i32, %class.A* }
%class.A = type { %class.Base, i32 }
%class.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1gi(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %class.B, align 8
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !28), !dbg !29
  call void @llvm.dbg.declare(metadata !{%class.B* %t}, metadata !30), !dbg !31
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: ssp uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %class.A, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !33), !dbg !34
  call void @_Z1fi(i32 0), !dbg !35
  call void @_Z1gi(i32 1), !dbg !36
  ret i32 0, !dbg !37
}

declare void @_Z1fi(i32) #3

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27, !38}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (http://llvm.org/git/clang.git f54e02f969d02d640103db73efc30c45439fceab) (http://llvm.org/git/llvm.git 284353b55896cb1babfaa7add7c0a363245342d2)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !19, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/mren/c_testing/type_unique_air/inher/bar.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"bar.cpp", metadata !"/Users/mren/c_testing/type_unique_air/inher"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !11, metadata !15}
!4 = metadata !{i32 786434, metadata !5, null, metadata !"B", i32 7, i64 128, i64 64, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 7, size 128, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !"./b.hpp", metadata !"/Users/mren/c_testing/type_unique_air/inher"}
!6 = metadata !{metadata !7, metadata !9}
!7 = metadata !{i32 786445, metadata !5, metadata !"_ZTS1B", metadata !"bb", i32 8, i64 32, i64 32, i64 0, i32 1, metadata !8} ; [ DW_TAG_member ] [bb] [line 8, size 32, align 32, offset 0] [private] [from int]
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786445, metadata !5, metadata !"_ZTS1B", metadata !"a", i32 9, i64 64, i64 64, i64 64, i32 1, metadata !10} ; [ DW_TAG_member ] [a] [line 9, size 64, align 64, offset 64] [private] [from ]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!11 = metadata !{i32 786434, metadata !12, null, metadata !"A", i32 3, i64 64, i64 32, i32 0, i32 0, null, metadata !13, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 3, size 64, align 32, offset 0] [def] [from ]
!12 = metadata !{metadata !"./a.hpp", metadata !"/Users/mren/c_testing/type_unique_air/inher"}
!13 = metadata !{metadata !14, metadata !18}
!14 = metadata !{i32 786460, null, metadata !"_ZTS1A", null, i32 0, i64 0, i64 0, i64 0, i32 1, metadata !15} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [private] [from Base]
!15 = metadata !{i32 786434, metadata !5, null, metadata !"Base", i32 3, i64 32, i64 32, i32 0, i32 0, null, metadata !16, i32 0, null, null, metadata !"_ZTS4Base"} ; [ DW_TAG_class_type ] [Base] [line 3, size 32, align 32, offset 0] [def] [from ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786445, metadata !5, metadata !"_ZTS4Base", metadata !"b", i32 4, i64 32, i64 32, i64 0, i32 1, metadata !8} ; [ DW_TAG_member ] [b] [line 4, size 32, align 32, offset 0] [private] [from int]
!18 = metadata !{i32 786445, metadata !12, metadata !"_ZTS1A", metadata !"x", i32 4, i64 32, i64 32, i64 32, i32 1, metadata !8} ; [ DW_TAG_member ] [x] [line 4, size 32, align 32, offset 32] [private] [from int]
!19 = metadata !{metadata !20, metadata !24}
!20 = metadata !{i32 786478, metadata !1, metadata !21, metadata !"g", metadata !"g", metadata !"_Z1gi", i32 4, metadata !22, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @_Z1gi, null, null, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [g]
!21 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/Users/mren/c_testing/type_unique_air/inher/bar.cpp]
!22 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !23, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{null, metadata !8}
!24 = metadata !{i32 786478, metadata !1, metadata !21, metadata !"main", metadata !"main", metadata !"", i32 9, metadata !25, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [main]
!25 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !26, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = metadata !{metadata !8}
!27 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!28 = metadata !{i32 786689, metadata !20, metadata !"a", metadata !21, i32 16777220, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 4]
!29 = metadata !{i32 4, i32 0, metadata !20, null}
!30 = metadata !{i32 786688, metadata !20, metadata !"t", metadata !21, i32 5, metadata !4, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [t] [line 5]
!31 = metadata !{i32 5, i32 0, metadata !20, null}
!32 = metadata !{i32 6, i32 0, metadata !20, null}
!33 = metadata !{i32 786688, metadata !24, metadata !"a", metadata !21, i32 10, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 10]
!34 = metadata !{i32 10, i32 0, metadata !24, null}
!35 = metadata !{i32 11, i32 0, metadata !24, null}
!36 = metadata !{i32 12, i32 0, metadata !24, null}
!37 = metadata !{i32 13, i32 0, metadata !24, null}
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
