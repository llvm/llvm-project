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
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !28, metadata !{!"0x102"}), !dbg !29
  call void @llvm.dbg.declare(metadata %class.B* %t, metadata !30, metadata !{!"0x102"}), !dbg !31
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %class.A, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !33, metadata !{!"0x102"}), !dbg !34
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

!0 = !{!"0x11\004\00clang version 3.4 (http://llvm.org/git/clang.git f54e02f969d02d640103db73efc30c45439fceab) (http://llvm.org/git/llvm.git 284353b55896cb1babfaa7add7c0a363245342d2)\000\00\000\00\000", !1, !2, !3, !19, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/mren/c_testing/type_unique_air/inher/bar.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"bar.cpp", !"/Users/mren/c_testing/type_unique_air/inher"}
!2 = !{i32 0}
!3 = !{!4, !11, !15}
!4 = !{!"0x2\00B\007\00128\0064\000\000\000", !5, null, null, !6, null, null, !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 7, size 128, align 64, offset 0] [def] [from ]
!5 = !{!"./b.hpp", !"/Users/mren/c_testing/type_unique_air/inher"}
!6 = !{!7, !9}
!7 = !{!"0xd\00bb\008\0032\0032\000\001", !5, !"_ZTS1B", !8} ; [ DW_TAG_member ] [bb] [line 8, size 32, align 32, offset 0] [private] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0xd\00a\009\0064\0064\0064\001", !5, !"_ZTS1B", !10} ; [ DW_TAG_member ] [a] [line 9, size 64, align 64, offset 64] [private] [from ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!11 = !{!"0x2\00A\003\0064\0032\000\000\000", !12, null, null, !13, null, null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 3, size 64, align 32, offset 0] [def] [from ]
!12 = !{!"./a.hpp", !"/Users/mren/c_testing/type_unique_air/inher"}
!13 = !{!14, !18}
!14 = !{!"0x1c\00\000\000\000\000\001", null, !"_ZTS1A", !15} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [private] [from Base]
!15 = !{!"0x2\00Base\003\0032\0032\000\000\000", !5, null, null, !16, null, null, !"_ZTS4Base"} ; [ DW_TAG_class_type ] [Base] [line 3, size 32, align 32, offset 0] [def] [from ]
!16 = !{!17}
!17 = !{!"0xd\00b\004\0032\0032\000\001", !5, !"_ZTS4Base", !8} ; [ DW_TAG_member ] [b] [line 4, size 32, align 32, offset 0] [private] [from int]
!18 = !{!"0xd\00x\004\0032\0032\0032\001", !12, !"_ZTS1A", !8} ; [ DW_TAG_member ] [x] [line 4, size 32, align 32, offset 32] [private] [from int]
!19 = !{!20, !24}
!20 = !{!"0x2e\00g\00g\00_Z1gi\004\000\001\000\006\00256\000\004", !1, !21, !22, null, void (i32)* @_Z1gi, null, null, !2} ; [ DW_TAG_subprogram ] [line 4] [def] [g]
!21 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/Users/mren/c_testing/type_unique_air/inher/bar.cpp]
!22 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{null, !8}
!24 = !{!"0x2e\00main\00main\00\009\000\001\000\006\00256\000\009", !1, !21, !25, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 9] [def] [main]
!25 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !26, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = !{!8}
!27 = !{i32 2, !"Dwarf Version", i32 2}
!28 = !{!"0x101\00a\0016777220\000", !20, !21, !8} ; [ DW_TAG_arg_variable ] [a] [line 4]
!29 = !MDLocation(line: 4, scope: !20)
!30 = !{!"0x100\00t\005\000", !20, !21, !4} ; [ DW_TAG_auto_variable ] [t] [line 5]
!31 = !MDLocation(line: 5, scope: !20)
!32 = !MDLocation(line: 6, scope: !20)
!33 = !{!"0x100\00a\0010\000", !24, !21, !11} ; [ DW_TAG_auto_variable ] [a] [line 10]
!34 = !MDLocation(line: 10, scope: !24)
!35 = !MDLocation(line: 11, scope: !24)
!36 = !MDLocation(line: 12, scope: !24)
!37 = !MDLocation(line: 13, scope: !24)
!38 = !{i32 1, !"Debug Info Version", i32 2}
