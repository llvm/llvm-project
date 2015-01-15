; ModuleID = 'bar.cpp'

%struct.Base = type { i32, %struct.Base* }

; Function Attrs: nounwind ssp uwtable
define void @_Z1gi(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %struct.Base, align 8
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !20, metadata !{!"0x102"}), !dbg !21
  call void @llvm.dbg.declare(metadata %struct.Base* %t, metadata !22, metadata !{!"0x102"}), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  call void @_Z1fi(i32 0), !dbg !25
  call void @_Z1gi(i32 1), !dbg !26
  ret i32 0, !dbg !27
}

declare void @_Z1fi(i32) #3

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !28}

!0 = !{!"0x11\004\00clang version 3.4 (http://llvm.org/git/clang.git 8a3f9e46cb988d2c664395b21910091e3730ae82) (http://llvm.org/git/llvm.git 4699e9549358bc77824a59114548eecc3f7c523c)\000\00\000\00\000", !1, !2, !3, !11, !2, !2} ; [ DW_TAG_compile_unit ] [bar.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"bar.cpp", !"."}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00Base\001\00128\0064\000\000\000", !5, null, null, !6, null, null, !"_ZTS4Base"} ; [ DW_TAG_structure_type ] [Base] [line 1, size 128, align 64, offset 0] [def] [from ]
!5 = !{!"./a.hpp", !"."}
!6 = !{!7, !9}
!7 = !{!"0xd\00a\002\0032\0032\000\000", !5, !"_ZTS4Base", !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0xd\00b\003\0064\0064\0064\000", !5, !"_ZTS4Base", !10} ; [ DW_TAG_member ] [b] [line 3, size 64, align 64, offset 64] [from ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS4Base"} ; [ DW_TAG_pointer_type ]
!11 = !{!12, !16}
!12 = !{!"0x2e\00g\00g\00_Z1gi\004\000\001\000\006\00256\000\004", !1, !13, !14, null, void (i32)* @_Z1gi, null, null, !2} ; [ DW_TAG_subprogram ] [line 4] [def] [g]
!13 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [bar.cpp]
!14 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{null, !8}
!16 = !{!"0x2e\00main\00main\00\007\000\001\000\006\00256\000\007", !1, !13, !17, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{!8}
!19 = !{i32 2, !"Dwarf Version", i32 2}
!20 = !{!"0x101\00a\0016777220\000", !12, !13, !8} ; [ DW_TAG_arg_variable ] [a] [line 4]
!21 = !MDLocation(line: 4, scope: !12)
!22 = !{!"0x100\00t\005\000", !12, !13, !4} ; [ DW_TAG_auto_variable ] [t] [line 5]
!23 = !MDLocation(line: 5, scope: !12)
!24 = !MDLocation(line: 6, scope: !12)
!25 = !MDLocation(line: 8, scope: !16)
!26 = !MDLocation(line: 9, scope: !16)
!27 = !MDLocation(line: 10, scope: !16)
!28 = !{i32 1, !"Debug Info Version", i32 2}
