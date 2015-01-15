%struct.Class = type { i8 }

define weak_odr i32 @_ZN5ClassIiE3fooEv(%struct.Class* %this) align 2 {
entry:
  %this.addr = alloca %struct.Class*, align 8
  store %struct.Class* %this, %struct.Class** %this.addr, align 8
  %this1 = load %struct.Class** %this.addr
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\004\00clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)\000\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2/t2.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"t2.cpp", !"/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\002\000\001\000\000\00256\000\002", !5, !6, !7, null, i32 (%struct.Class*)* @_ZN5ClassIiE3fooEv, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [foo]
!5 = !{!"../t.h", !"/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2"}
!6 = !{!"0x29", !5}    ; [ DW_TAG_file_type ] [/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2/../t.h]
!7 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 2}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)"}
!12 = !MDLocation(line: 2, column: 15, scope: !4)
