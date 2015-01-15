; REQUIRES: object-emission

; RUN: %llc_dwarf -dwarf-version=2 -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=V2 %s
; RUN: %llc_dwarf -dwarf-version=3 -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=V3 %s

; CHECK: DW_AT_name {{.*}} "dst"
; V2: DW_AT_type {{.*}} {[[PTR:0x.*]]}
; V3: DW_AT_type {{.*}} {[[RESTRICT:0x.*]]}
; V3: [[RESTRICT]]: {{.*}}DW_TAG_restrict_type
; V3-NEXT: DW_AT_type {{.*}} {[[PTR:0x.*]]}
; CHECK: [[PTR]]: {{.*}}DW_TAG_pointer_type
; CHECK-NOT: DW_AT_type

; Generated with clang from:
; void foo(void* __restrict__ dst) {
; }


; Function Attrs: nounwind uwtable
define void @_Z3fooPv(i8* noalias %dst) #0 {
entry:
  %dst.addr = alloca i8*, align 8
  store i8* %dst, i8** %dst.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %dst.addr, metadata !13, metadata !{!"0x102"}), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/restrict.c] [DW_LANG_C_plus_plus]
!1 = !{!"restrict.c", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00_Z3fooPv\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void (i8*)* @_Z3fooPv, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/restrict.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0x37\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_restrict_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 1, !"Debug Info Version", i32 2}
!12 = !{!"clang version 3.5.0 "}
!13 = !{!"0x101\00dst\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [dst] [line 1]
!14 = !MDLocation(line: 1, scope: !4)
!15 = !MDLocation(line: 2, scope: !4)
