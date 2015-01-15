; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; From source:
; struct foo {
;   int __attribute__((always_inline)) func(int x) { return x + 2; }
; };

; int i;

; int main() {
;   return foo().func(i);
; }

; CHECK: DW_TAG_structure_type
; CHECK:   DW_TAG_subprogram

; But make sure we emit DW_AT_object_pointer on the abstract definition.
; CHECK: [[ABSTRACT_ORIGIN:.*]]: DW_TAG_subprogram
; CHECK-NOT: {{NULL|TAG}}
; CHECK: DW_AT_specification {{.*}} "_ZN3foo4funcEi"
; CHECK-NOT: {{NULL|TAG}}
; CHECK: DW_AT_object_pointer

; Ensure we omit DW_AT_object_pointer on inlined subroutines.
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} {[[ABSTRACT_ORIGIN]]} "_ZN3foo4funcEi"
; CHECK-NOT: NULL
; CHECK-NOT: DW_AT_object_pointer
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_AT_artificial
; CHECK: DW_TAG

%struct.foo = type { i8 }

@i = global i32 0, align 4

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %this.addr.i = alloca %struct.foo*, align 8
  %x.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  %tmp = alloca %struct.foo, align 1
  store i32 0, i32* %retval
  %0 = load i32* @i, align 4, !dbg !23
  store %struct.foo* %tmp, %struct.foo** %this.addr.i, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr.i, metadata !24, metadata !{!"0x102"}), !dbg !26
  store i32 %0, i32* %x.addr.i, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr.i, metadata !27, metadata !{!"0x102"}), !dbg !28
  %this1.i = load %struct.foo** %this.addr.i
  %1 = load i32* %x.addr.i, align 4, !dbg !28
  %add.i = add nsw i32 %1, 2, !dbg !28
  ret i32 %add.i, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !3, !12, !18, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/inline.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"inline.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00foo\001\008\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0x2e\00func\00func\00_ZN3foo4funcEi\002\000\000\000\006\00256\000\002", !1, !"_ZTS3foo", !7, null, null, null, i32 0, !11} ; [ DW_TAG_subprogram ] [line 2] [func]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !10, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
!11 = !{i32 786468}
!12 = !{!13, !17}
!13 = !{!"0x2e\00main\00main\00\007\000\001\000\006\00256\000\007", !1, !14, !15, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!14 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/inline.cpp]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{!9}
!17 = !{!"0x2e\00func\00func\00_ZN3foo4funcEi\002\000\001\000\006\00256\000\002", !1, !"_ZTS3foo", !7, null, null, null, !6, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [func]
!18 = !{!19}
!19 = !{!"0x34\00i\00i\00\005\000\001", null, !14, !9, i32* @i, null} ; [ DW_TAG_variable ] [i] [line 5] [def]
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 1, !"Debug Info Version", i32 2}
!22 = !{!"clang version 3.5.0 "}
!23 = !MDLocation(line: 8, scope: !13)
!24 = !{!"0x101\00this\0016777216\001088", !17, null, !25} ; [ DW_TAG_arg_variable ] [this] [line 0]
!25 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3foo]
!26 = !MDLocation(line: 0, scope: !17, inlinedAt: !23)
!27 = !{!"0x101\00x\0033554434\000", !17, !14, !9} ; [ DW_TAG_arg_variable ] [x] [line 2]
!28 = !MDLocation(line: 2, scope: !17, inlinedAt: !23)
