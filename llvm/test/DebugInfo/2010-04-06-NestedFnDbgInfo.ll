; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj -o - < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Radar 7833483
; Do not emit a separate out-of-line definition DIE for the function-local 'foo'
; function (member of the function local 'A' type)
; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_class_type
; CHECK-NEXT: DW_AT_name {{.*}} "A"
; Check that the subprogram inside the class definition has low_pc, only
; attached to the definition.
; CHECK: [[FOO_INL:0x........]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_low_pc
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name {{.*}} "_ZZN1B2fnEvEN1A3fooEv"
; And just double check that there's no out of line definition that references
; this subprogram.
; CHECK-NOT: DW_AT_specification {{.*}} {[[FOO_INL]]}

%class.A = type { i8 }
%class.B = type { i8 }

define i32 @main() ssp {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=3]
  %b = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.A* %b, metadata !0, metadata !{!"0x102"}), !dbg !14
  %call = call i32 @_ZN1B2fnEv(%class.A* %b), !dbg !15 ; <i32> [#uses=1]
  store i32 %call, i32* %retval, !dbg !15
  %0 = load i32* %retval, !dbg !16                ; <i32> [#uses=1]
  ret i32 %0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr i32 @_ZN1B2fnEv(%class.A* %this) ssp align 2 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  %a = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  %i = alloca i32, align 4                        ; <i32*> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !17, metadata !{!"0x102"}), !dbg !18
  %this1 = load %class.A** %this.addr             ; <%class.A*> [#uses=0]
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !19, metadata !{!"0x102"}), !dbg !27
  call void @llvm.dbg.declare(metadata i32* %i, metadata !28, metadata !{!"0x102"}), !dbg !29
  %call = call i32 @_ZZN1B2fnEvEN1A3fooEv(%class.A* %a), !dbg !30 ; <i32> [#uses=1]
  store i32 %call, i32* %i, !dbg !30
  %tmp = load i32* %i, !dbg !31                   ; <i32> [#uses=1]
  store i32 %tmp, i32* %retval, !dbg !31
  %0 = load i32* %retval, !dbg !32                ; <i32> [#uses=1]
  ret i32 %0, !dbg !32
}

define internal i32 @_ZZN1B2fnEvEN1A3fooEv(%class.A* %this) ssp align 2 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !33, metadata !{!"0x102"}), !dbg !34
  %this1 = load %class.A** %this.addr             ; <%class.A*> [#uses=0]
  store i32 42, i32* %retval, !dbg !35
  %0 = load i32* %retval, !dbg !35                ; <i32> [#uses=1]
  ret i32 %0, !dbg !35
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!40}
!37 = !{!2, !10, !23}

!0 = !{!"0x100\00b\0016\000", !1, !3, !8} ; [ DW_TAG_auto_variable ]
!1 = !{!"0xb\0015\0012\000", !38, !2} ; [ DW_TAG_lexical_block ]
!2 = !{!"0x2e\00main\00main\00main\0015\000\001\000\006\000\000\0015", !38, !3, !5, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!3 = !{!"0x29", !38} ; [ DW_TAG_file_type ]
!4 = !{!"0x11\004\00clang 1.5\000\00\000\00\000", !38, !39, !39, !37, null,  null} ; [ DW_TAG_compile_unit ]
!5 = !{!"0x15\00\000\000\000\000\000\000", !38, !3, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = !{!7}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", !38, !3} ; [ DW_TAG_base_type ]
!8 = !{!"0x2\00B\002\008\008\000\000\000", !38, !3, null, !9, null, null, null} ; [ DW_TAG_class_type ] [B] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = !{!10}
!10 = !{!"0x2e\00fn\00fn\00_ZN1B2fnEv\004\000\001\000\006\000\000\004", !38, !8, !11, null, i32 (%class.A*)* @_ZN1B2fnEv, null, null, null} ; [ DW_TAG_subprogram ]
!11 = !{!"0x15\00\000\000\000\000\000\000", !38, !3, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!7, !13}
!13 = !{!"0xf\00\000\0064\0064\000\0064", !38, !3, !8} ; [ DW_TAG_pointer_type ]
!14 = !MDLocation(line: 16, column: 5, scope: !1)
!15 = !MDLocation(line: 17, column: 3, scope: !1)
!16 = !MDLocation(line: 18, column: 1, scope: !2)
!17 = !{!"0x101\00this\004\000", !10, !3, !13} ; [ DW_TAG_arg_variable ]
!18 = !MDLocation(line: 4, column: 7, scope: !10)
!19 = !{!"0x100\00a\009\000", !20, !3, !21} ; [ DW_TAG_auto_variable ]
!20 = !{!"0xb\004\0012\000", !38, !10} ; [ DW_TAG_lexical_block ]
!21 = !{!"0x2\00A\005\008\008\000\000\000", !38, !10, null, !22, null, null, null} ; [ DW_TAG_class_type ] [A] [line 5, size 8, align 8, offset 0] [def] [from ]
!22 = !{!23}
!23 = !{!"0x2e\00foo\00foo\00_ZZN1B2fnEvEN1A3fooEv\007\000\001\000\006\000\000\007", !38, !21, !24, null, i32 (%class.A*)* @_ZZN1B2fnEvEN1A3fooEv, null, null, null} ; [ DW_TAG_subprogram ]
!24 = !{!"0x15\00\000\000\000\000\000\000", !38, !3, null, !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = !{!7, !26}
!26 = !{!"0xf\00\000\0064\0064\000\0064", !38, !3, !21} ; [ DW_TAG_pointer_type ]
!27 = !MDLocation(line: 9, column: 7, scope: !20)
!28 = !{!"0x100\00i\0010\000", !20, !3, !7} ; [ DW_TAG_auto_variable ]
!29 = !MDLocation(line: 10, column: 9, scope: !20)
!30 = !MDLocation(line: 10, column: 5, scope: !20)
!31 = !MDLocation(line: 11, column: 5, scope: !20)
!32 = !MDLocation(line: 12, column: 3, scope: !10)
!33 = !{!"0x101\00this\007\000", !23, !3, !26} ; [ DW_TAG_arg_variable ]
!34 = !MDLocation(line: 7, column: 11, scope: !23)
!35 = !MDLocation(line: 7, column: 19, scope: !36)
!36 = !{!"0xb\007\0017\000", !38, !23} ; [ DW_TAG_lexical_block ]
!38 = !{!"one.cc", !"/tmp" }
!39 = !{i32 0}
!40 = !{i32 1, !"Debug Info Version", i32 2}
