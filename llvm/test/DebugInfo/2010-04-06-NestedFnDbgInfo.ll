; RUN: llvm-as < %s | llc -asm-verbose -O0 | grep AT_specification | count 2
; Radar 7833483
; Do not emit AT_specification for nested function foo.

%class.A = type { i8 }
%class.B = type { i8 }

define i32 @main() ssp {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=3]
  %b = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{%class.A* %b}, metadata !0), !dbg !14
  %call = call i32 @_ZN1B2fnEv(%class.A* %b), !dbg !15 ; <i32> [#uses=1]
  store i32 %call, i32* %retval, !dbg !15
  %0 = load i32* %retval, !dbg !16                ; <i32> [#uses=1]
  ret i32 %0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define linkonce_odr i32 @_ZN1B2fnEv(%class.A* %this) ssp align 2 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  %a = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  %i = alloca i32, align 4                        ; <i32*> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !17), !dbg !18
  %this1 = load %class.A** %this.addr             ; <%class.A*> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !19), !dbg !27
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !28), !dbg !29
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
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !33), !dbg !34
  %this1 = load %class.A** %this.addr             ; <%class.A*> [#uses=0]
  store i32 42, i32* %retval, !dbg !35
  %0 = load i32* %retval, !dbg !35                ; <i32> [#uses=1]
  ret i32 %0, !dbg !35
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!40}
!37 = metadata !{metadata !2, metadata !10, metadata !23}

!0 = metadata !{i32 786688, metadata !1, metadata !"b", metadata !3, i32 16, metadata !8, i32 0, null} ; [ DW_TAG_auto_variable ]
!1 = metadata !{i32 786443, metadata !38, metadata !2, i32 15, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 786478, metadata !38, metadata !3, metadata !"main", metadata !"main", metadata !"main", i32 15, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i1 false, i32 0, i32 ()* @main, null, null, null, i32 15} ; [ DW_TAG_subprogram ]
!3 = metadata !{i32 786473, metadata !38} ; [ DW_TAG_file_type ]
!4 = metadata !{i32 786449, metadata !38, i32 4, metadata !"clang 1.5", i1 false, metadata !"", i32 0, metadata !39, metadata !39, metadata !37, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 786453, metadata !38, metadata !3, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786468, metadata !38, metadata !3, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 786434, metadata !38, metadata !3, metadata !"B", i32 2, i64 8, i64 8, i64 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_class_type ] [B] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786478, metadata !38, metadata !8, metadata !"fn", metadata !"fn", metadata !"_ZN1B2fnEv", i32 4, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i1 false, i32 0, i32 (%class.A*)* @_ZN1B2fnEv, null, null, null, i32 4} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 786453, metadata !38, metadata !3, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !7, metadata !13}
!13 = metadata !{i32 786447, metadata !38, metadata !3, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 64, metadata !8} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 16, i32 5, metadata !1, null}
!15 = metadata !{i32 17, i32 3, metadata !1, null}
!16 = metadata !{i32 18, i32 1, metadata !2, null}
!17 = metadata !{i32 786689, metadata !10, metadata !"this", metadata !3, i32 4, metadata !13, i32 0, null} ; [ DW_TAG_arg_variable ]
!18 = metadata !{i32 4, i32 7, metadata !10, null}
!19 = metadata !{i32 786688, metadata !20, metadata !"a", metadata !3, i32 9, metadata !21, i32 0, null} ; [ DW_TAG_auto_variable ]
!20 = metadata !{i32 786443, metadata !38, metadata !10, i32 4, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!21 = metadata !{i32 786434, metadata !38, metadata !10, metadata !"A", i32 5, i64 8, i64 8, i64 0, i32 0, null, metadata !22, i32 0, null, null, null} ; [ DW_TAG_class_type ] [A] [line 5, size 8, align 8, offset 0] [def] [from ]
!22 = metadata !{metadata !23}
!23 = metadata !{i32 786478, metadata !38, metadata !21, metadata !"foo", metadata !"foo", metadata !"_ZZN1B2fnEvEN1A3fooEv", i32 7, metadata !24, i1 false, i1 true, i32 0, i32 0, null, i1 false, i32 0, i32 (%class.A*)* @_ZZN1B2fnEvEN1A3fooEv, null, null, null, i32 7} ; [ DW_TAG_subprogram ]
!24 = metadata !{i32 786453, metadata !38, metadata !3, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{metadata !7, metadata !26}
!26 = metadata !{i32 786447, metadata !38, metadata !3, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 64, metadata !21} ; [ DW_TAG_pointer_type ]
!27 = metadata !{i32 9, i32 7, metadata !20, null}
!28 = metadata !{i32 786688, metadata !20, metadata !"i", metadata !3, i32 10, metadata !7, i32 0, null} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 10, i32 9, metadata !20, null}
!30 = metadata !{i32 10, i32 5, metadata !20, null}
!31 = metadata !{i32 11, i32 5, metadata !20, null}
!32 = metadata !{i32 12, i32 3, metadata !10, null}
!33 = metadata !{i32 786689, metadata !23, metadata !"this", metadata !3, i32 7, metadata !26, i32 0, null} ; [ DW_TAG_arg_variable ]
!34 = metadata !{i32 7, i32 11, metadata !23, null}
!35 = metadata !{i32 7, i32 19, metadata !36, null}
!36 = metadata !{i32 786443, metadata !38, metadata !23, i32 7, i32 17, i32 0} ; [ DW_TAG_lexical_block ]
!38 = metadata !{metadata !"one.cc", metadata !"/tmp" }
!39 = metadata !{i32 0}
!40 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
