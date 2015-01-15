; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; REQUIRES: object-emission
;
; Test debug info for variadic function arguments.
; Created from tools/clang/tests/CodeGenCXX/debug-info-varargs.cpp
;
; The ... parameter of variadic should be emitted as
; DW_TAG_unspecified_parameters.
;
; Normal variadic function.
; void b(int c, ...);
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "a"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "b"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; Variadic C++ member function.
; struct A { void a(int c, ...); }
;
; Variadic function pointer.
; void (*fptr)(int, ...);
;
; CHECK: DW_TAG_subroutine_type
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; ModuleID = 'llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp'

%struct.A = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1biz(i32 %c, ...) #0 {
  %1 = alloca i32, align 4
  %a = alloca %struct.A, align 1
  %fptr = alloca void (i32, ...)*, align 8
  store i32 %c, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !21, metadata !{!"0x102"}), !dbg !22
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !23, metadata !{!"0x102"}), !dbg !24
  call void @llvm.dbg.declare(metadata void (i32, ...)** %fptr, metadata !25, metadata !{!"0x102"}), !dbg !27
  store void (i32, ...)* @_Z1biz, void (i32, ...)** %fptr, align 8, !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = !{!"0x11\004\00clang version 3.5 \000\00\000\00\000", !1, !2, !3, !13, !2, !2} ; [ DW_TAG_compile_unit ] [llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp", !"radar/13690847"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00A\003\008\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0x2e\00a\00a\00_ZN1A1aEiz\006\000\000\000\006\00256\000\006", !1, !"_ZTS1A", !7, null, null, null, i32 0, !12} ; [ DW_TAG_subprogram ] [line 6] [a]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9, !10, null}
!9 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!10 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = !{i32 786468}
!13 = !{!14}
!14 = !{!"0x2e\00b\00b\00_Z1biz\0013\000\001\000\006\00256\000\0013", !1, !15, !16, null, void (i32, ...)* @_Z1biz, null, null, !2} ; [ DW_TAG_subprogram ] [line 13] [def] [b]
!15 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp]
!16 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = !{null, !10, null}
!18 = !{i32 2, !"Dwarf Version", i32 2}
!19 = !{i32 1, !"Debug Info Version", i32 2}
!20 = !{!"clang version 3.5 "}
!21 = !{!"0x101\00c\0016777229\000", !14, !15, !10} ; [ DW_TAG_arg_variable ] [c] [line 13]
!22 = !MDLocation(line: 13, scope: !14)
!23 = !{!"0x100\00a\0016\000", !14, !15, !4} ; [ DW_TAG_auto_variable ] [a] [line 16]
!24 = !MDLocation(line: 16, scope: !14)
!25 = !{!"0x100\00fptr\0018\000", !14, !15, !26} ; [ DW_TAG_auto_variable ] [fptr] [line 18]
!26 = !{!"0xf\00\000\0064\0064\000\000", null, null, !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!27 = !MDLocation(line: 18, scope: !14)
!28 = !MDLocation(line: 22, scope: !14)
