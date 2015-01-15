; RUN: llvm-link %s %p/2011-08-18-unique-class-type2.ll -S -o - | FileCheck %s
; CHECK: DW_TAG_class_type
; CHECK-NOT: DW_TAG_class_type
; Test to check there is only one MDNode for class A after linking.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%"class.N1::A" = type { i8 }

define void @_Z3fooN2N11AE() nounwind uwtable ssp {
entry:
  %mya = alloca %"class.N1::A", align 1
  call void @llvm.dbg.declare(metadata %"class.N1::A"* %mya, metadata !9, metadata !{!"0x102"}), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = !{!"0x11\004\00clang version 3.0 (trunk 137954)\001\00\000\00\000", !16, !2, !2, !3, !2, null} ; [ DW_TAG_compile_unit ]
!1 = !{!2}
!2 = !{i32 0}
!3 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00_Z3fooN2N11AE\004\000\001\000\006\00256\000\000", !16, !6, !7, null, void ()* @_Z3fooN2N11AE, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [foo]
!6 = !{!"0x29", !16} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", !16, !6, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!9 = !{!"0x101\00mya\0016777220\000", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!10 = !{!"0x2\00A\003\008\008\000\000\000", !17, !11, null, !2, null, null, null} ; [ DW_TAG_class_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!11 = !{!"0x39\00N1\002", !17, null} ; [ DW_TAG_namespace ]
!12 = !{!"0x29", !17} ; [ DW_TAG_file_type ]
!13 = !MDLocation(line: 4, column: 12, scope: !5)
!14 = !MDLocation(line: 4, column: 18, scope: !15)
!15 = !{!"0xb\004\0017\000", !16, !5} ; [ DW_TAG_lexical_block ]
!16 = !{!"n1.c", !"/private/tmp"}
!17 = !{!"./n.h", !"/private/tmp"}
!18 = !{i32 1, !"Debug Info Version", i32 2}
