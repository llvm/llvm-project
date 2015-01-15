; RUN: llvm-link %s %p/DbgDeclare2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if metadata in dbg.declare is mapped properly or not.

; rdar://13089880
; CHECK: define i32 @main(i32 %argc, i8** %argv)
; CHECK: call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: define void @test(i32 %argc, i8** %argv)
; CHECK: call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata i32* %i, metadata !{{[0-9]+}}, metadata {{.*}})

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define i32 @main(i32 %argc, i8** %argv) uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !14, metadata !{!"0x102"}), !dbg !15
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !16, metadata !{!"0x102"}), !dbg !15
  %0 = load i32* %argc.addr, align 4, !dbg !17
  %1 = load i8*** %argv.addr, align 8, !dbg !17
  call void @test(i32 %0, i8** %1), !dbg !17
  ret i32 0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @test(i32, i8**)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 173515)\001\00\000\00\000", !20, !2, !2, !3, !2, null} ; [ DW_TAG_compile_unit ]
!1 = !{!2}
!2 = !{i32 0}
!3 = !{!5}
!5 = !{!"0x2e\00main\00main\00\003\000\001\000\006\00256\000\004", !20, null, !7, null, i32 (i32, i8**)* @main, null, null, !1} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0xf\00\000\0064\0064\000\000", null, null, !12} ; [ DW_TAG_pointer_type ]
!12 = !{!"0x26\00\000\000\000\000\000", null, null, !13} ; [ DW_TAG_const_type ]
!13 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!14 = !{!"0x101\00argc\0016777219\000", !5, !6, !9} ; [ DW_TAG_arg_variable ]
!15 = !MDLocation(line: 3, scope: !5)
!16 = !{!"0x101\00argv\0033554435\000", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!17 = !MDLocation(line: 5, scope: !18)
!18 = !{!"0xb\004\000\000", !20, !5} ; [ DW_TAG_lexical_block ]
!19 = !MDLocation(line: 6, scope: !18)
!20 = !{!"main.cpp", !"/private/tmp"}
!21 = !{i32 1, !"Debug Info Version", i32 2}
