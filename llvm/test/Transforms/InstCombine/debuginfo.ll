; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1) nounwind readnone

declare i8* @foo(i8*, i32, i64, i64) nounwind

define hidden i8* @foobar(i8* %__dest, i32 %__val, i64 %__len) nounwind inlinehint ssp {
entry:
  %__dest.addr = alloca i8*, align 8
  %__val.addr = alloca i32, align 4
  %__len.addr = alloca i64, align 8
  store i8* %__dest, i8** %__dest.addr, align 8
; CHECK-NOT: call void @llvm.dbg.declare
; CHECK: call void @llvm.dbg.value
  call void @llvm.dbg.declare(metadata i8** %__dest.addr, metadata !0, metadata !{}), !dbg !16
  store i32 %__val, i32* %__val.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__val.addr, metadata !7, metadata !{}), !dbg !18
  store i64 %__len, i64* %__len.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %__len.addr, metadata !9, metadata !{}), !dbg !20
  %tmp = load i8** %__dest.addr, align 8, !dbg !21
  %tmp1 = load i32* %__val.addr, align 4, !dbg !21
  %tmp2 = load i64* %__len.addr, align 8, !dbg !21
  %tmp3 = load i8** %__dest.addr, align 8, !dbg !21
  %0 = call i64 @llvm.objectsize.i64.p0i8(i8* %tmp3, i1 false), !dbg !21
  %call = call i8* @foo(i8* %tmp, i32 %tmp1, i64 %tmp2, i64 %0), !dbg !21
  ret i8* %call, !dbg !21
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!30}

!0 = !{!"0x101\00__dest\0016777294\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00foobar\00foobar\00\0079\001\001\000\006\00256\001\0079", !27, !2, !4, null, i8* (i8*, i32, i64)* @foobar, null, null, !25} ; [ DW_TAG_subprogram ] [line 79] [local] [def] [foobar]
!2 = !{!"0x29", !27} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\0012\00clang version 3.0 (trunk 127710)\001\00\000\00\000", !28, !29, !29, !24, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !27, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6}
!6 = !{!"0xf\00\000\0064\0064\000\000", null, !3, null} ; [ DW_TAG_pointer_type ]
!7 = !{!"0x101\00__val\0033554510\000", !1, !2, !8} ; [ DW_TAG_arg_variable ]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !3} ; [ DW_TAG_base_type ]
!9 = !{!"0x101\00__len\0050331726\000", !1, !2, !10} ; [ DW_TAG_arg_variable ]
!10 = !{!"0x16\00size_t\0080\000\000\000\000", !27, !3, !11} ; [ DW_TAG_typedef ]
!11 = !{!"0x16\00__darwin_size_t\0090\000\000\000\000", !27, !3, !12} ; [ DW_TAG_typedef ]
!12 = !{!"0x24\00long unsigned int\000\0064\0064\000\000\007", null, !3} ; [ DW_TAG_base_type ]
!16 = !MDLocation(line: 78, column: 28, scope: !1)
!18 = !MDLocation(line: 78, column: 40, scope: !1)
!20 = !MDLocation(line: 78, column: 54, scope: !1)
!21 = !MDLocation(line: 80, column: 3, scope: !22)
!22 = !{!"0xb\0080\003\007", !27, !23} ; [ DW_TAG_lexical_block ]
!23 = !{!"0xb\0079\001\006", !27, !1} ; [ DW_TAG_lexical_block ]
!24 = !{!1}
!25 = !{!0, !7, !9}
!26 = !{!"0x29", !28} ; [ DW_TAG_file_type ]
!27 = !{!"string.h", !"Game"}
!28 = !{!"bits.c", !"Game"}
!29 = !{i32 0}
!30 = !{i32 1, !"Debug Info Version", i32 2}
