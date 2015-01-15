; RUN: llc < %s -march=hexagon -mcpu=hexagonv4 -O2 -disable-lsr | FileCheck %s
; ModuleID = 'hwloop-dbg.o'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo(i32* nocapture %a, i32* nocapture %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata i32* %a, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !17
  tail call void @llvm.dbg.value(metadata i32* %b, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !19
  br label %for.body, !dbg !19

for.body:                                         ; preds = %for.body, %entry
; CHECK:     loop0(
; CHECK-NOT: add({{r[0-9]*}}, #
; CHECK:     endloop0
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %b.addr.01 = phi i32* [ %b, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32* %b.addr.01, i32 1, !dbg !21
  tail call void @llvm.dbg.value(metadata i32* %incdec.ptr, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !21
  %0 = load i32* %b.addr.01, align 4, !dbg !21
  store i32 %0, i32* %arrayidx.phi, align 4, !dbg !21
  %inc = add nsw i32 %i.02, 1, !dbg !26
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !26
  %exitcond = icmp eq i32 %inc, 10, !dbg !19
  %arrayidx.inc = getelementptr i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body, !dbg !19

for.end:                                          ; preds = %for.body
  ret void, !dbg !27
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = !{!"0x11\0012\00QuIC LLVM Hexagon Clang version 6.1-pre-unknown, (git://git-hexagon-aus.quicinc.com/llvm/clang-mainline.git e9382867661454cdf44addb39430741578e9765c) (llvm/llvm-mainline.git 36412bb1fcf03ed426d4437b41198bae066675ac)\001\00\000\00\001", !28, !2, !2, !3, !2, null} ; [ DW_TAG_compile_unit ] [/usr2/kparzysz/s.hex/t/hwloop-dbg.c] [DW_LANG_C99]
!2 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\001\001", !28, null, !7, null, void (i32*, i32*)* @foo, null, null, !11} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!6 = !{!"0x29", !28} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9, !9}
!9 = !{!"0xf\00\000\0032\0032\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from int]
!10 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = !{!13, !14, !15}
!13 = !{!"0x101\00a\0016777217\000", !5, !6, !9} ; [ DW_TAG_arg_variable ] [a] [line 1]
!14 = !{!"0x101\00b\0033554433\000", !5, !6, !9} ; [ DW_TAG_arg_variable ] [b] [line 1]
!15 = !{!"0x100\00i\002\000", !16, !6, !10} ; [ DW_TAG_auto_variable ] [i] [line 2]
!16 = !{!"0xb\001\0026\000", !28, !5} ; [ DW_TAG_lexical_block ] [/usr2/kparzysz/s.hex/t/hwloop-dbg.c]
!17 = !MDLocation(line: 1, column: 15, scope: !5)
!18 = !MDLocation(line: 1, column: 23, scope: !5)
!19 = !MDLocation(line: 3, column: 8, scope: !20)
!20 = !{!"0xb\003\003\001", !28, !16} ; [ DW_TAG_lexical_block ] [/usr2/kparzysz/s.hex/t/hwloop-dbg.c]
!21 = !MDLocation(line: 4, column: 5, scope: !22)
!22 = !{!"0xb\003\0028\002", !28, !20} ; [ DW_TAG_lexical_block ] [/usr2/kparzysz/s.hex/t/hwloop-dbg.c]
!26 = !MDLocation(line: 3, column: 23, scope: !20)
!27 = !MDLocation(line: 6, column: 1, scope: !16)
!28 = !{!"hwloop-dbg.c", !"/usr2/kparzysz/s.hex/t"}
!29 = !{i32 1, !"Debug Info Version", i32 2}
!30 = !{i32 0}
