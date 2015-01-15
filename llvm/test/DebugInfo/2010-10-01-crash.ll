; RUN: llc -O0 %s -o /dev/null
; XFAIL: hexagon
; PR 8235

define void @CGRectStandardize(i32* sret %agg.result, i32* byval %rect) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata i32* %rect, metadata !23, metadata !{!"0x102"}), !dbg !24
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!27}
!0 = !{!"0x2e\00CGRectStandardize\00CGRectStandardize\00CGRectStandardize\0054\000\001\000\006\000\000\000", !1, null, null, null, void (i32*, i32*)* @CGRectStandardize, null, null, null} ; [ DW_TAG_subprogram ] [line 54] [def] [scope 0] [CGRectStandardize]
!1 = !{!"0x29", !25} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0016\00clang version 2.9 (trunk 115292)\001\00\001\00\000", !25, !26, !26, null, null, null} ; [ DW_TAG_compile_unit ]
!5 = !{!"0x16\00CGRect\0049\000\000\000\000", !25, null, null} ; [ DW_TAG_typedef ]
!23 = !{!"0x101\00rect\0053\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!24 = !MDLocation(line: 53, column: 33, scope: !0)
!25 = !{!"GSFusedSilica.m", !"/Volumes/Data/Users/sabre/Desktop"}
!26 = !{i32 0}
!27 = !{i32 1, !"Debug Info Version", i32 2}
