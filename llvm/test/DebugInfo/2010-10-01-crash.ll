; RUN: llc -O0 %s -o /dev/null
; XFAIL: hexagon
; PR 8235

define void @CGRectStandardize(i32* sret %agg.result, i32* byval %rect) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !{i32* %rect}, metadata !23), !dbg !24
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!27}
!0 = metadata !{i32 589870, metadata !1, null, metadata !"CGRectStandardize", metadata !"CGRectStandardize", metadata !"CGRectStandardize", i32 54, null, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void (i32*, i32*)* @CGRectStandardize, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 54] [def] [scope 0] [CGRectStandardize]
!1 = metadata !{i32 589865, metadata !25}
!2 = metadata !{i32 589841, metadata !25, i32 16, metadata !"clang version 2.9 (trunk 115292)", i1 true, metadata !"", i32 1, metadata !26, metadata !26, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 589846, metadata !25, null, metadata !"CGRect", i32 49, i64 0, i64 0, i64 0, i32 0, null}
!23 = metadata !{i32 590081, metadata !0, metadata !"rect", metadata !1, i32 53, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!24 = metadata !{i32 53, i32 33, metadata !0, null}
!25 = metadata !{metadata !"GSFusedSilica.m", metadata !"/Volumes/Data/Users/sabre/Desktop"}
!26 = metadata !{i32 0}
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
