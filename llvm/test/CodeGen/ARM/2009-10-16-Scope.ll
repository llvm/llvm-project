; RUN: llc %s -O0 -o /dev/null -mtriple=arm-apple-darwin
; PR 5197
; There is not any llvm instruction assocated with !5. The code generator
; should be able to handle this.

define void @bar() nounwind ssp {
entry:
  %count_ = alloca i32, align 4                   ; <i32*> [#uses=2]
  br label %do.body, !dbg !0

do.body:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata !{i32* %count_}, metadata !4)
  %conv = ptrtoint i32* %count_ to i32, !dbg !0   ; <i32> [#uses=1]
  %call = call i32 @foo(i32 %conv) ssp, !dbg !0   ; <i32> [#uses=0]
  br label %do.end, !dbg !0

do.end:                                           ; preds = %do.body
  ret void, !dbg !7
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @foo(i32) ssp

!0 = metadata !{i32 5, i32 2, metadata !1, null}
!1 = metadata !{i32 458763, metadata !2, i32 1, i32 1}; [DW_TAG_lexical_block ]
!2 = metadata !{i32 458798, i32 0, metadata !3, metadata !"bar", metadata !"bar", metadata !"bar", metadata !3, i32 4, null, i1 false, i1 true}; [DW_TAG_subprogram ]
!3 = metadata !{i32 458769, i32 0, i32 12, metadata !"genmodes.i", metadata !"/Users/yash/Downloads", metadata !"clang 1.1", i1 true, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!4 = metadata !{i32 459008, metadata !5, metadata !"count_", metadata !3, i32 5, metadata !6}; [ DW_TAG_auto_variable ]
!5 = metadata !{i32 458763, metadata !1, i32 1, i32 1}; [DW_TAG_lexical_block ]
!6 = metadata !{i32 458788, metadata !3, metadata !"int", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
!7 = metadata !{i32 6, i32 1, metadata !2, null}
