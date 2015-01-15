; RUN: llc %s -O0 -o /dev/null -mtriple=arm-apple-darwin
; PR 5197
; There is not any llvm instruction assocated with !5. The code generator
; should be able to handle this.

define void @bar() nounwind ssp {
entry:
  %count_ = alloca i32, align 4                   ; <i32*> [#uses=2]
  br label %do.body, !dbg !0

do.body:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %count_, metadata !4, metadata !{!"0x102"})
  %conv = ptrtoint i32* %count_ to i32, !dbg !0   ; <i32> [#uses=1]
  %call = call i32 @foo(i32 %conv) ssp, !dbg !0   ; <i32> [#uses=0]
  br label %do.end, !dbg !0

do.end:                                           ; preds = %do.body
  ret void, !dbg !7
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @foo(i32) ssp

!0 = !MDLocation(line: 5, column: 2, scope: !1)
!1 = !{!"0xb\001\001\000", null, !2}; [DW_TAG_lexical_block ]
!2 = !{!"0x2e\00bar\00bar\00bar\004\000\001\000\006\000\000\000", i32 0, !3, null, null, null, null, null, null}; [DW_TAG_subprogram ]
!3 = !{!"0x11\0012\00clang 1.1\001\00\000\00\000", !8, null, !9, null, null, null}; [DW_TAG_compile_unit ]
!4 = !{!"0x100\00count_\005\000", !5, !3, !6}; [ DW_TAG_auto_variable ]
!5 = !{!"0xb\001\001\000", null, !1}; [DW_TAG_lexical_block ]
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !3}; [DW_TAG_base_type ]
!7 = !MDLocation(line: 6, column: 1, scope: !2)
!8 = !{!"genmodes.i", !"/Users/yash/Downloads"}
!9 = !{i32 0}
