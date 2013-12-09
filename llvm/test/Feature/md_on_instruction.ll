; RUN: llvm-as < %s | llvm-dis | grep " !dbg " | count 4
define i32 @foo() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  call void @llvm.dbg.func.start(metadata !0)
  store i32 42, i32* %retval, !dbg !3
  br label %0, !dbg !3

; <label>:0                                       ; preds = %entry
  call void @llvm.dbg.region.end(metadata !0)
  %1 = load i32* %retval, !dbg !3                  ; <i32> [#uses=1]
  ret i32 %1, !dbg !3
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!llvm.module.flags = !{!6}

!0 = metadata !{i32 458798, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 1, metadata !2, i1 false, i1 true}
!1 = metadata !{i32 458769, metadata !4, i32 12, metadata !"clang 1.0", i1 true, metadata !"", i32 0, metadata !5, metadata !5, metadata !4, null, null, metadata !""}
!2 = metadata !{i32 458788, null, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!3 = metadata !{i32 1, i32 13, metadata !1, metadata !1}
!4 = metadata !{metadata !"foo.c", metadata !"/tmp"}
!5 = metadata !{i32 0}
!6 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
