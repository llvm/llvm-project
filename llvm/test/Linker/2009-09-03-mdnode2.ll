; This file is used by 2009-09-03-mdnode.ll, so it doesn't actually do anything itself
;
; RUN: true

define i32 @f(...) nounwind {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=1]
  call void @llvm.dbg.func.start(metadata !0)
  br label %return

return:                                           ; preds = %entry
  %0 = load i32* %retval                          ; <i32> [#uses=1]
  call void @llvm.dbg.stoppoint(i32 3, i32 1, metadata !1)
  call void @llvm.dbg.region.end(metadata !0)
  ret i32 %0
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!0 = metadata !{i32 458798, i32 0, metadata !1, metadata !"f", metadata !"f", metadata !"f", metadata !1, i32 1, null, i1 false, i1 true}
!1 = metadata !{i32 458769, i32 0, i32 12, metadata !"b.c", metadata !"/home/rich/ellcc/test/source", metadata !"ellcc 0.1.0", i1 true, i1 true, metadata !"", i32 0}
