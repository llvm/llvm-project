; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/2009-09-03-mdnode2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc

declare void @f() nounwind

define i32 @main(...) nounwind {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  call void @llvm.dbg.func.start(metadata !0)
  store i32 0, i32* %retval
  call void @llvm.dbg.stoppoint(i32 4, i32 5, metadata !1)
  call void @f()
  br label %return

return:                                           ; preds = %entry
  %0 = load i32* %retval                          ; <i32> [#uses=1]
  call void @llvm.dbg.stoppoint(i32 5, i32 1, metadata !1)
  call void @llvm.dbg.region.end(metadata !0)
  ret i32 %0
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!0 = metadata !{i32 458798, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 2, null, i1 false, i1 true}
!1 = metadata !{i32 458769, i32 0, i32 12, metadata !"a.c", metadata !"/home/rich/ellcc/test/source", metadata !"ellcc 0.1.0", i1 true, i1 true, metadata !"", i32 0}
