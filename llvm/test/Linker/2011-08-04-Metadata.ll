; RUN: llvm-link %s %p/2011-08-04-Metadata2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if internal global variable's debug info is merged appropriately or not.

;CHECK:  metadata !{i32 589876, i32 0, metadata !{{[0-9]+}}, metadata !"x", metadata !"x", metadata !"", metadata !{{[0-9]+}}, i32 1, metadata !{{[0-9]+}}, i32 1, i32 1, i32* @x1}
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = internal global i32 0, align 4

define void @foo() nounwind uwtable ssp {
entry:
  store i32 1, i32* @x, align 4, !dbg !7
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1}
!llvm.dbg.gv = !{!5}

!0 = metadata !{i32 589841, i32 0, i32 12, metadata !"/tmp/one.c", metadata !"/Volumes/Lalgate/Slate/D", metadata !"clang version 3.0 ()", i1 true, i1 false, metadata !"", i32 0}
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"", metadata !2, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, void ()* @foo, null, null}
!2 = metadata !{i32 589865, metadata !"/tmp/one.c", metadata !"/Volumes/Lalgate/Slate/D", metadata !0}
!3 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0}
!4 = metadata !{null}
!5 = metadata !{i32 589876, i32 0, metadata !0, metadata !"x", metadata !"x", metadata !"", metadata !2, i32 2, metadata !6, i32 1, i32 1, i32* @x}
!6 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!7 = metadata !{i32 3, i32 14, metadata !8, null}
!8 = metadata !{i32 589835, metadata !1, i32 3, i32 12, metadata !2, i32 0}
