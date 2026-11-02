; RUN: llc %s --mtriple=aarch64-pc-windows-msvc -o - | FileCheck %s

; Regression test: a stack object inside the callee-saved register area must
; not be addressed via the base pointer, since its distance from the base
; pointer varies with the stack realignment padding.

; CHECK-LABEL: "?repro@@YAHH@Z":
; CHECK:       str x19, [sp, #-64]!
; CHECK:       add x29, sp, #8
; CHECK:       sub x[[TMP:[0-9]+]], sp, #64
; CHECK:       and sp, x[[TMP]], #0xffffffffffffffc0
; CHECK:       mov x19, sp

; This spill must be x29-relative, not reached through the base pointer, whose
; distance to it varies with the realignment padding.
; CHECK:       str wzr, [x29, #20]
; CHECK-NOT:   [x19, #92]


define i32 @"?repro@@YAHH@Z"(i32 %common.ret.op1) personality ptr @__CxxFrameHandler3 {
entry:
  %b.sroa.0 = alloca [16 x i32], align 64
  %e = alloca ptr, align 8
  %e156 = alloca ptr, align 8
  %e162 = alloca ptr, align 8
  call void @llvm.memset.p0.i64(ptr %b.sroa.0, i8 0, i64 0, i1 false)
  invoke void @"?thrower@@YAXH@Z"(i32 0)
          to label %common.ret unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch173, label %catch161, label %catch155, label %catch] unwind to caller

catch173:                                         ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 0, ptr null]
  catchret from %1 to label %common.ret

catch161:                                         ; preds = %catch.dispatch
  %2 = catchpad within %0 [ptr null, i32 0, ptr %e162]
  catchret from %2 to label %common.ret

common.ret:                                       ; preds = %catch, %catch155, %catch161, %catch173, %entry
  %common.ret.op11 = phi i32 [ 0, %catch155 ], [ 1, %catch ], [ 0, %catch161 ], [ 0, %catch173 ], [ 0, %entry ]
  ret i32 %common.ret.op11

catch155:                                         ; preds = %catch.dispatch
  %3 = catchpad within %0 [ptr null, i32 0, ptr %e156]
  br label %common.ret

catch:                                            ; preds = %catch.dispatch
  %4 = catchpad within %0 [ptr null, i32 0, ptr %e]
  catchret from %4 to label %common.ret
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #0

declare i32 @__CxxFrameHandler3(...)

declare void @"?thrower@@YAXH@Z"(i32)

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: write) }
