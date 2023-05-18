; RUN: llc < %s -mtriple=i386 -mcpu=pentium4 | FileCheck %s
; RUN: llc < %s -mtriple=i386 -mcpu=pentium4m | FileCheck %s
; RUN: llc < %s -mtriple=i386 -mcpu=pentium-m | FileCheck %s
; RUN: llc < %s -mtriple=i386 -mcpu=prescott | FileCheck %s
; RUN: llc < %s -mtriple=i386 -mcpu=nocona | FileCheck %s
;
; Verify that scheduling puts some distance between a load feeding into
; the address of another load, and that second load.  This currently
; happens during the post-RA-scheduler, which should be enabled by
; default with the above specified cpus.

@ptrs = external dso_local global [0 x ptr], align 4
@idxa = common global i32 0, align 4
@idxb = common global i32 0, align 4
@res = common global i32 0, align 4

define void @addindirect() {
; CHECK-LABEL: addindirect:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    movl idxb, %ecx
; CHECK-NEXT:    movl idxa, %eax
; CHECK-NEXT:    movl ptrs(,%ecx,4), %ecx
; CHECK-NEXT:    movl ptrs(,%eax,4), %eax
; CHECK-NEXT:    movl (%ecx), %ecx
; CHECK-NEXT:    addl (%eax), %ecx
; CHECK-NEXT:    movl %ecx, res
; CHECK-NEXT:    retl
entry:
  %0 = load i32, ptr @idxa, align 4
  %arrayidx = getelementptr inbounds [0 x ptr], ptr @ptrs, i32 0, i32 %0
  %1 = load ptr, ptr %arrayidx, align 4
  %2 = load i32, ptr %1, align 4
  %3 = load i32, ptr @idxb, align 4
  %arrayidx1 = getelementptr inbounds [0 x ptr], ptr @ptrs, i32 0, i32 %3
  %4 = load ptr, ptr %arrayidx1, align 4
  %5 = load i32, ptr %4, align 4
  %add = add i32 %5, %2
  store i32 %add, ptr @res, align 4
  ret void
}
