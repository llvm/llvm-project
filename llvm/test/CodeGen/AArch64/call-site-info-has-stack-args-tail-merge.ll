; RUN: llc -mtriple=aarch64-linux-gnu -emit-codegen-call-site-info < %s -stop-after=branch-folder -o - | FileCheck %s

; Tail merging can fold two identical-looking call instructions from different
; call sites into one. The surviving call then stands for both call sites, so
; its HasStackArguments info must be the conservative merge of both entries.

declare void @vararg(i32, ...)

; Conflicting entries merge to unknown.
; CHECK-LABEL: name: merge_conflicting
; CHECK: callSites:
; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
; CHECK: BL @vararg
; CHECK-NOT: BL @vararg
define void @merge_conflicting(i1 %c) {
entry:
  br i1 %c, label %with_stack, label %without_stack

with_stack:
  call void (i32, ...) @vararg(i32 1, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  br label %done

without_stack:
  call void (i32, ...) @vararg(i32 1, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7)
  br label %done

done:
  ret void
}

; Agreeing entries survive the merge: no + no stays no.
; CHECK-LABEL: name: merge_agreeing_no_stack
; CHECK: callSites:
; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [], hasStackArguments:
; CHECK-NEXT: no }
; CHECK: BL @vararg
; CHECK-NOT: BL @vararg
define void @merge_agreeing_no_stack(i1 %c, i32 %a, i32 %b) {
entry:
  br i1 %c, label %left, label %right

left:
  call void (i32, ...) @vararg(i32 %a, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7)
  br label %done

right:
  call void (i32, ...) @vararg(i32 %b, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7)
  br label %done

done:
  ret void
}

; Agreeing entries survive the merge: yes + yes stays yes.
; CHECK-LABEL: name: merge_agreeing_on_stack
; CHECK: callSites:
; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [], hasStackArguments:
; CHECK-NEXT: yes }
; CHECK: BL @vararg
; CHECK-NOT: BL @vararg
define void @merge_agreeing_on_stack(i1 %c, i64 %a, i64 %b) {
entry:
  br i1 %c, label %left, label %right

left:
  call void (i32, ...) @vararg(i32 1, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 %a)
  br label %done

right:
  call void (i32, ...) @vararg(i32 1, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 %b)
  br label %done

done:
  ret void
}
