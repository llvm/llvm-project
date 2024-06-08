; RUN: llc < %s  -mtriple=i386-pc-unknown-linux-gnu -relocation-model=pic | FileCheck %s

@a0 = global i32 0, align 4

define x86_regcallcc void @tail_call_regcall(i32 %a) nounwind {
  tail call x86_regcallcc void @__regcall3__func(i32 %a) nounwind
  ret void
}

define internal x86_regcallcc void @__regcall3__func(i32 %i1) {
entry:
  store i32 %i1, ptr @a0, align 4
  ret void
}

;CHECK-LABEL: tail_call_regcall:
;CHECK:       # %bb.0:
;CHECK-NEXT:  jmp     __regcall3__func                # TAILCALL
;CHECK-NEXT:  .Lfunc_end0:

;CHECK-LABEL: __regcall3__func:
;CHECK:       addl    $_GLOBAL_OFFSET_TABLE_+({{.*}}), %ecx
;CHECK-NEXT:  movl    a0@GOT(%ecx), %ecx
;CHECK-NEXT:  movl    %eax, (%ecx)
;CHECK-NEXT:  retl
;CHECK-NEXT:  .Lfunc_end1:
