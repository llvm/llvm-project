; This test checks -print-after/before on loop passes
; Besides of the loop itself it should be dumping loop pre-header and exits.
;
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -loop-deletion -print-before=loop-deletion \
; RUN:	   | FileCheck %s -check-prefix=DEL
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes='loop(loop-deletion)' -print-before-all \
; RUN:	   | FileCheck %s -check-prefix=DEL
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -loop-unroll -print-after=loop-unroll -filter-print-funcs=bar \
; RUN:	   | FileCheck %s -check-prefix=BAR -check-prefix=BAR-OLD
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes='require<opt-remark-emit>,loop(unroll-full)' -print-after-all -filter-print-funcs=bar \
; RUN:	   | FileCheck %s -check-prefix=BAR
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -loop-unroll -print-after=loop-unroll -filter-print-funcs=foo -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=FOO-MODULE -check-prefix=FOO-MODULE-OLD
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes='require<opt-remark-emit>,loop(unroll-full)' -print-after-all -filter-print-funcs=foo -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=FOO-MODULE

; DEL:	    IR Dump Before {{Delete dead loops|LoopDeletionPass}}
; DEL: 	    ; Preheader:
; DEL-NEXT:  %idx = alloca i32, align 4
; DEL:      ; Loop:
; DEL-NEXT:  loop:
; DEL:	     cont:
; DEL:	    ; Exit blocks
; DEL:	     done:
; DEL:	    IR Dump Before {{Delete dead loops|LoopDeletionPass}}
; DEL: 	    ; Preheader:
; DEL-NEXT:  br label %loop
; DEL:      ; Loop:
; DEL-NEXT:  loop:
; DEL:	    ; Exit blocks
; DEL:	     end:

; BAR:	    IR Dump After {{Unroll|LoopFullUnrollPass}}
; BAR: 	    ; Preheader:
; BAR-NEXT:  br label %loop
; BAR:      ; Loop:
; BAR-NEXT:  loop:
; BAR:	    ; Exit blocks
; BAR:	     end:
; BAR-OLD-NOT: IR Dump
; BAR-OLD-NOT:  ; Loop

; FOO-MODULE: IR Dump After {{Unroll|LoopFullUnrollPass}}
; FOO-MODULE-SAME: loop: %loop
; FOO-MODULE-NEXT: ModuleID =
; FOO-MODULE: define void @foo
; FOO-MODULE: define void @bar
; FOO-MODULE-OLD-NOT: IR Dump

define void @foo(){
  %idx = alloca i32, align 4
  store i32 0, i32* %idx, align 4
  br label %loop

loop:
  %1 = load i32, i32* %idx, align 4
  %2 = icmp slt i32 %1, 10
  br i1 %2, label %cont, label %done

cont:
  %3 = load i32, i32* %idx, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %idx, align 4
  br label %loop

done:
  ret void
}

define void @bar(){
  br label %loop
loop:
  br i1 1, label %loop, label %end
end:
  ret void
}
