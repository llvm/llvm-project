;; Tests for basic block sections applied on a function in a custom section.
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=all | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=all | FileCheck %s
; RUN: echo "!_Z3fooi" > %t1.list.txt
; RUN: echo "!!2" >> %t1.list.txt
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=%t1.list.txt | FileCheck %s --check-prefix=LIST1
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1.list.txt | FileCheck %s --check-prefix=LIST1
; RUN: echo "!_Z3fooi" > %t2.list.txt
; RUN: echo "!!0" >> %t2.list.txt
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=%t2.list.txt | FileCheck %s --check-prefix=LIST2
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2.list.txt | FileCheck %s --check-prefix=LIST2


; CHECK: .section	foo_section,"ax",@progbits,unique,1
; CHECK-LABEL: _Z3fooi:
; CHECK: .section	foo_section,"ax",@progbits,unique,2
; CHECK-NEXT: _Z3fooi.__part.1:
; CHECK: .section	foo_section,"ax",@progbits,unique,3
; CHECK-NEXT: _Z3fooi.__part.2:

; LIST1: .section	foo_section,"ax",@progbits,unique,1
; LIST1-LABEL: _Z3fooi:
; LIST1: .section	foo_section,"ax",@progbits,unique,2
; LIST1-NEXT: _Z3fooi.__part.0:
; LIST1-NOT: .section	foo_section,"ax",@progbits,unique,3

; LIST2: .section	foo_section,"ax",@progbits,unique,1
; LIST2-LABEL: _Z3fooi:
; LIST2: .section	foo_section,"ax",@progbits,unique,2
; LIST2-NEXT: _Z3fooi.cold:
; LIST2-NOT: .section	foo_section,"ax",@progbits,unique,3

;; Source to generate the IR:
;; #pragma clang section text = "foo_section"
;; int foo(int n) {
;;   if (n < 0)
;;     exit(-1);
;;   return 0;
;; }

define dso_local i32 @_Z3fooi(i32 %n) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @exit(i32 -1) #2
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

declare dso_local void @exit(i32) local_unnamed_addr

attributes #0 = {"implicit-section-name"="foo_section" }
