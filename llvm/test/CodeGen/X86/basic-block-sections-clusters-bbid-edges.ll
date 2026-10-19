; Tests that BBID matching only matches a block if its successors and
; predecessors in the Propeller CFG profile are also its successors and
; predecessors in the MIR. Profiles may omit edges without samples.
; REQUIRES: asserts
;
; Test 1: The profile omits the edges without samples (#0->#2 and #2->#3).
; All blocks match, and block #2 goes to the cold section.
; RUN: echo 'v1' > %t1
; RUN: echo 'f foo' >> %t1
; RUN: echo 'g 0:100,1:100 1:100,3:100 2:0 3:100' >> %t1
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 -basic-block-section-match-infer -propeller-match-bbid \
; RUN:   -debug-only=machine-block-match-infer 2>&1 | FileCheck %s --check-prefix=OMIT
; OMIT-NOT:    has profile edges that are not in the CFG
; OMIT:        .section        .text.split.foo,"ax",@progbits
; OMIT-NEXT:   foo.cold:
; OMIT-NEXT:   callq   baz@PLT
;
; Test 2: The profile has an edge #1->#2 that is not in the MIR, so blocks #1
; and #2 do not match and their weights are inferred.
; RUN: echo 'v1' > %t2
; RUN: echo 'f foo' >> %t2
; RUN: echo 'g 0:100,1:100,2:0 1:100,2:100 2:0,3:0 3:100' >> %t2
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 -basic-block-section-match-infer -propeller-match-bbid \
; RUN:   -debug-only=machine-block-match-infer 2>&1 | FileCheck %s --check-prefix=EXTRA
; EXTRA-NOT:   foo: bb.0 has profile edges that are not in the CFG
; EXTRA:       foo: bb.1 has profile edges that are not in the CFG
; EXTRA-NEXT:  foo: bb.2 has profile edges that are not in the CFG
; EXTRA-NOT:   foo: bb.3 has profile edges that are not in the CFG
;
; Test 3: The profile has block #3 and its clone #3.1 along the path #2->#3.
; The clone's edges are edges of block #3 in the MIR, so it matches #3, and
; the counts of both are added up. Block #1 goes to the cold section.
; RUN: echo 'v1' > %t3
; RUN: echo 'f foo' >> %t3
; RUN: echo 'g 0:100,1:0,2:100 1:0,3:0 2:100,3.1:100 3:0 3.1:100' >> %t3
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t3 -basic-block-section-match-infer -propeller-match-bbid \
; RUN:   -debug-only=machine-block-match-infer 2>&1 | FileCheck %s --check-prefix=CLONE
; CLONE-NOT:   has profile edges that are not in the CFG
; CLONE:       .section        .text.split.foo,"ax",@progbits
; CLONE-NEXT:  foo.cold:
; CLONE-NEXT:  callq   bar@PLT

define void @foo(i1 zeroext %0) nounwind {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, ptr %2, align 1
  %4 = load i8, ptr %2, align 1
  %5 = trunc i8 %4 to i1
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = call i32 @bar()
  br label %10

8:                                                ; preds = %1
  %9 = call i32 @baz()
  br label %10

10:                                               ; preds = %8, %6
  ret void
}

declare i32 @bar()
declare i32 @baz()
