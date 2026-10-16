; Tests that BasicBlockMatchingAndInference matches the blocks of a Propeller
; CFG profile to machine basic blocks by their BB IDs, without BB hashes in the
; profile, and infers the hot and cold clusters from the matched weights.
;
; Test 1: Blocks #0, #1 and #3 are hot; block #2 goes to the cold section.
; RUN: echo 'v1' > %t1
; RUN: echo 'f foo' >> %t1
; RUN: echo 'g 0:100,1:100,2:0 1:100,3:100 2:0,3:0 3:100' >> %t1
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 -basic-block-section-match-infer -propeller-match-bbid | \
; RUN: FileCheck %s --check-prefixes=CHECK,BBID1
;
; Test 2: Blocks #0, #2 and #3 are hot; block #1 goes to the cold section.
; RUN: echo 'v1' > %t2
; RUN: echo 'f foo' >> %t2
; RUN: echo 'g 0:100,1:0,2:100 1:0,3:0 2:100,3:100 3:100' >> %t2
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 -basic-block-section-match-infer -propeller-match-bbid | \
; RUN: FileCheck %s --check-prefixes=CHECK,BBID2
;
; Test 3: With hash matching, the profile without BB hashes matches no blocks,
; so foo is not split.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 -basic-block-section-match-infer -propeller-match-bbid=false | \
; RUN: FileCheck %s --check-prefixes=CHECK,HASH
;
; Test 4: Blocks are matched by their base IDs, so the counts of clone #2.1 in
; the profile apply to block #2, giving the same clusters as Test 2.
; RUN: echo 'v1' > %t4
; RUN: echo 'f foo' >> %t4
; RUN: echo 'g 0:100,1:0,2.1:100 1:0,3:0 2.1:100,3:100 3:100' >> %t4
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t4 -basic-block-section-match-infer -propeller-match-bbid | \
; RUN: FileCheck %s --check-prefixes=CHECK,BBID2

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

; qux has no CFG profile, so it keeps its layout and is never split.
define void @qux(i1 %c) nounwind {
  br i1 %c, label %a, label %b
a:
  %r = call i32 @bar()
  br label %b
b:
  ret void
}

declare i32 @bar()
declare i32 @baz()

; CHECK:                .section        .text.foo,"ax",@progbits
; CHECK-LABEL:          foo:
; BBID1:                je      foo.cold
; BBID1-NEXT:           # %bb.1:
; BBID1-NEXT:           callq   bar@PLT
; BBID2:                jne     foo.cold
; BBID2-NEXT:           # %bb.2:
; BBID2-NEXT:           callq   baz@PLT
; BBID1-NEXT:           .LBB0_3:
; BBID2-NEXT:           .LBB0_3:
; BBID1:                .section        .text.split.foo,"ax",@progbits
; BBID2:                .section        .text.split.foo,"ax",@progbits
; BBID1-NEXT:           foo.cold:
; BBID2-NEXT:           foo.cold:
; BBID1-NEXT:           callq   baz@PLT
; BBID2-NEXT:           callq   bar@PLT
; HASH:                 je      .LBB0_2
; HASH:                 callq   bar@PLT
; HASH:                 callq   baz@PLT
; HASH-NOT:             .text.split.foo

; CHECK:                .section        .text.qux,"ax",@progbits
; CHECK-LABEL:          qux:
; CHECK-NOT:            .text.split.qux
