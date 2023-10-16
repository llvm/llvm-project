;; Tests for path cloning with -basic-block-sections.

declare void @effect(i32 zeroext)

;; Test valid application of path cloning.
; RUN: echo 'v1' > %t1
; RUN: echo 'f foo' >> %t1
; RUN: echo 'p 0 3 5' >> %t1
; RUN: echo 'c 0 3.1 5.1 1 2 3 4 5' >> %t1
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t1 | FileCheck %s --check-prefixes=PATH1,FOOSECTIONS,FOOCLONE
; RUN: echo 'v1' > %t2
; RUN: echo 'f foo' >> %t2
; RUN: echo 'p 0 3 5' >> %t2
; RUN: echo 'p 1 3 4 5' >> %t2
; RUN: echo 'c 0 3.1 5.1' >> %t2
; RUN: echo 'c 1 3.2 4.1 5.2 2 3 4 5' >> %t2
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t2 | FileCheck %s --check-prefixes=PATH2,FOOSECTIONS,FOOCLONE

;; Test failed application of path cloning.
; RUN: echo 'v1' > %t3
; RUN: echo 'f foo' >> %t3
; RUN: echo 'p 0 2 3' >> %t3
; RUN: echo 'c 0 2.1 3.1 1' >> %t3
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t3 2> %t3.err | FileCheck %s --check-prefixes=FOONOCLONE,FOOSECTIONS
; RUN: FileCheck %s --check-prefixes=PATH3-WARN < %t3.err
;; Test that valid clonings are applied correctly, even if invalid clonings exist.
; RUN: echo 'v1' > %t3_1
; RUN: echo 'f foo' >> %t3_1
; RUN: echo 'p 0 2 3' >> %t3_1
; RUN: echo 'p 0 1 3' >> %t3_1
; RUN: echo 'c 0 1.1 3.2 2.1 3.1 1' >> %t3_1
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t3_1 2> %t3_1.err | FileCheck %s --check-prefixes=PATH3_1,FOONOCLONE,FOOSECTIONS
; RUN: FileCheck %s --check-prefixes=PATH3-WARN < %t3_1.err
; RUN: echo 'v1' > %t4
; RUN: echo 'f foo' >> %t4
; RUN: echo 'p 0 100' >> %t4
; RUN: echo 'c 0 100.1 1' >> %t4
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t4 2> %t4.err | FileCheck %s --check-prefixes=FOONOCLONE,FOOSECTIONS
; RUN: FileCheck %s --check-prefixes=PATH4-WARN < %t4.err

define void @foo(i1 %a, i1 %b, i1 %c, i1 %d) {
b0:
  call void @effect(i32 0)
  br i1 %a, label %b1, label %b3

b1:                                           ; preds = %b0
  call void @effect(i32 1)
  br i1 %b, label %b2, label %b3

b2:                                             ; preds = %b1
  call void @effect(i32 2)
  br label %b3

b3:                                            ; preds = %b0, %b1, %b2
  call void @effect(i32 3)
  br i1 %c, label %b4, label %b5

b4:                                             ; preds = %b3
  call void @effect(i32 4)
  br i1 %d, label %b5, label %cold

b5:                                            ; preds = %b3, %b4
  call void @effect(i32 5)
  ret void
cold:
  call void @effect(i32 6)                     ; preds = %b4
  ret void
}

; FOOSECTIONS:   .section    .text.foo,"ax",@progbits
; FOOSECTIONS: foo:
; FOOSECTIONS: # %bb.0:        # %b0

; FOONOCLONE:    je .LBB0_3
; PATH3_1:     # %bb.7:      # %b1
; PATH3_1:     # %bb.8:      # %b3
; PATH3_1:       jne .LBB0_4
; FOONOCLONE:  # %bb.1:      # %b1
; FOONOCLONE:    jne foo.cold

; PATH1:         jne .LBB0_1
; PATH1-NEXT:  # %bb.7:        # %b3
; PATH1:         jne .LBB0_4
; PATH1-NEXT:  # %bb.8:        # %b5
; PATH1:        retq
; PATH1-NEXT:  .LBB0_1:        # %b1
; PATH1:         je .LBB0_3
; PATH1-NEXT:  # %bb.2:        # %b2
; PATH1:         callq effect@PLT
; PATH1-NEXT:  .LBB0_3:        # %b3
; PATH1:         je .LBB0_5
; PATH1-NEXT:  .LBB0_4:        # %b4
; PATH1:         je foo.cold
; PATH1-NEXT:  .LBB0_5:        # %b5
; PATH1:         retq

; PATH2:         jne foo.__part.1
; PATH2-NEXT:  # %bb.7:        # %b3
; PATH2:         jne .LBB0_4
; PATH2-NEXT:  # %bb.8:        # %b5
; PATH2:        retq
; PATH2:         .section    .text.foo,"ax",@progbits,unique,1
; PATH2-NEXT:  foo.__part.1:   # %b1
; PATH2:         jne .LBB0_2
; PATH2-NEXT:  # %bb.9:        # %b3
; PATH2:         je .LBB0_5
; PATH2-NEXT:  # %bb.10:       # %b4
; PATH2:         je foo.cold
; PATH2-NEXT:  # %bb.11:       # %b5
; PATH2:         retq
; PATH2-NEXT:  .LBB0_2:        # %b2
; PATH2:         callq	effect@PLT
; PATH2-NEXT:  # %bb.3:        # %b3
; PATH2:         je .LBB0_5
; PATH2-NEXT:  .LBB0_4:        # %b4
; PATH2:          je foo.cold
; PATH2-NEXT:  .LBB0_5:       # %b5
; PATH2:          retq
; FOOSECTIONS:        .section    .text.split.foo,"ax",@progbits
; FOOCLONE:    foo.cold:      # %cold
; FOONOCLONE:  foo.cold:      # %b2

; PATH3-WARN: warning: block #2 is not a successor of block #0 in function foo
; PATH4-WARN: warning: no block with id 100 in function foo


;; Test valid application of cloning for paths with indirect branches.
; RUN: echo 'v1' > %t5
; RUN: echo 'f bar' >> %t5
; RUN: echo 'p 0 1' >> %t5
; RUN: echo 'c 0 1.1 2 1' >> %t5
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t5 | FileCheck %s --check-prefixes=PATH5,BARSECTIONS

;; Test failed application of path cloning for paths with indirect branches.
; RUN: echo 'v1' > %t6
; RUN: echo 'f bar' >> %t6
; RUN: echo 'p 0 1 2' >> %t6
; RUN: echo 'c 0 1.1 2.1 1' >> %t6
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t6 2> %t6.err | FileCheck %s --check-prefixes=BARNOCLONE,BARSECTIONS
; RUN: FileCheck %s --check-prefixes=PATH-INDIR-WARN < %t6.err
; RUN: echo 'v1' > %t7
; RUN: echo 'f bar' >> %t7
; RUN: echo 'p 1 2' >> %t7
; RUN: echo 'c 0 1 2.1' >> %t7
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t7 2> %t7.err | FileCheck %s --check-prefixes=BARNOCLONE,BARSECTIONS
; RUN: FileCheck %s --check-prefixes=PATH-INDIR-WARN < %t7.err


define void @bar(i1 %a, i1 %b) {
b0:
  call void @effect(i32 0)
  br i1 %a, label %b1, label %b2
b1:                                              ; preds = %b0
  call void @effect(i32 1)
  %0 = select i1 %b,                           ; <ptr> [#uses=1]
              ptr blockaddress(@bar, %b2),
              ptr blockaddress(@bar, %b3)
  indirectbr ptr %0, [label %b2, label %b3]
b2:                                              ; preds = %b0, %b1
  call void @effect(i32 2)
  ret void
b3:
  call void @effect(i32 3)                       ; preds = %b1
  ret void
}


; BARSECTIONS:       .section    .text.bar,"ax",@progbits
; BARSECTIONS:     bar:
; BARSECTIONS:     # %bb.0:        # %b0
; PATH5:             je .LBB1_2
; BARNOCLONE:        je bar.cold
; BARNOCLONE-NEXT: # %bb.1:        # %b1
; PATH5-NEXT:      # %bb.4:        # %b1
; PATH5:             jmpq *%rax
; PATH5-NEXT:      .Ltmp0:         # Block address taken
; PATH5-NEXT:      .LBB1_2:        # %b2
; PATH5:             retq
; PATH5-NEXT:      # %bb.1:        # %b1
; PATH5:             jmpq *%rax
; BARSECTIONS:       .section    .text.split.bar,"ax",@progbits
; PATH5:           bar.cold:       # %b3
; BARNOCLONE:      bar.cold:       # %b2   

; PATH-INDIR-WARN: warning: block #1 has indirect branch and appears as the non-tail block of a path in function bar
