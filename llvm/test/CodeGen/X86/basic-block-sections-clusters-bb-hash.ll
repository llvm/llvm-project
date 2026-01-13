; BB cluster section tests when using edges profile and basic blocks hashes to generate clusters.
; In the tests, we first generate hash values for basic blocks and write them to the profile.
; When generating basic blocks clusters, we match the hashes of basic blocks in the current CFG
; with those in the profile. After a successful match, we retrieve the weights of the basic blocks
; and edges from the profile. Subsequently, we use an inference algorithm to deduce the complete
; weights of all basic blocks and edges. Finally, we generate "hot" and "cold" clusters based on
; these complete weights. 
; In Test 1 and Test 2, the weights of basic blocks and edges in the profiles are different, which
; will ultimately result in distinct cluster partitioning outcomes.
;
; RUN: llc %s -O0 -mtriple=x86_64-pc-linux -function-sections -filetype=obj -basic-block-address-map -emit-bb-hash -o %t.o
;
; Test1: Basic blocks #0 (entry), #1 and #3 will be placed in the same section.
; The rest will be placed in the cold section.
;
; RUN: echo 'v1' > %t1
; RUN: echo 'f foo' >> %t1
; RUN: echo 'g 0:100,1:100,2:0 1:100,3:100 2:0,3:0 3:100' >> %t1
;
; These commands read BB hashes from SHT_LLVM_BB_ADDR_MAP 
; and put them into the basic blocks sections profile.
; RUN: llvm-readobj %t.o --bb-addr-map | \
; RUN: awk 'BEGIN {printf "h"} \
; RUN:     /ID: [0-9]+/ {id=$2} \
; RUN:     /Hash: 0x[0-9A-Fa-f]+/ {gsub(/^0x/, "", $2); hash=$2; printf " %%s:%%s", id, hash} \
; RUN:     END {print ""}' \
; RUN: >> %t1
;
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 -basic-block-section-match-infer | \
; RUN: FileCheck %s -check-prefixes=CHECK,LINUX-SECTIONS1
;
; Test2: Basic #0 (entry), #2 and #3 will be placed in the same section.
; The rest will be placed in the cold section.
;
; RUN: echo 'v1' > %t2
; RUN: echo 'f foo' >> %t2
; RUN: echo 'g 0:100,1:0,2:100 1:0,3:0 2:100,3:100 3:100' >> %t2
;
; These commands read BB hashes from SHT_LLVM_BB_ADDR_MAP 
; and put them into the basic blocks sections profile.
; RUN: llvm-readobj %t.o --bb-addr-map | \
; RUN: awk 'BEGIN {printf "h"} \
; RUN:     /ID: [0-9]+/ {id=$2} \
; RUN:     /Hash: 0x[0-9A-Fa-f]+/ {gsub(/^0x/, "", $2); hash=$2; printf " %%s:%%s", id, hash} \
; RUN:     END {print ""}' \
; RUN: >> %t2
;
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 -basic-block-section-match-infer | \
; RUN: FileCheck %s -check-prefixes=CHECK,LINUX-SECTIONS2

define void @foo(i1 zeroext) nounwind {
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

declare i32 @bar() #1

declare i32 @baz() #1

; CHECK:                   .section        .text.foo,"ax",@progbits
; CHECK-NOT:               .section
; CHECK-LABEL:             foo:
; CHECK-NOT:               .section
; CHECK-NOT:               .LBB_END0_{{0-9}}+
; LINUX-SECTIONS1-LABEL:   # %bb.1:
; LINUX-SECTIONS2-LABEL:   # %bb.2:
; CHECK-NOT:               .section
; CHECK-NOT:               .LBB_END0_{{0-9}}+
; CHECK-LABEL:             .LBB0_3:
; CHECK-LABEL:             .LBB_END0_3:
; CHECK-NEXT:              .section        .text.split.foo,"ax",@progbits
; CHECK-LABEL:             foo.cold:
; LINUX-SECTIONS1-LABEL:   .LBB_END0_2:
; LINUX-SECTIONS2-LABEL:   .LBB_END0_1:
; LINUX-SECTIONS1-LABEL:   .size   foo.cold, .LBB_END0_2-foo.cold
; LINUX-SECTIONS2-LABEL:   .size   foo.cold, .LBB_END0_1-foo.cold
; CHECK-LABEL:             .Lfunc_end0:
; CHECK-NEXT:              .size foo, .Lfunc_end0-foo
