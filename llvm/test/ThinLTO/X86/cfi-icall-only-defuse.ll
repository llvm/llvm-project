; Verifies that only functions defined or used by each module make it into the
; CFI functions sets in that module's distributed index.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t1.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %S/Inputs/cfi-icall-only-bazqux.ll -o %t2.bc
; RUN: llvm-lto2 run  -thinlto-distributed-indexes %t1.bc %t2.bc -o %t.out \
; RUN:   -r %t1.bc,bar,plx \
; RUN:   -r %t1.bc,baz,x   \
; RUN:   -r %t1.bc,f,plx   \
; RUN:   -r %t1.bc,foo,plx \
; RUN:   -r %t2.bc,bar,x   \
; RUN:   -r %t2.bc,baz,plx \
; RUN:   -r %t2.bc,g,plx   \
; RUN:   -r %t2.bc,qux,plx
; RUN: llvm-bcanalyzer -dump %t1.bc.thinlto.bc | FileCheck %s --check-prefix=FOOBAZ
; RUN: llvm-bcanalyzer -dump %t2.bc.thinlto.bc | FileCheck %s --check-prefix=BARQUX

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare !type !0 i8 @baz(i8*)
declare i1 @llvm.type.test(i8* %ptr, metadata %type) nounwind readnone

define i8 @foo(i8* %p) !type !0 {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"t1")
  %1 = select i1 %x, i8 0, i8 1
  ret i8 %1
}

define i8 @bar(i8* %p) !type !0 {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"t1")
  ret i8 2
}

define i8 @f(i1 %i, i8* %p) {
  %1 = select i1 %i, i8(i8*)* @foo, i8(i8*)* @baz
  %2 = call i8 %1(i8* %p)
  ret i8 %2
}

!0 = !{i64 0, !"t1"}

; FOOBAZ:   <GLOBALVAL_SUMMARY_BLOCK
; FOOBAZ:     <CFI_FUNCTION_DEFS op0=0 op1=3 op2=3 op3=3 op4=6 op5=3/>
; FOOBAZ:     <TYPE_ID op0=9 op1=2 op2=4 op3=7 op4=0 op5=0 op6=0 op7=0/>
; FOOBAZ:   </GLOBALVAL_SUMMARY_BLOCK>
; FOOBAZ:      <STRTAB_BLOCK
; FOOBAZ-NEXT:   <BLOB abbrevid=4/> blob data = 'barbazfoot1'
; FOOBAZ-NEXT: </STRTAB_BLOCK>

; BARQUX:   <GLOBALVAL_SUMMARY_BLOCK
; BARQUX:     <CFI_FUNCTION_DEFS op0=0 op1=3 op2=3 op3=3 op4=6 op5=3/>
; BARQUX:     <TYPE_ID op0=9 op1=2 op2=4 op3=7 op4=0 op5=0 op6=0 op7=0/>
; BARQUX:   </GLOBALVAL_SUMMARY_BLOCK>
; BARQUX:      <STRTAB_BLOCK
; BARQUX-NEXT:   <BLOB abbrevid=4/> blob data = 'barbazquxt1'
; BARQUX-NEXT: </STRTAB_BLOCK>
