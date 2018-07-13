; RUN: opt -thinlto-bc %s -o %t1.bc
; RUN: llvm-lto2 run  -thinlto-distributed-indexes %t1.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,foo,plx \
; RUN:   -r %t1.bc,bar,x \
; RUN:   -r %t1.bc,addrtaken,px
; RUN: llvm-bcanalyzer -dump %t.out.index.bc | FileCheck %s --check-prefix=COMBINED

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @foo(i8* %p) !type !0 {
entry:
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

declare !type !0 i1 @bar(i8*)

; Functions must be address taken to have jump table entries emitted
define void @addrtaken(i1 %i) {
  %1 = select i1 %i, i1(i8*)* @foo, i1(i8*)* @bar
  ret void
}

declare i1 @llvm.type.test(i8* %ptr, metadata %type) nounwind readnone

!0 = !{i64 0, !"typeid1"}

; COMBINED:   <GLOBALVAL_SUMMARY_BLOCK
; COMBINED:     <CFI_FUNCTION_DEFS op0=0 op1=3/>
; COMBINED:     <CFI_FUNCTION_DECLS op0=3 op1=3/>
; COMBINED:     <TYPE_ID op0=6 op1=7 op2=4 op3=7 op4=0 op5=0 op6=0 op7=0/>
; COMBINED:   </GLOBALVAL_SUMMARY_BLOCK>

; COMBINED:      <STRTAB_BLOCK
; COMBINED-NEXT:   <BLOB abbrevid=4/> blob data = 'foobartypeid1'
; COMBINED-NEXT: </STRTAB_BLOCK>
