; RUN: opt -thinlto-bc %s -o %t1.bc
; RUN: llvm-lto2 run  -thinlto-distributed-indexes %t1.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,foo,plx \
; RUN:   -r %t1.bc,bar,x
; RUN: llvm-bcanalyzer -dump %t.out.index.bc | FileCheck %s --check-prefix=COMBINED

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @foo(i8* %p) !type !0 {
entry:
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

declare !type !0 void @bar()

declare i1 @llvm.type.test(i8* %ptr, metadata %type) nounwind readnone

!0 = !{i64 0, !"typeid1"}

; COMBINED:   <GLOBALVAL_SUMMARY_BLOCK
; COMBINED:     <CFI_FUNCTION_DEFS op0=0 op1=3/>
; COMBINED:     <CFI_FUNCTION_DECLS op0=3 op1=3/>
; COMBINED:   </GLOBALVAL_SUMMARY_BLOCK>

; COMBINED:      <STRTAB_BLOCK
; COMBINED-NEXT:   <BLOB abbrevid=4/> blob data = 'foobar'
; COMBINED-NEXT: </STRTAB_BLOCK>
