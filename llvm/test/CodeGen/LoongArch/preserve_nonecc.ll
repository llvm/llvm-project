; RUN: sed -e "s/RETTYPE/void/;s/RETVAL//" %s | llc -mtriple=loongarch64 | FileCheck --check-prefixes=ALL %s
; RUN: sed -e "s/RETTYPE/i32/;s/RETVAL/poison/" %s | llc -mtriple=loongarch64 | FileCheck --check-prefixes=ALL %s
; RUN: sed -e "s/RETTYPE/\{i64\,i64\}/;s/RETVAL/poison/" %s | llc -mtriple=loongarch64 | FileCheck --check-prefixes=ALL %s
; RUN: sed -e "s/RETTYPE/double/;s/RETVAL/0./" %s | llc -mtriple=loongarch64 | FileCheck --check-prefixes=ALL,DOUBLE %s

; We don't need to save registers before using them inside preserve_none function.
define preserve_nonecc RETTYPE @preserve_nonecc1(i64, i64, double, double) nounwind {
entry:
;ALL-LABEL:   preserve_nonecc1
;ALL:         # %bb.0:
;ALL-NEXT:    #APP
;ALL-NEXT:    #NO_APP
;DOUBLE-NEXT: movgr2fr.d $fa0, $zero
;ALL-NEXT:    ret
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
  ret RETTYPE RETVAL
}

; When calling a preserve_none function, all live registers must be saved and
; restored around the function call.
declare preserve_nonecc RETTYPE @preserve_nonecc2(i64, i64, double, double)
define void @bar() nounwind {
entry:
;ALL-LABEL: bar
;ALL:       #APP
;ALL:       st.d $t0
;ALL:       st.d $t1
;ALL:       st.d $t2
;ALL:       st.d $t3
;ALL:       st.d $t4
;ALL:       st.d $t5
;ALL:       st.d $t6
;ALL:       st.d $t7
;ALL:       st.d $t8
;ALL:       st.d $s0
;ALL:       st.d $s1
;ALL:       st.d $s2
;ALL:       st.d $s3
;ALL:       st.d $s4
;ALL:       st.d $s5
;ALL:       st.d $s6
;ALL:       st.d $s7
;ALL:       move $fp, $s8
;ALL:       fst.d $fa7
;ALL:       fst.d $fs0
;ALL:       fst.d $fs1
;ALL:       fst.d $fs2
;ALL:       fst.d $fs3
;ALL:       fst.d $fs4
;ALL:       fst.d $fs5
;ALL:       fst.d $fs6
;ALL:       fst.d $fs7
;ALL:       ld.d $t0
;ALL:       ld.d $t1
;ALL:       ld.d $t2
;ALL:       ld.d $t3
;ALL:       ld.d $t4
;ALL:       ld.d $t5
;ALL:       ld.d $t6
;ALL:       ld.d $t7
;ALL:       ld.d $t8
;ALL:       ld.d $s0
;ALL:       ld.d $s1
;ALL:       ld.d $s2
;ALL:       ld.d $s3
;ALL:       ld.d $s4
;ALL:       ld.d $s5
;ALL:       ld.d $s6
;ALL:       ld.d $s7
;ALL:       move $s8, $fp
;ALL:       fld.d $fa7
;ALL:       fld.d $fs0
;ALL:       fld.d $fs1
;ALL:       fld.d $fs2
;ALL:       fld.d $fs3
;ALL:       fld.d $fs4
;ALL:       fld.d $fs5
;ALL:       fld.d $fs6
;ALL:       fld.d $fs7
;ALL:       #APP
  %a0 = call i64 asm sideeffect "", "={r12}"() nounwind
  %a1 = call i64 asm sideeffect "", "={r13}"() nounwind
  %a2 = call i64 asm sideeffect "", "={r14}"() nounwind
  %a3 = call i64 asm sideeffect "", "={r15}"() nounwind
  %a4 = call i64 asm sideeffect "", "={r16}"() nounwind
  %a5 = call i64 asm sideeffect "", "={r17}"() nounwind
  %a6 = call i64 asm sideeffect "", "={r18}"() nounwind
  %a7 = call i64 asm sideeffect "", "={r19}"() nounwind
  %a8 = call i64 asm sideeffect "", "={r20}"() nounwind
  %a9 = call i64 asm sideeffect "", "={r23}"() nounwind
  %a10 = call i64 asm sideeffect "", "={r24}"() nounwind
  %a11 = call i64 asm sideeffect "", "={r25}"() nounwind
  %a12 = call i64 asm sideeffect "", "={r26}"() nounwind
  %a13 = call i64 asm sideeffect "", "={r27}"() nounwind
  %a14 = call i64 asm sideeffect "", "={r28}"() nounwind
  %a15 = call i64 asm sideeffect "", "={r29}"() nounwind
  %a16 = call i64 asm sideeffect "", "={r30}"() nounwind
  %a17 = call i64 asm sideeffect "", "={r31}"() nounwind

  %f0 = call double asm sideeffect "", "={$f7}"() nounwind
  %f1 = call double asm sideeffect "", "={$f24}"() nounwind
  %f2 = call double asm sideeffect "", "={$f25}"() nounwind
  %f3 = call double asm sideeffect "", "={$f26}"() nounwind
  %f4 = call double asm sideeffect "", "={$f27}"() nounwind
  %f5 = call double asm sideeffect "", "={$f28}"() nounwind
  %f6 = call double asm sideeffect "", "={$f29}"() nounwind
  %f7 = call double asm sideeffect "", "={$f30}"() nounwind
  %f8 = call double asm sideeffect "", "={$f31}"() nounwind

  call preserve_nonecc RETTYPE @preserve_nonecc2(i64 1, i64 2, double 3.0, double 4.0)
  call void asm sideeffect "", "{r12},{r13},{r14},{r15},{r16},{r17},{r18},{r19},{r20},{r23},{r24},{r25},{r26},{r27},{r28},{r29},{r30},{r31},{$f7},{$f24},{$f25},{$f26},{$f27},{$f28},{$f29},{$f30},{$f31}"(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, double %f0, double %f1, double %f2, double %f3, double %f4, double %f5, double %f6, double %f7, double %f8)
  ret void
}
