; RUN: llc -mtriple=aarch64-apple-ios -asm-verbose=false \
; RUN:   -aarch64-enable-collect-loh=false -aarch64-enable-global-merge \
; RUN:   -global-merge-group-by-use -global-merge-ignore-single-use=false %s \
; RUN:   -o - | FileCheck %s

; We assume that globals of the same size aren't reordered inside a set.

; Check that we create two MergedGlobal instances for two functions using
; disjoint sets of globals

@m1 = internal global i32 0, align 4
@n1 = internal global i32 0, align 4

; CHECK-LABEL: f1:
define void @f1(i32 %a1, i32 %a2) #0 {
; CHECK-NEXT: adrp x8, [[SET1:l__MergedGlobals.[0-9]*]]@PAGE
; CHECK-NEXT: add x8, x8, [[SET1]]@PAGEOFF
; CHECK-NEXT: stp w0, w1, [x8]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m1, align 4
  store i32 %a2, i32* @n1, align 4
  ret void
}

@m2 = internal global i32 0, align 4
@n2 = internal global i32 0, align 4
@o2 = internal global i32 0, align 4

; CHECK-LABEL: f2:
define void @f2(i32 %a1, i32 %a2, i32 %a3) #0 {
; CHECK-NEXT: adrp x8, [[SET2:l__MergedGlobals.[0-9]*]]@PAGE
; CHECK-NEXT: add x8, x8, [[SET2]]@PAGEOFF
; CHECK-NEXT: stp w0, w1, [x8]
; CHECK-NEXT: str w2, [x8, #8]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m2, align 4
  store i32 %a2, i32* @n2, align 4
  store i32 %a3, i32* @o2, align 4
  ret void
}

; Sanity-check (don't worry about cost models) that we pick the biggest subset
; of all global used "together" directly or indirectly.  Here, that means
; merging n3, m4, and n4 together, but ignoring m3.

@m3 = internal global i32 0, align 4
@n3 = internal global i32 0, align 4

; CHECK-LABEL: f3:
define void @f3(i32 %a1, i32 %a2) #0 {
; CHECK-NEXT: adrp x8, _m3@PAGE
; CHECK-NEXT: adrp x9, [[SET3:l__MergedGlobals[0-9]*]]@PAGE
; CHECK-NEXT: str w0, [x8, _m3@PAGEOFF]
; CHECK-NEXT: str w1, [x9, [[SET3]]@PAGEOFF]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m3, align 4
  store i32 %a2, i32* @n3, align 4
  ret void
}

@m4 = internal global i32 0, align 4
@n4 = internal global i32 0, align 4

; CHECK-LABEL: f4:
define void @f4(i32 %a1, i32 %a2, i32 %a3) #0 {
; CHECK-NEXT: adrp x8, [[SET3]]@PAGE
; CHECK-NEXT: add x8, x8, [[SET3]]@PAGEOFF
; CHECK-NEXT: stp w2, w0, [x8]
; CHECK-NEXT: str w1, [x8, #8]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m4, align 4
  store i32 %a2, i32* @n4, align 4
  store i32 %a3, i32* @n3, align 4
  ret void
}

; Finally, check that we don't do anything with one-element global sets.
@o5 = internal global i32 0, align 4

; CHECK-LABEL: f5:
define void @f5(i32 %a1) #0 {
; CHECK-NEXT: adrp x8, _o5@PAGE
; CHECK-NEXT: str w0, [x8, _o5@PAGEOFF]
; CHECK-NEXT: ret
  store i32 %a1, i32* @o5, align 4
  ret void
}

; CHECK-DAG: .zerofill __DATA,__bss,_o5,4,2

; CHECK-DAG: .zerofill __DATA,__bss,[[SET1]],8,3
; CHECK-DAG: .zerofill __DATA,__bss,[[SET2]],12,3
; CHECK-DAG: .zerofill __DATA,__bss,[[SET3]],12,3

attributes #0 = { nounwind }
