; RUN: llc -mtriple=aarch64-apple-ios -asm-verbose=false \
; RUN:   -aarch64-enable-collect-loh=false -aarch64-enable-global-merge \
; RUN:   -global-merge-group-by-use -global-merge-ignore-single-use %s -o - \
; RUN:   | FileCheck %s

; We assume that globals of the same size aren't reordered inside a set.

@m1 = internal global i32 0, align 4
@n1 = internal global i32 0, align 4
@o1 = internal global i32 0, align 4

; CHECK-LABEL: f1:
define void @f1(i32 %a1, i32 %a2) #0 {
; CHECK-NEXT: adrp x8, [[SET:l__MergedGlobals]]@PAGE
; CHECK-NEXT: add x8, x8, [[SET]]@PAGEOFF
; CHECK-NEXT: stp w0, w1, [x8]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m1, align 4
  store i32 %a2, i32* @n1, align 4
  ret void
}

@m2 = internal global i32 0, align 4
@n2 = internal global i32 0, align 4

; CHECK-LABEL: f2:
define void @f2(i32 %a1, i32 %a2, i32 %a3) #0 {
; CHECK-NEXT: adrp x8, [[SET]]@PAGE
; CHECK-NEXT: add x8, x8, [[SET]]@PAGEOFF
; CHECK-NEXT: stp w0, w1, [x8]
; CHECK-NEXT: str w2, [x8, #8]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m1, align 4
  store i32 %a2, i32* @n1, align 4
  store i32 %a3, i32* @o1, align 4
  ret void
}

; CHECK-LABEL: f3:
define void @f3(i32 %a1, i32 %a2) #0 {
; CHECK-NEXT: adrp x8, [[SET]]@PAGE
; CHECK-NEXT: add x8, x8, [[SET]]@PAGEOFF
; CHECK-NEXT: stp w0, w1, [x8, #12]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m2, align 4
  store i32 %a2, i32* @n2, align 4
  ret void
}

@o2 = internal global i32 0, align 4

; CHECK-LABEL: f4:
define void @f4(i32 %a1) #0 {
; CHECK-NEXT: adrp x8, _o2@PAGE
; CHECK-NEXT: str w0, [x8, _o2@PAGEOFF]
; CHECK-NEXT: ret
  store i32 %a1, i32* @o2, align 4
  ret void
}

; CHECK-DAG: .zerofill __DATA,__bss,[[SET]],20,4
; CHECK-DAG: .zerofill __DATA,__bss,_o2,4,2

attributes #0 = { nounwind }
