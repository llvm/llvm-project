; RUN: llc -mtriple=aarch64-apple-ios -asm-verbose=false \
; RUN:   -aarch64-enable-collect-loh=false -O1 -global-merge-group-by-use \
; RUN:   -global-merge-ignore-single-use %s -o - | FileCheck %s

; Check that, at -O1, we only merge globals used in minsize functions.
; We assume that globals of the same size aren't reordered inside a set.
; We use -global-merge-ignore-single-use, and thus only expect one merged set.

@m1 = internal global i32 0, align 4
@n1 = internal global i32 0, align 4

; CHECK-LABEL: f1:
define void @f1(i32 %a1, i32 %a2) minsize nounwind {
; CHECK-NEXT: adrp x8, [[SET:__MergedGlobals]]@PAGE
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
define void @f2(i32 %a1, i32 %a2) nounwind {
; CHECK-NEXT: adrp x8, _m2@PAGE
; CHECK-NEXT: adrp x9, _n2@PAGE
; CHECK-NEXT: str w0, [x8, _m2@PAGEOFF]
; CHECK-NEXT: str w1, [x9, _n2@PAGEOFF]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m2, align 4
  store i32 %a2, i32* @n2, align 4
  ret void
}

; If we have use sets partially overlapping between a minsize and a non-minsize
; function, explicitly check that we only consider the globals used in the
; minsize function for merging.

@m3 = internal global i32 0, align 4
@n3 = internal global i32 0, align 4

; CHECK-LABEL: f3:
define void @f3(i32 %a1, i32 %a2) minsize nounwind {
; CHECK-NEXT: adrp x8, [[SET]]@PAGE+8
; CHECK-NEXT: add x8, x8, [[SET]]@PAGEOFF+8
; CHECK-NEXT: stp w0, w1, [x8]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m3, align 4
  store i32 %a2, i32* @n3, align 4
  ret void
}

@n4 = internal global i32 0, align 4

; CHECK-LABEL: f4:
define void @f4(i32 %a1, i32 %a2) nounwind {
; CHECK-NEXT: adrp x8, [[SET]]@PAGE+8
; CHECK-NEXT: adrp x9, _n4@PAGE
; CHECK-NEXT: str w0, [x8, [[SET]]@PAGEOFF+8]
; CHECK-NEXT: str w1, [x9, _n4@PAGEOFF]
; CHECK-NEXT: ret
  store i32 %a1, i32* @m3, align 4
  store i32 %a2, i32* @n4, align 4
  ret void
}

; CHECK-DAG: .zerofill __DATA,__bss,[[SET]],16,2
; CHECK-DAG: .zerofill __DATA,__bss,_m2,4,2
; CHECK-DAG: .zerofill __DATA,__bss,_n2,4,2
; CHECK-DAG: .zerofill __DATA,__bss,_n4,4,2
