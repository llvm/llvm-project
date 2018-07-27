; RUN: llc %s -o - -mtriple=armv8-unknown -mattr=-fuse-literals,+use-misched | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKDONT
; RUN: llc %s -o - -mtriple=armv8-unknown -mattr=+fuse-literals,+use-misched | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE

@g = common global i32* zeroinitializer

define i32* @litp(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %b, %a
  %ptr = getelementptr i32, i32* bitcast (i32* (i32, i32)* @litp to i32*), i32 %add
  %res = getelementptr i32, i32* bitcast (i32** @g to i32*), i32 %add
  store i32* %ptr, i32** @g, align 4
  ret i32* %res

; CHECK-LABEL: litp:
; CHECK:          movw [[R:r[0-9]+]], :lower16:litp
; CHECKDONT-NEXT: movw [[S:r[0-9]+]], :lower16:g
; CHECKFUSE-NEXT: movt [[R]], :upper16:litp
; CHECKFUSE-NEXT: movw [[S:r[0-9]+]], :lower16:g
; CHECKFUSE-NEXT: movt [[S]], :upper16:g
}

define i32 @liti(i32 %a, i32 %b) {
entry:
  %adda = add i32 %a, -262095121
  %add1 = add i32 %adda, %b
  %addb = add i32 %b, 121110837
  %add2 = add i32 %addb, %a
  store i32 %add1, i32* bitcast (i32** @g to i32*), align 4
  ret i32 %add2

; CHECK-LABEL: liti:
; CHECK:          movw [[R:r[0-9]+]], #309
; CHECKDONT-NEXT: add {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
; CHECKFUSE-NEXT: movt [[R]], #1848
; CHECKFUSE:      movw [[S:r[0-9]+]], :lower16:g
; CHECKFUSE-NEXT: movt [[S]], :upper16:g
; CHECKFUSE-NEXT: movw [[T:r[0-9]+]], #48879
; CHECKFUSE-NEXT: movt [[T]], #61536
}
