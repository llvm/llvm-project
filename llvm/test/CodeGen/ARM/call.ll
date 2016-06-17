; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - \
; RUN:   | FileCheck %s -check-prefix=CHECKV4

; RUN: llc -mtriple=arm-eabi -mattr=+v5t %s -o - \
; RUN:   | FileCheck %s -check-prefix=CHECKV5

; RUN: llc -mtriple=armv6-linux-gnueabi -relocation-model=pic %s -o - \
; RUN:   | FileCheck %s -check-prefix=CHECKELF

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; CHECKELF: bl g
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @g.upgrd.1() {
; CHECKV4: mov lr, pc
; CHECKV5: blx
        %tmp = load i32 ()*, i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}

define i32* @m_231b(i32, i32, i32*, i32*, i32*) nounwind {
; CHECKV4: m_231b
; CHECKV4: bx r{{.*}}
BB0:
  %5 = inttoptr i32 %0 to i32*                    ; <i32*> [#uses=1]
  %t35 = load volatile i32, i32* %5                    ; <i32> [#uses=1]
  %6 = inttoptr i32 %t35 to i32**                 ; <i32**> [#uses=1]
  %7 = getelementptr i32*, i32** %6, i32 86             ; <i32**> [#uses=1]
  %8 = load i32*, i32** %7                              ; <i32*> [#uses=1]
  %9 = bitcast i32* %8 to i32* (i32, i32*, i32, i32*, i32*, i32*)* ; <i32* (i32, i32*, i32, i32*, i32*, i32*)*> [#uses=1]
  %10 = call i32* %9(i32 %0, i32* null, i32 %1, i32* %2, i32* %3, i32* %4) ; <i32*> [#uses=1]
  ret i32* %10
}
