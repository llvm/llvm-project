; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - \
; RUN:   | FileCheck %s -check-prefix=CHECKV4

; RUN: llc -mtriple=arm-eabi -mattr=+v5t %s -o - \
; RUN:   | FileCheck %s -check-prefix=CHECKV5

; RUN: llc -mtriple=armv6-linux-gnueabi -relocation-model=pic %s -o - \
; RUN:   | FileCheck %s -check-prefix=CHECKELF

@t = weak global ptr null           ; <ptr> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; CHECKELF: bl g
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @g.upgrd.1() {
; CHECKV4: mov lr, pc
; CHECKV5: blx
        %tmp = load ptr, ptr @t         ; <ptr> [#uses=1]
        %tmp.upgrd.2 = call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}

define ptr @m_231b(i32, i32, ptr, ptr, ptr) nounwind {
; CHECKV4: m_231b
; CHECKV4: bx r{{.*}}
BB0:
  %5 = inttoptr i32 %0 to ptr                    ; <ptr> [#uses=1]
  %t35 = load volatile i32, ptr %5                    ; <i32> [#uses=1]
  %6 = inttoptr i32 %t35 to ptr                 ; <ptr> [#uses=1]
  %7 = getelementptr ptr, ptr %6, i32 86             ; <ptr> [#uses=1]
  %8 = load ptr, ptr %7                              ; <ptr> [#uses=1]
  %9 = call ptr %8(i32 %0, ptr null, i32 %1, ptr %2, ptr %3, ptr %4) ; <ptr> [#uses=1]
  ret ptr %9
}
