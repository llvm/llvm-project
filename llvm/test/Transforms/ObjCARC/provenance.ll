; RUN: opt -disable-output -aa-pipeline= -passes=pa-eval %s 2>&1 | FileCheck %s

@"\01l_objc_msgSend_fixup_" = global i8 0
@g1 = global i8 0, section "__OBJC,__message_refs,literal_pointers,no_dead_strip"
@g2 = global i8 0, section "__DATA, __objc_classrefs, regular, no_dead_strip"
@g3 = global i8 0, section "__DATA, __objc_superrefs, regular, no_dead_strip"
@g4 = global i8 0, section "__TEXT,__objc_methname,cstring_literals"
@g5 = global i8 0, section "__TEXT,__cstring,cstring_literals"

declare void @g(i8)

define void @f(i8* %a, i8** %b, i8** %c) {
  %y1 = load i8, i8* %a
  call void @g(i8 %y1)

  %y2 = load i8*, i8** %b
  %y3 = load i8*, i8** %c

  %x0 = load i8, i8* @"\01l_objc_msgSend_fixup_"
  call void @g(i8 %x0)

  %x1 = load i8, i8* @g1
  call void @g(i8 %x1)

  %x2 = load i8, i8* @g2
  call void @g(i8 %x2)

  %x3 = load i8, i8* @g3
  call void @g(i8 %x3)

  %x4 = load i8, i8* @g4
  call void @g(i8 %x4)

  %x5 = load i8, i8* @g5
  call void @g(i8 %x5)
  ret void
}

; CHECK: y1 and y2 are related.
; CHECK: y1 and y3 are related.
; CHECK: y2 and y3 are related.
; CHECK: x0 and y1 are not related.
; CHECK: x0 and y2 are not related.
; CHECK: x0 and y3 are not related.
; CHECK: l_objc_msgSend_fixup_ and y1 are not related.
; CHECK: l_objc_msgSend_fixup_ and y2 are not related.
; CHECK: l_objc_msgSend_fixup_ and y3 are not related.
; CHECK: x1 and y1 are not related.
; CHECK: x2 and y1 are not related.
; CHECK: x3 and y1 are not related.
; CHECK: x4 and y1 are not related.
; CHECK: x5 and y1 are not related.
