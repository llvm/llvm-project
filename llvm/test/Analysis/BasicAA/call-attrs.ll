; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare void @readonly_attr(ptr readonly nocapture)
declare void @writeonly_attr(ptr writeonly nocapture)
declare void @readnone_attr(ptr readnone nocapture)

declare void @readonly_func(ptr nocapture) readonly
declare void @writeonly_func(ptr nocapture) writeonly
declare void @readnone_func(ptr nocapture) readnone

declare void @read_write(ptr writeonly nocapture, ptr readonly nocapture, ptr readnone nocapture)

declare void @func()

define void @test(ptr noalias %p) {
entry:
  load i8, ptr %p
  call void @readonly_attr(ptr %p)
  call void @readonly_func(ptr %p)

  call void @writeonly_attr(ptr %p)
  call void @writeonly_func(ptr %p)

  call void @readnone_attr(ptr %p)
  call void @readnone_func(ptr %p)

  call void @read_write(ptr %p, ptr %p, ptr %p)

  call void @func() ["deopt" (ptr %p)]
  call void @writeonly_attr(ptr %p) ["deopt" (ptr %p)]

  ret void
}

; CHECK:  Just Ref:  Ptr: i8* %p	<->  call void @readonly_attr(ptr %p)
; CHECK:  Just Ref:  Ptr: i8* %p	<->  call void @readonly_func(ptr %p)
; CHECK:  Just Mod:  Ptr: i8* %p	<->  call void @writeonly_attr(ptr %p)
; CHECK:  Just Mod:  Ptr: i8* %p	<->  call void @writeonly_func(ptr %p)
; CHECK:  NoModRef:  Ptr: i8* %p	<->  call void @readnone_attr(ptr %p)
; CHECK:  NoModRef:  Ptr: i8* %p	<->  call void @readnone_func(ptr %p)
; CHECK:  Both ModRef:  Ptr: i8* %p	<->  call void @read_write(ptr %p, ptr %p, ptr %p)
; CHECK:  Just Ref:  Ptr: i8* %p	<->  call void @func() [ "deopt"(ptr %p) ]
; CHECK:  Both ModRef:  Ptr: i8* %p	<->  call void @writeonly_attr(ptr %p) [ "deopt"(ptr %p) ]
