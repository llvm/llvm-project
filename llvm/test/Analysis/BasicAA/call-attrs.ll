; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare void @readonly_attr(i8* readonly nocapture)
declare void @writeonly_attr(i8* writeonly nocapture)
declare void @readnone_attr(i8* readnone nocapture)

declare void @readonly_func(i8* nocapture) readonly
declare void @writeonly_func(i8* nocapture) writeonly
declare void @readnone_func(i8* nocapture) readnone

declare void @read_write(i8* writeonly nocapture, i8* readonly nocapture, i8* readnone nocapture)

declare void @func()

define void @test(i8* noalias %p) {
entry:
  load i8, i8* %p
  call void @readonly_attr(i8* %p)
  call void @readonly_func(i8* %p)

  call void @writeonly_attr(i8* %p)
  call void @writeonly_func(i8* %p)

  call void @readnone_attr(i8* %p)
  call void @readnone_func(i8* %p)

  call void @read_write(i8* %p, i8* %p, i8* %p)

  call void @func() ["deopt" (i8* %p)]
  call void @writeonly_attr(i8* %p) ["deopt" (i8* %p)]

  ret void
}

; CHECK:  Just Ref (MustAlias):  Ptr: i8* %p	<->  call void @readonly_attr(i8* %p)
; CHECK:  Just Ref:  Ptr: i8* %p	<->  call void @readonly_func(i8* %p)
; CHECK:  Just Mod (MustAlias):  Ptr: i8* %p	<->  call void @writeonly_attr(i8* %p)
; CHECK:  Just Mod:  Ptr: i8* %p	<->  call void @writeonly_func(i8* %p)
; CHECK:  NoModRef:  Ptr: i8* %p	<->  call void @readnone_attr(i8* %p)
; CHECK:  NoModRef:  Ptr: i8* %p	<->  call void @readnone_func(i8* %p)
; CHECK:  Both ModRef:  Ptr: i8* %p	<->  call void @read_write(i8* %p, i8* %p, i8* %p)
; CHECK:  Just Ref (MustAlias):  Ptr: i8* %p	<->  call void @func() [ "deopt"(i8* %p) ]
; CHECK:  Both ModRef:  Ptr: i8* %p	<->  call void @writeonly_attr(i8* %p) [ "deopt"(i8* %p) ]
