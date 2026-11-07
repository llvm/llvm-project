; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

declare void @read_fpcontrol_func() memory(none, fpcontrol: read)
declare void @write_fpcontrol_func() memory(none, fpcontrol: write)
declare void @read_write_fpcontrol_func() memory(none, fpcontrol: readwrite)
declare void @fpoperation_func() memory(none, fpcontrol: read, fpstatus: readwrite)
declare void @unknown()
declare void @none() memory(none)
declare void @write() memory(write)


define void @test_call_modref(ptr %s) {
entry:
  %v = load float, ptr %s
  call void @unknown()
  call void @none()
  call void @write()
  call void @read_fpcontrol_func()
  call void @write_fpcontrol_func()
  call void @read_write_fpcontrol_func()
  call void @fpoperation_func()

  ret void
}

; CHECK:  NoModRef:  Ptr: float* %s	<->  call void @read_fpcontrol_func()
; CHECK:  NoModRef:  Ptr: float* %s	<->  call void @write_fpcontrol_func()
; CHECK:  NoModRef:  Ptr: float* %s	<->  call void @read_write_fpcontrol_func()
; CHECK:  NoModRef:  Ptr: float* %s	<->  call void @fpoperation_func()

; CHECK:  Both ModRef:   call void @unknown() <->   call void @read_fpcontrol_func()
; CHECK:  Both ModRef:   call void @unknown() <->   call void @write_fpcontrol_func()
; CHECK:  Both ModRef:   call void @unknown() <->   call void @read_write_fpcontrol_func()
; CHECK:  Both ModRef:   call void @unknown() <->   call void @fpoperation_func()

; CHECK:  NoModRef:   call void @none() <->   call void @read_fpcontrol_func()
; CHECK:  NoModRef:   call void @none() <->   call void @write_fpcontrol_func()
; CHECK:  NoModRef:   call void @none() <->   call void @read_write_fpcontrol_func()
; CHECK:  NoModRef:   call void @none() <->   call void @fpoperation_func()

; CHECK:  Just Mod:   call void @write() <->   call void @read_fpcontrol_func()
; CHECK:  Just Mod:   call void @write() <->   call void @write_fpcontrol_func()
; CHECK:  Just Mod:   call void @write() <->   call void @read_write_fpcontrol_func()
; CHECK:  Just Mod:   call void @write() <->   call void @fpoperation_func()

; CHECK:  Just Ref:   call void @read_fpcontrol_func() <->   call void @unknown()
; CHECK:  NoModRef:   call void @read_fpcontrol_func() <->   call void @none()
; CHECK:  Just Ref:   call void @read_fpcontrol_func() <->   call void @write()
; CHECK:  Just Ref:   call void @read_fpcontrol_func() <->   call void @write_fpcontrol_func()
; CHECK:  Just Ref:   call void @read_fpcontrol_func() <->   call void @read_write_fpcontrol_func()
; CHECK:  NoModRef:   call void @read_fpcontrol_func() <->   call void @fpoperation_func()

; CHECK:  Just Mod:   call void @write_fpcontrol_func() <->   call void @unknown()
; CHECK:  NoModRef:   call void @write_fpcontrol_func() <->   call void @none()
; CHECK:  Just Mod:   call void @write_fpcontrol_func() <->   call void @write()
; CHECK:  Just Mod:   call void @write_fpcontrol_func() <->   call void @read_fpcontrol_func()
; CHECK:  Just Mod:   call void @write_fpcontrol_func() <->   call void @read_write_fpcontrol_func()
; CHECK:  Just Mod:   call void @write_fpcontrol_func() <->   call void @fpoperation_func()

; CHECK:  Both ModRef:   call void @read_write_fpcontrol_func() <->   call void @unknown()
; CHECK:  NoModRef:   call void @read_write_fpcontrol_func() <->   call void @none()
; CHECK:  Both ModRef:   call void @read_write_fpcontrol_func() <->   call void @write()
; CHECK:  Both ModRef:   call void @read_write_fpcontrol_func() <->   call void @read_fpcontrol_func()
; CHECK:  Both ModRef:   call void @read_write_fpcontrol_func() <->   call void @write_fpcontrol_func()
; CHECK:  Both ModRef:   call void @read_write_fpcontrol_func() <->   call void @fpoperation_func()

; CHECK:  Both ModRef:   call void @fpoperation_func() <->   call void @unknown()
; CHECK:  NoModRef:   call void @fpoperation_func() <->   call void @none()
; CHECK:  Both ModRef:   call void @fpoperation_func() <->   call void @write()
; CHECK:  NoModRef:   call void @fpoperation_func() <->   call void @read_fpcontrol_func()
; CHECK:  Just Ref:   call void @fpoperation_func() <->   call void @write_fpcontrol_func()
; CHECK:  Just Ref:   call void @fpoperation_func() <->   call void @read_write_fpcontrol_func()
