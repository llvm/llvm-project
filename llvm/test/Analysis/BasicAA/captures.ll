; RUN: opt < %s -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare void @capture(ptr)
declare ptr @get_ptr()

; CHECK-LABEL: address_capture
; CHECK: NoAlias:	i32* %a, i32* %p
; CHECK: NoModRef:  Ptr: i32* %a	<->  %p = call ptr @get_ptr()
define void @address_capture() {
  %a = alloca i32
  call void @capture(ptr captures(address) %a)
  %p = call ptr @get_ptr()
  store i32 0, ptr %p
  load i32, ptr %a
  ret void
}

; CHECK-LABEL: read_only_capture
; CHECK: MayAlias:	i32* %a, i32* %p
; CHECK: Just Ref:  Ptr: i32* %a	<->  %p = call ptr @get_ptr()
define void @read_only_capture() {
  %a = alloca i32
  call void @capture(ptr captures(address, read_provenance) %a)
  %p = call ptr @get_ptr()
  store i32 0, ptr %p
  load i32, ptr %a
  ret void
}

; CHECK-LABEL: address_capture_and_full_capture
; CHECK: MayAlias:	i32* %a, i32* %p
; CHECK: Both ModRef:  Ptr: i32* %a	<->  %p = call ptr @get_ptr()
define void @address_capture_and_full_capture() {
  %a = alloca i32
  call void @capture(ptr captures(address) %a)
  call void @capture(ptr %a)
  %p = call ptr @get_ptr()
  store i32 0, ptr %p
  load i32, ptr %a
  ret void
}

; CHECK-LABEL: address_capture_and_full_capture_commuted
; CHECK: MayAlias:	i32* %a, i32* %p
; CHECK: Both ModRef:  Ptr: i32* %a	<->  %p = call ptr @get_ptr()
define void @address_capture_and_full_capture_commuted() {
  %a = alloca i32
  call void @capture(ptr %a)
  call void @capture(ptr captures(address) %a)
  %p = call ptr @get_ptr()
  store i32 0, ptr %p
  load i32, ptr %a
  ret void
}

; CHECK-LABEL: read_only_capture_and_full_capture
; CHECK: MayAlias:	i32* %a, i32* %p
; CHECK: Both ModRef:  Ptr: i32* %a	<->  %p = call ptr @get_ptr()
define void @read_only_capture_and_full_capture() {
  %a = alloca i32
  call void @capture(ptr captures(address, read_provenance) %a)
  call void @capture(ptr %a)
  %p = call ptr @get_ptr()
  store i32 0, ptr %p
  load i32, ptr %a
  ret void
}

; CHECK-LABEL: read_only_capture_and_full_capture_commuted
; CHECK: MayAlias:	i32* %a, i32* %p
; CHECK: Both ModRef:  Ptr: i32* %a	<->  %p = call ptr @get_ptr()
define void @read_only_capture_and_full_capture_commuted() {
  %a = alloca i32
  call void @capture(ptr %a)
  call void @capture(ptr captures(address, read_provenance) %a)
  %p = call ptr @get_ptr()
  store i32 0, ptr %p
  load i32, ptr %a
  ret void
}
