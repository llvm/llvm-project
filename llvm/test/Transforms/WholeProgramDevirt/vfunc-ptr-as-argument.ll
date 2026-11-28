; RUN: opt -S -passes=wholeprogramdevirt -devirtualize-speculatively %s | FileCheck %s

; A virtual function pointer loaded from the vtable and passed to another
; function as an argument is not a virtual call. Treating it as one tripped
; the "devirtualizing direct call?" assertion.

target datalayout = "e-p:64:64"

@vt = constant [1 x ptr] [ptr @vf], !type !0

define i1 @vf(ptr %this) {
  ret i1 true
}

declare void @use(ptr)

; CHECK-LABEL: define void @arg(
; CHECK:      [[FPTR:%.*]] = load ptr, ptr %vtable
; CHECK-NEXT: call void @use(ptr [[FPTR]])
; CHECK-NEXT: ret void
define void @arg(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call void @use(ptr %fptr)
  ret void
}

; A virtual call next to the argument use is still devirtualized, and the
; argument is left alone.
; CHECK-LABEL: define void @both(
; CHECK:      [[FPTR2:%.*]] = load ptr, ptr %vtable
; CHECK-NEXT: call void @use(ptr [[FPTR2]])
; CHECK:      if.true.direct_targ:
; CHECK-NEXT: call i1 @vf(ptr %obj)
; CHECK:      if.false.orig_indirect:
; CHECK-NEXT: call i1 [[FPTR2]](ptr %obj)
define void @both(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call void @use(ptr %fptr)
  call i1 %fptr(ptr %obj)
  ret void
}

; The same through llvm.type.checked.load: the argument is a non-call use,
; so it gets a plain load of the slot.
; CHECK-LABEL: define void @checked_load(
; CHECK:      [[SLOT:%.*]] = getelementptr i8, ptr %vtable, i64 0
; CHECK-NEXT: [[FPTR3:%.*]] = load ptr, ptr [[SLOT]]
; CHECK-NEXT: call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
; CHECK:      call void @use(ptr [[FPTR3]])
define void @checked_load(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  br i1 %p, label %cont, label %trap

trap:
  call void @llvm.trap()
  unreachable

cont:
  call void @use(ptr %fptr)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare void @llvm.trap()

!0 = !{i32 0, !"typeid"}
