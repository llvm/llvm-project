; RUN: opt < %s -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

@c = constant [8 x i32] zeroinitializer

declare void @dummy()

declare void @foo(ptr)

; CHECK-LABEL: Function: basic
; CHECK: NoModRef: Ptr: i32* @c	<->  call void @dummy()
define void @basic(ptr %p) {
  call void @dummy()
  load i32, ptr @c
  ret void
}

; CHECK-LABEL: Function: recphi
; CHECK: NoModRef: Ptr: i32* %p	<->  call void @dummy()
define void @recphi() {
entry:
  br label %loop

loop:
  %p = phi ptr [ @c, %entry ], [ %p.next, %loop ]
  call void @dummy()
  load i32, ptr %p
  %p.next = getelementptr i32, ptr %p, i64 1
  %c = icmp ne ptr %p.next, getelementptr (i32, ptr @c, i64 8)
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; Tests that readonly noalias implies !Mod.
;
; CHECK-LABEL: Function: readonly_noalias
; CHECK: Just Ref: Ptr: i32* %p <->  call void @foo(ptr %p)
define void @readonly_noalias(ptr readonly noalias %p) {
    call void @foo(ptr %p)
    load i32, ptr %p
    ret void
}

; Tests that readnone noalias implies !Mod.
;
; CHECK-LABEL: Function: readnone_noalias
; CHECK: Just Ref: Ptr: i32* %p <->  call void @foo(ptr %p)
define void @readnone_noalias(ptr readnone noalias %p) {
    call void @foo(ptr %p)
    load i32, ptr %p
    ret void
}

; Tests that writeonly noalias doesn't imply !Ref (since it's still possible
; to read from the object through other pointers if the pointer wasn't
; written).
;
; CHECK-LABEL: Function: writeonly_noalias
; CHECK: Both ModRef: Ptr: i32* %p <->  call void @foo(ptr %p)
define void @writeonly_noalias(ptr writeonly noalias %p) {
    call void @foo(ptr %p)
    load i32, ptr %p
    ret void
}

; Tests that readonly doesn't imply !Mod without noalias.
;
; CHECK-LABEL: Function: just_readonly
; CHECK: Both ModRef: Ptr: i32* %p <->  call void @foo(ptr %p)
define void @just_readonly(ptr readonly %p) {
    call void @foo(ptr %p)
    load i32, ptr %p
    ret void
}

; Tests that readnone doesn't imply !Mod without noalias.
;
; CHECK-LABEL: Function: just_readnone
; CHECK: Both ModRef: Ptr: i32* %p <->  call void @foo(ptr %p)
define void @just_readnone(ptr readnone %p) {
    call void @foo(ptr %p)
    load i32, ptr %p
    ret void
}

; Tests that writeonly doesn't imply !Ref.
;
; CHECK-LABEL: Function: just_writeonly
; CHECK: Both ModRef: Ptr: i32* %p <->  call void @foo(ptr %p)
define void @just_writeonly(ptr writeonly %p) {
    call void @foo(ptr %p)
    load i32, ptr %p
    ret void
}
