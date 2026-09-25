; RUN: opt -S -passes=no-op-module < %s | FileCheck %s --check-prefix=COMPACT
; RUN: opt -disable-output -passes=print < %s 2>&1 | FileCheck %s --check-prefix=PERSISTENT-MODULE
; RUN: opt -disable-output -passes=print < %s 2> %t
; RUN: opt -disable-output < %t
; RUN: opt -disable-output -passes='function(print)' -filter-print-funcs=second \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=PERSISTENT-FUNCTION
; RUN: opt -disable-output -passes='function(no-op-function)' \
; RUN:   -print-before=no-op-function -filter-print-funcs=second \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=PERSISTENT-FUNCTION
; RUN: opt -disable-output -passes='function(no-op-function)' -print-after-all \
; RUN:   -filter-print-funcs=second < %s 2>&1 | FileCheck %s --check-prefix=PERSISTENT-FUNCTION
; RUN: opt -disable-output -passes=no-op-module -print-before=no-op-module \
; RUN:   -filter-print-funcs=first,second < %s 2>&1 | FileCheck %s --check-prefix=PERSISTENT-MULTI
; RUN: opt -disable-output -passes='loop(no-op-loop)' -print-before=no-op-loop \
; RUN:   -filter-print-funcs=loop < %s 2>&1 | FileCheck %s --check-prefix=PERSISTENT-LOOP
; RUN: opt -disable-output -passes='print,function(print)' < %s 2>&1 | FileCheck %s --check-prefix=SAME-ID
$group = comdat any

@named = global ptr @0, comdat($group), !annotation !5
@0 = global i32 0
@1 = global i32 1

declare void @callee(ptr)

define void @first() #0 {
  call void @callee(ptr @0) #1
  call void @callee(ptr @1) #1
  ret void, !annotation !1
}

define void @second() {
  call void @callee(ptr @1) #1, !annotation !3
  ret void, !annotation !3
}

define void @loop() {
entry:
  call void @callee(ptr @0) #1
  call void @callee(ptr @1) #1
  br label %loop

loop:
  call void @callee(ptr @1) #1
  br i1 false, label %loop, label %exit

exit:
  ret void
}

attributes #0 = { noinline }
attributes #1 = { nounwind }

!named = !{!0}
!0 = !{!"named metadata"}
!1 = !{!2}
!2 = !{!"first metadata"}
!3 = !{!4}
!4 = !{!"second metadata"}
!5 = !{!6}
!6 = !{!"global metadata"}

; COMPACT: @named = global ptr @0, comdat($group), !annotation !0
; COMPACT: ret void, !annotation !3
; COMPACT: call void @callee(ptr @1) #1, !annotation !5
; COMPACT: ret void, !annotation !5
; COMPACT: !named = !{!2}

; PERSISTENT-MODULE: @named = global ptr @0, comdat($group), !annotation ![[GLOBAL:[0-9]+]]
; PERSISTENT-MODULE: ret void, !annotation ![[FIRST:[0-9]+]]
; PERSISTENT-MODULE: call void @callee(ptr @1) #1, !annotation ![[SECOND:[0-9]+]]
; PERSISTENT-MODULE: ret void, !annotation ![[SECOND]]
; PERSISTENT-MODULE: !named = !{![[NAMED:[0-9]+]]}

; PERSISTENT-FUNCTION: define void @second() {
; PERSISTENT-FUNCTION: call void @callee(ptr @1) #1, !annotation ![[SECOND:[0-9]+]]
; PERSISTENT-FUNCTION: ret void, !annotation ![[SECOND]]

; PERSISTENT-MULTI: define void @first() #0 {
; PERSISTENT-MULTI: call void @callee(ptr @0) #1
; PERSISTENT-MULTI: call void @callee(ptr @1) #1
; PERSISTENT-MULTI: define void @second() {
; PERSISTENT-MULTI: call void @callee(ptr @1) #1

; PERSISTENT-LOOP: ; Preheader:
; PERSISTENT-LOOP: call void @callee(ptr @0) #1
; PERSISTENT-LOOP: call void @callee(ptr @1) #1
; PERSISTENT-LOOP: ; Loop:
; PERSISTENT-LOOP: call void @callee(ptr @1) #1

; SAME-ID: define void @second() {
; SAME-ID: call void @callee(ptr @1) #1, !annotation ![[SAME_SECOND:[0-9]+]]
; SAME-ID: ![[SAME_SECOND]] = !{!{{[0-9]+}}}
; SAME-ID: define void @second() {
; SAME-ID: call void @callee(ptr @1) #1, !annotation ![[SAME_SECOND]]
