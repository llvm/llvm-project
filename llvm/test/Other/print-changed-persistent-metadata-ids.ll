; RUN: opt -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=quiet -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHANGED
; RUN: opt -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-before=instsimplify -print-after=instsimplify \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=STABLE
; RUN: opt -passes='function(instsimplify),globaldce' -filter-print-funcs=second \
; RUN:   -print-changed=quiet -print-module-scope -disable-output < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CROSS-KIND
; RUN: opt -passes='function(instsimplify),print' -disable-output < %s 2> %t
; RUN: FileCheck %s --check-prefix=SPARSE < %t
; RUN: opt -disable-output < %t
; RUN: opt -S -passes='function(instsimplify)' < %s \
; RUN:   | FileCheck %s --check-prefix=COMPACT

declare i32 @opaque(i32)

@dead = internal global i32 0

define i32 @first(i32 %arg) #0 {
  %keep = call i32 @opaque(i32 %arg)
  ret i32 %keep
}

define i32 @second(i32 %arg) #1 {
  %constant = add i32 2, 3, !annotation !0
  %keep = call i32 @opaque(i32 %arg), !annotation !1, !other !3
  %result = add i32 %keep, %constant
  ret i32 %result
}

!0 = !{!"removed metadata"}
!1 = !{!2}
!2 = !{!"second metadata"}
!3 = !{!"other metadata"}

attributes #0 = { nounwind }
attributes #1 = { noinline }

; CHANGED: *** IR Dump After InstSimplifyPass on second ***
; CHANGED: define i32 @second(i32 %arg) #1 {
; CHANGED: %keep = call i32 @opaque(i32 %arg)
; CHANGED-SAME: !annotation ![[ANNOTATION:[0-9]+]], !other ![[OTHER:[0-9]+]]

; STABLE: *** IR Dump Before InstSimplifyPass on second ***
; STABLE: %constant = add i32 2, 3, !annotation !{{[0-9]+}}
; STABLE: %keep = call i32 @opaque(i32 %arg)
; STABLE-SAME: !annotation ![[STABLE_ANNOTATION:[0-9]+]], !other ![[STABLE_OTHER:[0-9]+]]
; STABLE: *** IR Dump After InstSimplifyPass on second ***
; STABLE-NOT: %constant
; STABLE: %keep = call i32 @opaque(i32 %arg)
; STABLE-SAME: !annotation ![[STABLE_ANNOTATION]], !other ![[STABLE_OTHER]]

; CROSS-KIND: *** IR Dump After InstSimplifyPass on second ***
; CROSS-KIND: @dead = internal global i32 0
; CROSS-KIND: define i32 @second(i32 %arg) #1 {
; CROSS-KIND: %keep = call i32 @opaque(i32 %arg)
; CROSS-KIND-SAME: !annotation ![[CROSS_ANNOTATION:[0-9]+]], !other ![[CROSS_OTHER:[0-9]+]]
; CROSS-KIND: *** IR Dump After GlobalDCEPass on [module] ***
; CROSS-KIND-NOT: @dead
; CROSS-KIND: define i32 @second(i32 %arg) #1 {
; CROSS-KIND: %keep = call i32 @opaque(i32 %arg)
; CROSS-KIND-SAME: !annotation ![[CROSS_ANNOTATION]], !other ![[CROSS_OTHER]]

; SPARSE: define i32 @second(i32 %arg) #1 {
; SPARSE: %keep = call i32 @opaque(i32 %arg)
; SPARSE-SAME: !annotation ![[SPARSE_ANNOTATION:[1-9][0-9]*]], !other ![[SPARSE_OTHER:[0-9]+]]
; SPARSE-NOT: !0 =
; SPARSE: ![[SPARSE_ANNOTATION]] = !{![[SPARSE_NESTED:[0-9]+]]}
; SPARSE-DAG: ![[SPARSE_NESTED]] = !{!"second metadata"}
; SPARSE-DAG: ![[SPARSE_OTHER]] = !{!"other metadata"}

; COMPACT: define i32 @second(i32 %arg) #1 {
; COMPACT: %keep = call i32 @opaque(i32 %arg)
; COMPACT-SAME: !annotation !0, !other !2
; COMPACT: !0 = !{!1}
; COMPACT: !1 = !{!"second metadata"}
; COMPACT: !2 = !{!"other metadata"}
