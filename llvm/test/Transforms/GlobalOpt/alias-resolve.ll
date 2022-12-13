; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@foo1 = alias void (), ptr @foo2
;; foo2 is dso_local and non-weak. Resolved.
; CHECK: @foo1 = alias void (), ptr @bar2

@foo2 = dso_local alias void(), ptr @bar1
;; bar1 is dso_local and non-weak. Resolved.
; CHECK: @foo2 = dso_local alias void (), ptr @bar2

@bar1  = dso_local alias void (), ptr @bar2
; CHECK: @bar1 = dso_local alias void (), ptr @bar2

@weak1 = weak dso_local alias void (), ptr @bar2
;; weak1 may be replaced with another definition in the linkage unit. Not resolved.
; CHECK: @weak1 = weak dso_local alias void (), ptr @bar2

@bar4 = private unnamed_addr constant [2 x ptr] zeroinitializer
@foo4 = weak_odr unnamed_addr alias ptr, getelementptr inbounds ([2 x ptr], ptr @bar4, i32 0, i32 1)
; CHECK: @foo4 = weak_odr unnamed_addr alias ptr, getelementptr inbounds ([2 x ptr], ptr @bar4, i32 0, i32 1)

@priva  = private alias void (), ptr @bar5
; CHECK: @priva = private alias void (), ptr @bar5

define dso_local void @bar2() {
  ret void
}
; CHECK: define dso_local void @bar2()

define weak void @bar5() {
  ret void
}
; CHECK: define weak void @bar5()

define void @baz() {
entry:
         call void @foo1()
;; foo1 is dso_preemptable. Not resolved.
; CHECK: call void @foo1()

         call void @foo2()
;; foo2 is dso_local and non-weak. Resolved.
; CHECK: call void @bar2()

         call void @bar1()
;; bar1 is dso_local and non-weak. Resolved.
; CHECK: call void @bar2()

         call void @weak1()
;; weak1 is weak. Not resolved.
; CHECK: call void @weak1()

         call void @priva()
;; priva has a local linkage. Resolved.
; CHECK: call void @priva()

         ret void
}

@foo3 = dso_local alias void (), ptr @bar3
; CHECK-NOT: bar3

define internal void @bar3() {
  ret void
}
;CHECK: define dso_local void @foo3
