; RUN: opt < %s -disable-output -passes=print-profile-summary -S 2>&1 | FileCheck %s

; The "hot" attribute is a manual alternative to PGO hotness data, so it must
; also apply when the module has no profile summary at all.

define void @hot_attr() hot {
; CHECK: hot_attr :hot entry
  ret void
}

define void @cold_attr() cold {
; CHECK: cold_attr :cold entry
  ret void
}

define void @plain() {
; CHECK: plain{{$}}
  ret void
}
