; RUN: opt %s -passes=inline -S | FileCheck %s

define internal void @innerSmall() "stack-probe-size"="4096" {
  ret void
}

define internal void @innerLarge() "stack-probe-size"="8192" {
  ret void
}

define void @outerNoAttribute() {
  call void @innerSmall()
  ret void
}

define void @outerConflictingAttributeSmall() "stack-probe-size"="4096" {
  call void @innerLarge()
  ret void
}

define void @outerConflictingAttributeLarge() "stack-probe-size"="8192" {
  call void @innerSmall()
  ret void
}

; CHECK: define void @outerNoAttribute() #0
; CHECK: define void @outerConflictingAttributeSmall() #0
; CHECK: define void @outerConflictingAttributeLarge() #0
; CHECK: attributes #0 = { "stack-probe-size"="4096" }
