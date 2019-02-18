; RUN: opt %s -inline -S | FileCheck %s

define internal void @innerSmall() "min-legal-vector-width"="128" {
  ret void
}

define internal void @innerLarge() "min-legal-vector-width"="512" {
  ret void
}

define internal void @innerNoAttribute() {
  ret void
}

; We should not add an attribute during inlining. No attribute means unknown.
; Inlining doesn't change the fact that we don't know anything about this
; function.
define void @outerNoAttribute() {
  call void @innerLarge()
  ret void
}

define void @outerConflictingAttributeSmall() "min-legal-vector-width"="128" {
  call void @innerLarge()
  ret void
}

define void @outerConflictingAttributeLarge() "min-legal-vector-width"="512" {
  call void @innerSmall()
  ret void
}

; We should remove the attribute after inlining since the callee's
; vector width requirements are unknown.
define void @outerAttribute() "min-legal-vector-width"="128" {
  call void @innerNoAttribute()
  ret void
}

; CHECK: define void @outerNoAttribute() {
; CHECK: define void @outerConflictingAttributeSmall() #0
; CHECK: define void @outerConflictingAttributeLarge() #0
; CHECK: define void @outerAttribute() {
; CHECK: attributes #0 = { "min-legal-vector-width"="512" }
