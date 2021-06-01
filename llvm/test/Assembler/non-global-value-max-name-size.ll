; RUN: opt < %s -S -non-global-value-max-name-size=4
; Test that local value name lookup works if the name is capped

define void @f() {
bb0:
  br label %testz

testz:
  br label %testz
}
