; RUN: opt -passes="ipsccp<func-spec>" -funcspec-min-function-size=3 -S < %s \
; RUN: | FileCheck %s --implicit-check-not=specialized

; Check the attribute "no-func-spec" blocks function specialization. 'do_spec'
; gets specialized but 'no_spec' which is identical except for the attribute
; shouldn't (--implicit-check-not).

; CHECK: %call = tail call i32 @do_spec.specialized.1(i32 noundef 0)
; CHECK: define internal i32 @do_spec.specialized.1(i32 noundef %a)

declare i32 @ext(i32 noundef)

define hidden i32 @do_spec(i32 noundef %a) {
entry:
  %call = tail call i32 @ext(i32 noundef %a)
  ret i32 %call
}

define hidden void @a() {
entry:
  %call = tail call i32 @do_spec(i32 noundef 0)
  ret void
}

define hidden i32 @no_spec(i32 noundef %a) #0 {
entry:
  %call = tail call i32 @ext(i32 noundef %a)
  ret i32 %call
}

define hidden void @b() {
entry:
  %call = tail call i32 @no_spec(i32 noundef 0)
  ret void
}

attributes #0 = { "no-func-spec" }
