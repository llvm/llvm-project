@fp = external global ptr

; Keep bar as a declaration to model a target defined by a shared library.
declare i32 @bar(i32)

define i32 @foo(i32 %x) {
entry:
  %target = load ptr, ptr @fp
  %result = call i32 %target(i32 %x)
  ret i32 %result
}
