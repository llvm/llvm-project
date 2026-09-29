
@foo = weak alias i32 (...), @foo1

define i32 @foo1() {
    ret i32 42
}