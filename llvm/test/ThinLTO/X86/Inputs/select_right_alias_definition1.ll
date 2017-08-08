
@foo = weak alias i32 (...), bitcast (i32 ()* @foo1 to i32 (...)*)

define i32 @foo1() {
    ret i32 42
}