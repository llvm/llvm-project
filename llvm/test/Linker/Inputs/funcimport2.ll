define linkonce_odr hidden void @foo() {
    ret void
}

define void @bar() {
    call void @foo()
    ret void
}
