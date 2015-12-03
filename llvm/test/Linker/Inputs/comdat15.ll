$a1 = comdat any
@baz = private global i32 42, comdat($a1)
@a1 = internal alias i32, i32* @baz
define i32* @abc() {
  ret i32* @a1
}
