$c = comdat any

@v2 = weak dllexport global i32 0, comdat ($c)
define i32* @f2() {
  ret i32* @v2
}

@v3 = weak alias i32, i32* @v2
define i32* @f3() {
  ret i32* @v3
}

