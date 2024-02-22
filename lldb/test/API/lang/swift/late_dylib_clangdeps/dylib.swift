import ClangMod

@_silgen_name("f") public func f() {
  let x = FromClang(x: 42)
  print(x) // line 5
}
