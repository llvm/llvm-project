import ClangMod

@_silgen_name("f") public func f() {
  let fromClang = FromClang(x: 42)
  print(fromClang) // line 5
}
