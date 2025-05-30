func work() {
  let noParams = Foo() // break here
  let oneParam = Foo(string: "Bar")
  print("done")
}

work()
