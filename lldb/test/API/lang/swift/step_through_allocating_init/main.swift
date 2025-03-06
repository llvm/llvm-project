class Foo {
    init() {
      let some_string = "foo"
      print(some_string)
    }
}

func bar(_ _: Foo) {
    print("bar")
}

func doSomething()
{
  let f = Foo() // Break here to step into init
  bar(f)
}

doSomething()
