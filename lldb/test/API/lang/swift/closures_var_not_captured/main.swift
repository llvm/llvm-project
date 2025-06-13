func func_1(arg: Int) {
  let var_in_foo = "Alice"
  do {
    let dont_find_me = "haha"
    print(dont_find_me)
  }
  let simple_closure = {
    print("Hi there!")  // break_simple_closure
  }
  simple_closure()
  let dont_find_me = "haha"
  print(dont_find_me)
}

func func_2(arg: Int) {
  do {
    let dont_find_me = "haha"
    print(dont_find_me)
  }
  let var_in_foo = "Alice"
  let shadowed_var = "shadow"
  let outer_closure = {
    do {
      let dont_find_me = "haha"
      print(dont_find_me)
    }
    let var_in_outer_closure = "Alice"
    let shadowed_var = "shadow2"
    let inner_closure_1 = {
      print("Hi inside!")  // break_double_closure_1
    }
    inner_closure_1()

    do {
      let dont_find_me = "haha"
      print(dont_find_me)
    }
    let inner_closure_2 = {
      print("Hi inside!")  // break_double_closure_2
    }
    inner_closure_2()
    let dont_find_me = "haha"
    print(dont_find_me)
  }
  let dont_find_me = "haha"
  print(dont_find_me)
  outer_closure()
}

func func_3(arg: Int) async {
  do {
    let dont_find_me = "haha"
    print(dont_find_me)
  }
  let var_in_foo = "Alice"

  // FIXME: if we comment the line below, the test fails. For some reason,
  // without this line, most variables don't have debug info in the entry
  // funclet, which is the "parent name" derived from the closure name.
  // rdar://152271048
  try! await Task.sleep(for: .seconds(0))

  let outer_closure = {
    do {
      let dont_find_me = "haha"
      print(dont_find_me)
    }
    let var_in_outer_closure = "Alice"

    let inner_closure_1 = {
      print("Hi inside!")  // break_async_closure_1
    }
    inner_closure_1()

    try await Task.sleep(for: .seconds(0))

    let inner_closure_2 = {
      print("Hi inside!")  // break_async_closure_2
    }
    inner_closure_2()
    let dont_find_me = "haha"
    print(dont_find_me)
  }

  try! await outer_closure()
  let dont_find_me = "haha"
  print(dont_find_me)
}

func_1(arg: 42)
func_2(arg: 42)
await func_3(arg: 42)
