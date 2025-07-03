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

class MY_CLASS {
  init(input: [Int]) {
    let find_me = "hello"
    let _ = input.map {
      return $0  // break_ctor_class
    }
    let dont_find_me = "hello"
  }

  static func static_func(input_static: [Int]) {
    let find_me_static = "hello"
    let _ = input_static.map {
      return $0  // break_static_member_class
    }
    let dont_find_me_static = "hello"
  }

  public var class_computed_property: Int {
    get {
      let find_me = "hello"
      let _ = {
        print("break_class_computed_property_getter")
        return 10
      }()
      let dont_find_me = "hello"
      return 42
    }
    set {
      let find_me = "hello"
      let _ = {
        print("break_class_computed_property_setter")
        return 10
      }()
      let dont_find_me = "hello"
    }
  }
}

struct MY_STRUCT {
  init(input: [Int]) {
    let find_me = "hello"
    let _ = input.map {
      print("break_ctor_struct")
      return $0
    }
    let dont_find_me = "hello"
  }

  static func static_func(input_static: [Int]) {
    let find_me_static = "hello"
    let _ = input_static.map {
      print("break_static_member_struct")
      return $0
    }
    let dont_find_me_static = "hello"
  }

  public var struct_computed_property: Int {
    get {
      let find_me = "hello"
      let _ = {
        print("break_struct_computed_property_getter")
        return 10
      }()
      let dont_find_me = "hello"
      return 42
    }
    set {
      let find_me = "hello"
      let _ = {
        print("break_struct_computed_property_setter")
        return 10
      }()
      let dont_find_me = "hello"
    }
  }
}

enum MY_ENUM {
  case case1(Double)
  case case2(Double)

  init(input: [Int]) {
    let find_me = "hello"
    let _ = input.map {
      print("break_ctor_enum")
      return $0
    }

    let dont_find_me = "hello"
    self = .case1(42.0)
  }

  static func static_func(input_static: [Int]) {
    let find_me_static = "hello"
    let _ = input_static.map {
      print("break_static_member_enum")
      return $0
    }
    let dont_find_me_static = "hello"
  }
}

func_1(arg: 42)
func_2(arg: 42)
await func_3(arg: 42)
var my_class = MY_CLASS(input: [1, 2])
MY_CLASS.static_func(input_static: [42])
print(my_class.class_computed_property)
my_class.class_computed_property = 10
var my_struct = MY_STRUCT(input: [1, 2])
MY_STRUCT.static_func(input_static: [42])
print(my_struct.struct_computed_property)
my_struct.struct_computed_property = 10
let _ = MY_ENUM(input: [1,2])
MY_ENUM.static_func(input_static: [42])
