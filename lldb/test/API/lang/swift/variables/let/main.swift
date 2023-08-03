func use<T>(_ t : T) {}

let global_constant = 1111
var global_variable = 2222

func f(_ parameter_constant : Int, _ parameter_variable : inout Int) {
  let local_constant = parameter_constant
  var local_variable = parameter_variable
  use((local_constant, local_variable)) // break here
}

f(global_constant, &global_variable)
