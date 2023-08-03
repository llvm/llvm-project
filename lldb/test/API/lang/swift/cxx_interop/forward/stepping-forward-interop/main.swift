import CxxFunctions

func testFunction() {
  cxxFunction() // Break here for function
}

func testMethod() {
  var cxxClass = CxxClass()
  cxxClass.cxxMethod() // Break here for method
}

func testContructor() {
  //FIXME: remove this statement rdar://105569287
  print(1) // Break here for constructor
  var classWithConstructor = ClassWithConstructor(4, true, 5.7); 
}

protocol P {
  mutating func definedInExtension() -> Int32
}

extension ClassWithExtension: P {}

func testClassWithExtension() {
  var classWithExtension: P = ClassWithExtension()
  classWithExtension.definedInExtension() // Break here for extension
}


func testCallOperator() {
  var classWithCallOperator = ClassWithCallOperator()
  classWithCallOperator() // Break here for call operator
}


testFunction()
testMethod()
testContructor()
testClassWithExtension()
testCallOperator()
