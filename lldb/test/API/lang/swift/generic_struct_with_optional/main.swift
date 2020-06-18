import Foundation

enum CustomError {
  case err1
}

struct StructWithGenericContents<A, B> {
  var field1 : A
  var field2 : B
}

let EmptyData = Foundation.Data()

let OptionalEmptyData = Swift.Optional<Foundation.Data>.some(EmptyData)

let StructWithGenericEnumContents = StructWithGenericContents(field1: OptionalEmptyData, field2: CustomError.err1)

func onError(_ handler: (StructWithGenericContents<Foundation.Data?, CustomError>) -> Void) {
  handler(StructWithGenericEnumContents)
}

onError({e in
  print(e) //% self.expect("expr -O -- e", substrs=["StructWithGenericContents<Optional<Data>, CustomError>"])
  print(e) //% self.expect("frame var -d run-target -- e", substrs=["field1 = 0 bytes", "field2 = err1"])
}) 
