enum MyError: Error {
  case MyError1
  case MyError2
}

func willthrow(_ arg: Int) async throws {
  if arg == 1 { throw MyError.MyError1 } else { throw MyError.MyError2 }
}

func foo(_ argument: Int) async {
  do {
    switch argument {
    case 1:
      try await willthrow(1)
    case 2:
      try await willthrow(2)
    default:
      return
    }
  } catch {
    print("breakhere")
  }
}

print("breakpoint_start")
await foo(1)
await foo(2)
