enum EnumError: Error {
    case TrivialError
    case ImportantError
}

func untyped(_ input : Int) throws -> Int {
    if input > 100 {
        throw EnumError.ImportantError
    } else if input > 10 {
        throw EnumError.TrivialError
    } else {
        return input + 2
    }
}

func typed(_ input : Int) throws(EnumError) -> Int {
    if input > 100 {
        throw EnumError.ImportantError
    } else if input > 10 {
        throw EnumError.TrivialError
    } else {
        return input + 2
    }
}

do {
    let mode = CommandLine.arguments[1]
    if mode == "untyped" {
        try untyped(101)
    } else if mode == "typed" {
        try typed(101)
    }
} catch {
    print(error)
}
