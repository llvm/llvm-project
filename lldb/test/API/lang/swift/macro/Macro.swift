@freestanding(expression) public macro stringify<T>(_ value: T) -> (T, String) = #externalMacro(module: "MacroImpl", type: "StringifyMacro")

@freestanding(expression) public macro no_return<T>(_ value: T) = #externalMacro(module: "MacroImpl", type: "NoReturnMacro")
