func foo<T>(fn: () -> T) {
    // break here for generic type
   let result = fn() 
   print(result)
}

func bar(fn: () -> Int) {
    // break here for static type
    foo(fn: fn)
}

bar(fn: { 
    print("I am about to return 3") ; return 3 })
