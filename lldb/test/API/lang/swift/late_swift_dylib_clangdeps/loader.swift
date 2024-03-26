import Darwin

// This lookup should fail in debugger.
let fromClang = FromClang(x: 23)
print(fromClang) // break here

// Now load the dylib.
let arg0 = CommandLine.arguments[0]
let dylibName = arg0.replacing(/a\.out$/, with: "dylib.dylib")
let dylib = dlopen(dylibName, Darwin.RTLD_NOW)
let fsym = dlsym(dylib!, "f")
typealias voidTy = @convention(c) () -> ()
let f = unsafeBitCast(fsym, to: voidTy.self)
f()
