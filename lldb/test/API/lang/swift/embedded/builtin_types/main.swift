// A thick function (escaping closure) consists of:
// - yyXf: Thin function pointer () -> ()
// - Bo: Builtin.NativeObject (context for captures)
struct ClosureHolder {
    var callback: () -> Void
}

func doNothing() {}

func test() {
    let holder = ClosureHolder(callback: doNothing)
    print("break here")
}

test()
