enum Event: Int {
    case idle = 0
    case start = 1
    case fault = 100
}

@inline(never)
func rawOf(_ n: Int) -> Int {
    if let e = Event(rawValue: n) { return e.rawValue }
    return -1
}

@inline(never)
func show(_ ev: Event) {
    let s = StaticString("break here")
    print(s) // break here
    print(ev.rawValue)
}

func f() {
    print(rawOf(1))
    show(.fault)
}

f()
