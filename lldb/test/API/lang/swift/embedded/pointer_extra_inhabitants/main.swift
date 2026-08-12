// The extra inhabitant count of a pointer-sized builtin drives how deeply an
// Optional can be nested before the layout has to grow an out-of-line
// discriminator byte. A thin function pointer ("yyXf") is not nullable, so it
// has many extra inhabitants and three nested Optionals still fit in 8 bytes.
// Builtin.RawPointer ("Bp") is nullable and null is its only extra inhabitant,
// so each additional Optional level costs a byte. Both counts come from the
// same hardcoded-builtin-descriptor path in LLDB, which makes RawHolder the
// control for FnHolder: if the thin function pointer were also reported as
// having a single extra inhabitant, FnHolder would take on RawHolder's layout.
typealias CFn = @convention(c) () -> Void

func target() {}

struct FnHolder {
    var a: CFn?
    var b: CFn??
    var c: CFn???
}

struct RawHolder {
    var a: UnsafeRawPointer?
    var b: UnsafeRawPointer??
    var c: UnsafeRawPointer???
}

@inline(never)
func blackHole<T>(_ x: T) {}

func test() {
    var pointer = UnsafeRawPointer(bitPattern: 0x1000)
    var fnHolder = FnHolder(a: target, b: target, c: target)
    var rawHolder = RawHolder(a: pointer, b: pointer, c: pointer)
    print("break here")
    blackHole(fnHolder)
    blackHole(rawHolder)
    fnHolder.a = nil
    rawHolder.a = nil
    pointer = nil
}

test()
