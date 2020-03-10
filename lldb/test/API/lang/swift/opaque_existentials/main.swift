protocol P {}

struct S : P {
    var type = "pata"
    var stringValue = "tino"
}

let tinky : P = S()
print() //%self.expect("frame var -d run-target -- tinky", substrs=['(a.S) tinky = (type = \"pata\", stringValue = \"tino\")'])
        //%self.expect("expr -d run-target -- tinky", substrs=['(a.S) $R0 = (type = \"pata\", stringValue = \"tino\")'])
