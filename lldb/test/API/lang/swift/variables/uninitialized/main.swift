struct A<T> {
    let d : Int
    var cs : [T]

    func process(i: Int, k1: T?, k2: T?, k3: T?) -> T? {
        switch (i) {
        case 0: return k1
        case 1: return k2
        case 2: return k3
        default: return nil
        }
    }

    func b(t: T) -> T? {
        return t
    }

    mutating func a() -> T? {
        var adict : [String: Any]  //% self.runCmd("frame variable adict", msg="Frame variable of an uninitialized dict returns")
        adict = [String: Any]() 
        adict["key1"] = 1.0
        var t : T? = nil
        let c = cs[0]

        let k1 = b(t:c)
        let k2 = b(t:c) //% self.expect("expr -d run -- c", "Unreadable variable is ignored", substrs=["= 3"])
        let k3 = b(t:c)

        if let maybeT = process(i : adict.count, k1:k1, k2:k2, k3:k3) {
            t = maybeT
        }

        return t
    }
}

var myA = A<Int>(d: 3, cs: [3,5,7])
print(myA.a())
