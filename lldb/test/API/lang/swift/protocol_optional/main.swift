protocol Key {
    associatedtype Value
}

struct Key1: Key {
    typealias Value = Int?
}

struct KeyTransformer<K1: Key> {
    let input: K1.Value

    func printOutput() {
        let patatino = input
        print(patatino) //%self.expect('frame variable -d run-target -- patatino', substrs=['(Int?) patatino = 5'])
                        //%self.expect('expr -d run-target -- patatino', substrs=['(Int?) $R0 = 5'])
    }
}

var xformer: KeyTransformer<Key1> = KeyTransformer(input: 5)
xformer.printOutput()
