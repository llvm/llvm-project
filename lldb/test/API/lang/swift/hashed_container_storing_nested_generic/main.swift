public class A<T> {
    public func addValue() {
        self.dict["key"] = B()
    }
    private class B {}
    private var dict: [String: B] = [:]
    func test() {
        let val = dict["key"]!
        print(1) // break here
    }
}

let foo = A<Int>()
foo.addValue()
foo.test()

