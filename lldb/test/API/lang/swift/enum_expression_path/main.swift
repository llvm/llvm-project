struct Leaf { var a: Int; var b: Int }
struct Mid { var leaf: Leaf }
enum Box { case empty; case boxed(Mid) }
struct Outer { var box: Box }

final class Holder {
    var outer = Outer(box: .boxed(Mid(leaf: Leaf(a: 1, b: 2))))
    func run() {
        print("break here")
        outer.box = .boxed(Mid(leaf: Leaf(a: 99, b: 3)))
        if case .boxed(let mid) = outer.box {
            print(mid.leaf.a)
        }
    }
}

Holder().run()
