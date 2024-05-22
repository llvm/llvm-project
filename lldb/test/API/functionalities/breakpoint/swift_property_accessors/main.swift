struct Thing {
    var read_only: Int { 22 + 15 }

    var read_write: Int {
        get { 23 + 41 }
        set { print("nothing") }
    }

    var observed: Int {
        willSet { print("willSet") }
        didSet { print("didSet") }
    }
}
