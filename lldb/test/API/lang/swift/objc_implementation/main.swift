@objc @implementation
extension Gadget {
    var integer: Int = 15
    var boolean: Bool = true
    var object: NSObject = NSObject()
    var string: String = "Ace"
    var stringObject: NSObject = "Joker" as NSString
}

func main() {
    let g = Gadget()
    print("break here")
}

main()
