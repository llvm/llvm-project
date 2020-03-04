import ObjCModule

func use<T>(_ t: T) {}

class SwiftChild : ObjCClass {
}


let pureSwift = 42
let obj = ObjCClass()
let proto = getProto()
let swiftChild = SwiftChild()

use(pureSwift) // break here
use(obj)
use(proto)
use(swiftChild)
