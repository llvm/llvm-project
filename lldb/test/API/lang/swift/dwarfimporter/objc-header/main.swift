import ObjCModule

func use<T>(_ t : T) {}

guard let obj = ObjCClass() else { exit(1) }
let myobj = obj
let mangled = obj.getMangled()!
let rawname = obj.getRawname()!
let s = obj.getString()
use(s) // break here
