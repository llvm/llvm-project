func use<T>(_ t: T) {}

import Foundation
let s = NSString(string: "Hello MacABI")
use(s) // break here
