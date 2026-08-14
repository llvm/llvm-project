import Foundation

func f() {
    // A large, non-ASCII string literal bridges to a __SharedStringStorage
    // instance.
    let ns = "café — a long enough non-ASCII shared-storage NSString value" as NSString
    let cf = "alçada — a long enough non-ASCII shared-storage CFString value" as CFString
    print("break here")
}
f()

