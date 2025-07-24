import Foundation

@main struct Entry {
    static func main() {
        let s = "one.two.three"
        let d = NSMutableDictionary()
        d["key"] = s.replacingOccurrences(of: "two", with: "2")
        print("break here")
    }
}
