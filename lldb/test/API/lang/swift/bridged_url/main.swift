import Foundation

@main enum Entry {
    static func main() {
        let nsurl = NSURL(fileURLWithPath: "/tmp")
        let url = nsurl as URL
        print("break here", url)
    }
}
