import Foundation

struct Document {
    var kind: Kind
    var path: String

    enum Kind {
        case binary
        case text
    }
}

@objc(App)
class App : NSObject {
    var name: String = "Debugger"
    var version: (Int, Int) = (1, 0)
    var recentDocuments: [Document]? = [
        Document(kind: .binary, path: "/path/to/something"),
    ]
}
