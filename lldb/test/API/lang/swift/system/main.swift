import System

@main enum Entry {
    static func main() {
        let path = FilePath("/usr/local/bin")
        let empty = FilePath("")
        let root = FilePath("/")
        print("break here", path, empty, root)
    }
}
