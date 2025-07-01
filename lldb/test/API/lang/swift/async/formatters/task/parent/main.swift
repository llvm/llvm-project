func breakHere() {}

@main struct Main {
    static func main() async {
        await breakHere()
        async let x = breakHere()
        await x
    }
}
