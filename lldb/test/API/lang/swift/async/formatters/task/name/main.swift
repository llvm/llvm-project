@main struct Main {
    static func main() async {
        let task = Task(name: "Chore") {
            print("break inside")
            _ = readLine()
            return 15
        }
        print("break outside")
        _ = await task.value
    }
}
