func getTimestamp(x: Int) async -> Int {
  return 40 + x
}

func work() {}

func foo() async {
  do {
    work() // BREAK_NOTHROW
    async let timestamp1 = getTimestamp(x:1)
    work()
    async let timestamp2 = getTimestamp(x:2)
    work()
    let timestamps = await [timestamp1, timestamp2]
    print(timestamps)
  }
  async let timestamp3 = getTimestamp(x:3)
  work()
  let actual_timestamp3 = await timestamp3
  print(actual_timestamp3)
}

struct NegativeInputError: Error {}
func getTimestamp_throwing(x: Int) async throws -> Int {
  if (x < 0) {
    throw NegativeInputError()
  }
  return 40 + x
}

func foo_throwing() async {
  do {
    work() // BREAK_THROW
    async let timestamp1 = getTimestamp_throwing(x:1)
    work()
    async let timestamp2 = getTimestamp_throwing(x:2)
    work()
    let timestamps = try await [timestamp1, timestamp2]
    print(timestamps)
  } catch {print(error)}
  work()
}

@main enum entry {
  static func main() async {
    await foo()
    await foo_throwing()
  }
}
