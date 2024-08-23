import Swift

enum E {
  @TaskLocal static var taskLocal: Int = 23
}

class C {
  func withClosure<R>(_ closure : @Sendable () async throws -> R) async rethrows -> R {
    return try await closure()
  }
  func run<R : Sendable>(_ action : @Sendable () async throws -> R) async rethrows -> R {
    return try await E.$taskLocal.withValue(42) {
      return try await action()
    }
  }
}

@main struct Main {
  static func main() async {
    try await C().run() {
      print("break here")
      return 0
    }
  }
}
