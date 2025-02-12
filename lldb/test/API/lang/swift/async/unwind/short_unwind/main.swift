func work() async -> Int {
  try? await Task.sleep(for: .seconds(3))  // BREAK HERE
  return 10
}
func foo_should_step_over() async -> Int {
  async let an_array = [work(), work(), work()]
  await an_array
  return 10
}

@main struct Main {
  static func main() async {
    await foo_should_step_over()
  }
}
