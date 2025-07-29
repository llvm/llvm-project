func syncFunc() {
  print("break here")
}

func callSyncFunc() async {
  syncFunc()
}

@main struct Main {
  static func main() async {
    await callSyncFunc()
  }
}
