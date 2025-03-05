@main struct Main {
  static func main() async {
    await withUnsafeContinuation { (cont: UnsafeContinuation<Void, Never>) in
      print("break here")
      cont.resume()
    }
  }
}
