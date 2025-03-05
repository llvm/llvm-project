@main struct Main {
  static func main() async {
    await withUnsafeContinuation { (cont: UnsafeContinuation<Void, Never>) in
      print("break unsafe continuation")
      cont.resume()
    }

    _ = await withCheckedContinuation { (cont: CheckedContinuation<Int, Never>) in
      print("break checked continuation")
      cont.resume(returning: 15)
    }
  }
}
