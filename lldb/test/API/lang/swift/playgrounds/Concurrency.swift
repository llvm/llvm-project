import Dispatch
// Test that the pass pipeline is set up to support concurrency.

let semaphore = DispatchSemaphore(value: 0)
var i: Int = 0

if #available(macOS 12, iOS 15, watchOS 8, tvOS 15, *) {

  actor Actor {
    func f() -> Int { return 42 }
  }

  let queue = Actor()
  Task.detached() {
    i = await queue.f()
    semaphore.signal()
  }

} else {
  // Still make the test pass if we don't have concurrency.
  i = 42
  semaphore.signal()
}

semaphore.wait()
i - 19
