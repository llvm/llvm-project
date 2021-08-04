import _Concurrency
import Darwin
// Test that the pass pipeline is set up to support concurrency.
var i: Int = 0

if #available(macOS 12, iOS 15, watchOS 8, tvOS 15, *) {

  actor Actor {
    func f() -> Int { return 42 }
  }

  let queue = Actor()
  async {
    i = await queue.f()
  }

} else {
  // Still make the test pass if we don't have concurrency.
  i = 42
}

while (i == 0) { sleep(1) }
i - 19
