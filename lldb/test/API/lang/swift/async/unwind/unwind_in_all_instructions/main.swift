func async_work() async {
  print("async working")
}
func work() {
  print("working")
}

func ASYNC___1___(cond: Int) async -> Int {
  work()
  await async_work()
  for i in 1...3 {
    work()
    await async_work()
    work()
    if (cond + i == 3) {
      await async_work()
      print("exiting loop here!")
      break;
    }
  }
  return 0
}

func ASYNC___2___() async -> Int {
  let result = await ASYNC___1___(cond: 1)
  return result
}

func ASYNC___3___() async -> Int {
  let result = await ASYNC___2___()
  return result
}

func ASYNC___4___() async -> Int {
  let result = await ASYNC___3___()
  return result
}

func ASYNC___5___() async -> Int {
  let result = await ASYNC___4___()
  return result
}

@main struct Main {
  static func main() async {
    let result = await ASYNC___5___() // BREAK HERE
    print(result)
  }
}
