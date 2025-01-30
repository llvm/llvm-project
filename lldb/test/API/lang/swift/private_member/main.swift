fileprivate struct U {
  let i = 23
}

fileprivate struct V {
  let member = U()
}

func main() {
  let x = V()
  print(x) // break here
}

main()
