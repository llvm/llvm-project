func main() {
  // This type is hidden due to the frontend flag in Makefile.
  typealias Pair<T> = (T, T)
  let invisible : Pair<Int> = (1, 2)
  print("break here \(invisible)")
}

main()
