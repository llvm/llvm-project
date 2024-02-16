func main() {
  typealias Pair<T> = (T, T)
  let patatino : Pair<Int> = (1, 2)
  print(patatino) //%self.expect('expr -d run -- patatino', substrs=['Pair<Int>'])
}

main()
