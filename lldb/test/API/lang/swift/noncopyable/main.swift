struct S: ~Copyable {
  let i = 42
}

func f() {
  let s = S()
  print("break here")
}

f()
