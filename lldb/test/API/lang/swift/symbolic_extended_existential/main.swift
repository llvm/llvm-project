class C {}

struct S {
  let l0 = 0
  let s1 : any Sequence<Int> = 1...1
  let l1 = 10
  let s2 : any Sequence<Int> = 1...200
  let l2 = 20
  let s3 : (any Sequence<Int>)? = 1...2
  let l3 = 30
  let s4 : (any Sequence<Int>)? = nil
  let l4 = 40
  let s5 : any AnyObject = C()
  let l5 = 50
  let s6 : any Any.Type = Int.self
  let l6 = 60
}

func main() {
  var s = S()
  print("break here")
}

main()

