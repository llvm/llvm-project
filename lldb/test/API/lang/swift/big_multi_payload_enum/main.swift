public enum A6 {
  case item1
  case item2
}
public enum A7 {
  case item1
  case item2
}

public enum Request {
  case a1
  case a2
  case a3
  case a4
  case a5
  case a6(A6)
  case a7(A7)
}

func f() {
  let a1 = Request.a1
  let a2 = Request.a2
  let a3 = Request.a3
  let a4 = Request.a4
  let a5 = Request.a5
  let a6_item1 = Request.a6(.item1)
  let a6_item2 = Request.a6(.item2)
  let a7_item1 = Request.a7(.item1)
  let a7_item2 = Request.a7(.item2)
  print("break here")
}

f()
