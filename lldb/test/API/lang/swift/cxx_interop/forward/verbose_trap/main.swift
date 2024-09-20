import Aborts

func takes(_ t: some UnsafeCxxInputIterator) {
    t.pointee
}

func main() {
  var x = std.ConstIterator(137);
  takes(x);
  print("Break here");
}

main()

