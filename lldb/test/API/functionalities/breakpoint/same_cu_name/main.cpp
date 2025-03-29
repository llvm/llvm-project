namespace ns1 {
extern void DoSomeStuff();
}

namespace ns2 {
extern void DoSomeStuff();
}

namespace ns3 {
extern void DoSomeStuff();
}

namespace ns4 {
extern void DoSomeStuff();
}

int main(int argc, char *argv[]) {
  ns1::DoSomeStuff();
  ns2::DoSomeStuff();
  ns3::DoSomeStuff();
  ns4::DoSomeStuff();

  return 0;
}
