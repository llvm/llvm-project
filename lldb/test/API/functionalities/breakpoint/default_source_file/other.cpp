// A function whose base name is also "main". Being a real debug function (not
// just a symbol), it competes with the entry point in a base-name lookup and
// must not be mistaken for it when choosing the default source file.
namespace ns {
int main() {
  int value = 1;
  return value; // ns::main body
}
} // namespace ns
