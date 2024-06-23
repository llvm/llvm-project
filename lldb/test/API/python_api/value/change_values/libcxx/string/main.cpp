#include <string>

using namespace std;

int main() {
  // Currently changing value for string requires
  // string's operator= to be in debug executable
  string s;
  string l;
  wstring ws;
  wstring wl;
  u16string u16s;
  u32string u32s;
  s = "small";
  l = "looooooooooooooooooooooooooooooooong";
  ws = L"wsmall";
  wl = L"wlooooooooooooooooooooooooooooooooong";
  u16s = u"small";
  u32s = U"looooooooooooooooooooooooooooooooong";
  return 0; // Set break point at this line.
}
