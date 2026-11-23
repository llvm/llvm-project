namespace N0 {
namespace N1 {

char *buf0 = nullptr;
char buf1[] = {0, 1, 2, 3, 4, 5, 6, 7};

char sum(char *buf, int size) {
  char result = 0;
  for (int i = 0; i < size; i++)
    result += buf[i];
  return result;
}

} // namespace N1
} // namespace N0

int main() {
  char result = N0::N1::sum(N0::N1::buf1, sizeof(N0::N1::buf1));
  return 0;
}
