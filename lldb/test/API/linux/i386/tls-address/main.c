__thread int tls_value = 42;

int main(void) {
  ++tls_value;
  return tls_value; // break here
}
