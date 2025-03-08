// This is a test program that makes a deep stack
// so we can test unwinding from multiple threads.

void call_me(int input) {
  if (input > 1000) {
    input += 1; // Set a breakpoint here
    if (input > 1001)
      input += 1;
    return;
  } else
    call_me(++input);
}

int main() {
  call_me(0);
  return 0;
}
